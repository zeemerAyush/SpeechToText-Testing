import re
from types import SimpleNamespace

import pytest

from worker.pipeline.audio_clean import StageError
from worker.pipeline.redact import (
    Entity,
    PresidioAnalyzer,
    Redactor,
    apply_redactions,
    is_specific_date,
    load_presidio_engine,
    merge_entities,
    rule_entities,
)

rules = Redactor([rule_entities])


def red(text: str) -> str:
    return rules.redact(text)


# ---------- identifiers the rules must catch ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("My phone number is 555-014-2237.", "My phone number is [PHONE_NUMBER]."),
        ("Call me on (312) 555 0199 please", "Call me on [PHONE_NUMBER] please"),
        ("It is 555-0142 at home", "It is [PHONE_NUMBER] at home"),
        ("Write to jane.doe@example.com today", "Write to [EMAIL_ADDRESS] today"),
        ("it is john at gmail dot com", "it is [EMAIL_ADDRESS]"),
        ("SSN is 123-45-6789", "SSN is [US_SSN]"),
        ("My MRN is 4471829.", "My MRN is [MEDICAL_RECORD_NUMBER]."),
        ("medical record number is A12345 thanks", "medical record number is [MEDICAL_RECORD_NUMBER] thanks"),
        ("insurance id 99-1234567", "insurance id [MEDICAL_RECORD_NUMBER]"),
        ("I live at 123 Main Street now", "I live at [LOCATION] now"),
        ("born on March 5th, 1980 in the morning", "born on [DATE_TIME] in the morning"),
        ("birthday 12/05/1980", "birthday [DATE_TIME]"),
        ("admitted 2026-11-26 evening", "admitted [DATE_TIME] evening"),
        ("on the 5th of March", "on the [DATE_TIME]"),
        ("seen on May 12, 2025", "seen on [DATE_TIME]"),
    ],
)
def test_rules_redact_identifiers(text, expected):
    assert red(text) == expected


def test_spoken_digit_runs():
    ten = "my number is five five five zero one four two two three seven ok"
    assert red(ten) == "my number is [PHONE_NUMBER] ok"
    six = "the id is one two three four five six ok"
    assert red(six) == "the id is [ID_NUMBER] ok"
    assert red("It is Five Five Five Zero One Four Two Two Three Seven") == "It is [PHONE_NUMBER]"


def test_spoken_digits_after_a_record_cue_use_the_record_tag():
    assert red("chart number is one two three four five six.") == "chart number is [MEDICAL_RECORD_NUMBER]."


def test_short_digit_runs_are_clinical_not_identifiers():
    assert red("take five five milligrams") == "take five five milligrams"
    assert red("one two three four five") == "one two three four five"


# ---------- clinical language must survive ----------

@pytest.mark.parametrize(
    "text",
    [
        "We'll remove the stitches next week.",
        "Avoid lifting anything heavy for another two weeks.",
        "Take 10 milligrams twice a day for 14 days.",
        "Blood pressure is 120 over 80 and heart rate 72.",
        "He is 45 years old and weighs 180 pounds.",
        "The surgery was in 1985.",
        "Come back in 2 days, around 3:30 in the afternoon.",
        "Two sutures were left in place after the 8 hour drive.",
        "The patient may 5 times a day need rest.",
        "It may 10 days to heal, so we may 3 times review it.",
    ],
)
def test_clinical_text_is_left_alone(text):
    assert red(text) == text


# ---------- merging / replacement ----------

def test_overlap_resolved_by_score_then_length():
    text = "call 555-014-2237 now"
    ents = [Entity(5, 17, "ID_NUMBER", 0.6), Entity(5, 17, "PHONE_NUMBER", 0.9)]
    assert apply_redactions(text, merge_entities(ents, text)) == "call [PHONE_NUMBER] now"


def test_adjacent_same_type_entities_become_one_tag():
    text = "this is Maria Lopez speaking"
    ents = [Entity(8, 13, "PERSON"), Entity(14, 19, "PERSON")]
    assert apply_redactions(text, merge_entities(ents, text)) == "this is [PERSON] speaking"


def test_different_types_stay_separate_and_empty_spans_are_ignored():
    text = "Maria in Chicago"
    ents = [Entity(0, 5, "PERSON"), Entity(9, 16, "LOCATION"), Entity(6, 6, "PERSON")]
    assert apply_redactions(text, merge_entities(ents, text)) == "[PERSON] in [LOCATION]"


def test_nothing_to_redact_returns_text_unchanged():
    assert red("") == ""
    assert red("Good morning, how are you feeling?") == "Good morning, how are you feeling?"


# ---------- analyzers combined + allow list ----------

def names(*words):
    """A stand-in for NER: tags the given words as PERSON."""
    def analyze(text):
        return [Entity(m.start(), m.end(), "PERSON", 0.85)
                for w in words for m in re.finditer(rf"\b{re.escape(w)}\b", text)]
    return analyze


def test_redactor_combines_analyzers():
    r = Redactor([rule_entities, names("Maria", "Lopez")])
    assert r.redact("I am Maria Lopez, call 555-014-2237.") == "I am [PERSON], call [PHONE_NUMBER]."


def test_eponymous_conditions_are_not_mistaken_for_people():
    r = Redactor([names("Parkinson", "Foley", "Crohn", "Smith")])
    assert r.redact("Parkinson's with a Foley catheter and Crohn disease") == \
        "Parkinson's with a Foley catheter and Crohn disease"
    assert r.redact("Dr Smith ordered it") == "Dr [PERSON] ordered it"


def test_allow_list_can_be_extended():
    r = Redactor([names("Smith")], allow_list=frozenset({"smith"}))
    assert r.redact("Dr Smith") == "Dr Smith"


# ---------- specific-date filter ----------

@pytest.mark.parametrize("span", ["March 5, 1980", "5/12", "1980", "the 3rd of June", "May 2025"])
def test_specific_dates(span):
    assert is_specific_date(span)


@pytest.mark.parametrize("span", ["next week", "two days", "yesterday", "this morning", "3:30 pm", "Monday", "you may"])
def test_relative_times_are_not_specific(span):
    assert not is_specific_date(span)


# ---------- Presidio adapter (fake engine; the real one needs spaCy) ----------

class FakeEngine:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def analyze(self, **kw):
        self.calls.append(kw)
        return self.results


def result(etype, start, end, score=0.85):
    return SimpleNamespace(entity_type=etype, start=start, end=end, score=score)


def test_presidio_adapter_converts_results_and_filters_relative_dates():
    text = "Maria Lopez said next week or March 5, 1980"
    engine = FakeEngine([
        result("PERSON", 0, 11),
        result("DATE_TIME", 17, 26),            # "next week" -> dropped
        result("DATE_TIME", 30, 43),            # "March 5, 1980" -> kept
    ])
    ents = PresidioAnalyzer(engine)(text)
    assert [(e.type, text[e.start:e.end]) for e in ents] == [("PERSON", "Maria Lopez"), ("DATE_TIME", "March 5, 1980")]
    call = engine.calls[0]
    assert call["language"] == "en" and "PERSON" in call["entities"] and call["score_threshold"] == 0.4


def test_full_redactor_with_presidio_adapter():
    text = "Maria Lopez will come next week"
    r = Redactor([rule_entities, PresidioAnalyzer(FakeEngine([result("PERSON", 0, 11), result("DATE_TIME", 22, 31)]))])
    assert r.redact(text) == "[PERSON] will come next week"


def test_missing_presidio_is_a_clear_stage_error():
    with pytest.raises(StageError, match="presidio-analyzer not installed"):
        load_presidio_engine("en_core_web_lg")
