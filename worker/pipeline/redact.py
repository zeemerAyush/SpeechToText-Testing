"""PHI redaction on transcript text (DESIGN.md 6.6). Best effort: access control stays the primary safeguard.

Two layers feed one redactor:
  * rules      - pure Python, always on. Catches what spoken transcripts need and NER misses: spoken digit
                 runs ("five five five ..."), record/insurance numbers after a cue, emails, specific dates,
                 street addresses, SSNs and long digit strings.
  * Presidio   - spaCy-based NER for names and places (plus Presidio's own pattern recognizers).
                 Imported lazily, so everything else works (and is testable) without it installed.

Tags look like [PERSON], [PHONE_NUMBER], [DATE_TIME].
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from worker.pipeline.audio_clean import StageError


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    type: str
    score: float = 0.8


Analyzer = Callable[[str], list[Entity]]

# ---------------------------------------------------------------- rule-based detection

_DIGIT_WORDS = "zero|oh|one|two|three|four|five|six|seven|eight|nine"
_DW = rf"(?:{_DIGIT_WORDS})"
_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|(?-i:May)|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"  # "may" the verb must not match
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

_SPOKEN_DIGITS = re.compile(rf"\b{_DW}(?:[\s,.-]+{_DW}){{5,}}\b", re.IGNORECASE)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SPOKEN_EMAIL = re.compile(r"\b[\w.]+\s+at\s+[\w-]+\s+dot\s+(?:com|org|net|edu|gov)\b", re.IGNORECASE)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_PHONE_LOCAL = re.compile(r"(?<!\d)\d{3}[-.\s]\d{4}(?!\d)")
_LONG_DIGITS = re.compile(r"\b\d(?:[ .-]?\d){5,}\b")

_CUE = (
    r"(?:mrn|medical record(?: number)?|record number|chart (?:number|no\.?)|patient (?:id|number)|"
    r"account (?:number|no\.?)|insurance (?:id|number)|member id|policy number)"
)
_ID_TOKEN = rf"(?:[A-Za-z]*\d[\w-]*|{_DW}\b)"
_CUED_ID = re.compile(
    rf"\b{_CUE}\b[\s:#,-]*(?:is\s+|was\s+)?({_ID_TOKEN}(?:[\s,-]+{_ID_TOKEN})*)", re.IGNORECASE
)

_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTHS}\b(?:,?\s+\d{{4}})?", re.IGNORECASE),
    re.compile(rf"\b{_MONTHS}\.?\s+(?:of\s+)?\d{{4}}\b", re.IGNORECASE),
]

# The street suffix must be capitalised ("Main Street"), so clinical phrases like "left in place" or
# "an 8 hour drive" are not mistaken for addresses.
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Court|Place|Dr|Way|Pl|Ct|Drive)\b\.?"
)


def _spoken_digit_type(span: str) -> str:
    n = len(re.findall(_DW, span, flags=re.IGNORECASE))
    return "PHONE_NUMBER" if n in (7, 10, 11) else "ID_NUMBER"


def rule_entities(text: str) -> list[Entity]:
    found: list[Entity] = []

    def add(pattern: re.Pattern, etype: str, score: float, group: int = 0) -> None:
        for m in pattern.finditer(text):
            found.append(Entity(m.start(group), m.end(group), etype, score))

    add(_EMAIL, "EMAIL_ADDRESS", 0.95)
    add(_SPOKEN_EMAIL, "EMAIL_ADDRESS", 0.7)
    add(_SSN, "US_SSN", 0.95)
    add(_PHONE, "PHONE_NUMBER", 0.9)
    add(_PHONE_LOCAL, "PHONE_NUMBER", 0.7)
    add(_LONG_DIGITS, "ID_NUMBER", 0.6)
    add(_CUED_ID, "MEDICAL_RECORD_NUMBER", 0.9, group=1)
    add(_ADDRESS, "LOCATION", 0.8)
    for pat in _DATE_PATTERNS:
        add(pat, "DATE_TIME", 0.95)
    for m in _SPOKEN_DIGITS.finditer(text):
        found.append(Entity(m.start(), m.end(), _spoken_digit_type(m.group()), 0.85))
    return found


# ---------------------------------------------------------------- filters

_YEAR = re.compile(r"\b(?:1[89]\d\d|20\d\d)\b")
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}\b")
_MONTH_WORD = re.compile(rf"\b{_MONTHS}\b", re.IGNORECASE)


def is_specific_date(span: str) -> bool:
    """True for dates that could identify someone (month/day/year), not 'next week' or 'two days'."""
    return bool(_YEAR.search(span) or _NUMERIC_DATE.search(span) or _MONTH_WORD.search(span))


# Eponymous conditions, procedures and devices that NER mistakes for people.
ALLOW_LIST = frozenset(
    "parkinson alzheimer crohn hodgkin whipple billroth foley heimlich bell raynaud hashimoto graves addison "
    "cushing marfan huntington tourette asperger hirschsprung barrett meniere osgood schatzki hartmann nissen "
    "roux kocher kaposi ewing wilms burkitt paget peyronie dupuytren achilles".split()
)


def _allowed(span: str, allow: frozenset[str]) -> bool:
    return re.sub(r"[’']s$", "", span.strip().lower()) in allow


# ---------------------------------------------------------------- merging and replacement

def merge_entities(entities: list[Entity], text: str = "") -> list[Entity]:
    """Resolve overlaps (higher score, then longer span, wins) and join same-type neighbours."""
    chosen: list[Entity] = []
    for e in sorted(entities, key=lambda e: (-e.score, -(e.end - e.start), e.start)):
        if e.end > e.start and not any(e.start < c.end and c.start < e.end for c in chosen):
            chosen.append(e)
    chosen.sort(key=lambda e: e.start)

    merged: list[Entity] = []
    for e in chosen:
        prev = merged[-1] if merged else None
        if prev and prev.type == e.type and not text[prev.end:e.start].strip(" ,.-"):
            merged[-1] = Entity(prev.start, e.end, e.type, max(prev.score, e.score))
        else:
            merged.append(e)
    return merged


def apply_redactions(text: str, entities: list[Entity]) -> str:
    out = text
    for e in sorted(entities, key=lambda e: e.start, reverse=True):
        out = out[: e.start] + f"[{e.type}]" + out[e.end:]
    return out


class Redactor:
    def __init__(self, analyzers: list[Analyzer], allow_list: frozenset[str] = ALLOW_LIST):
        self.analyzers = analyzers
        self.allow_list = allow_list

    def entities(self, text: str) -> list[Entity]:
        found: list[Entity] = []
        for analyze in self.analyzers:
            for e in analyze(text):
                if e.type in {"PERSON", "LOCATION"} and _allowed(text[e.start:e.end], self.allow_list):
                    continue
                found.append(e)
        return merge_entities(found, text)

    def redact(self, text: str) -> str:
        return apply_redactions(text, self.entities(text))


# ---------------------------------------------------------------- Presidio (lazy)

PRESIDIO_ENTITIES = [
    "PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME", "US_SSN", "US_DRIVER_LICENSE",
    "US_PASSPORT", "CREDIT_CARD", "IP_ADDRESS", "MEDICAL_LICENSE", "US_BANK_NUMBER",
]


class PresidioAnalyzer:
    """Adapts a Presidio AnalyzerEngine to the Analyzer interface."""

    def __init__(self, engine, threshold: float = 0.4):
        self.engine = engine
        self.threshold = threshold

    def __call__(self, text: str) -> list[Entity]:
        results = self.engine.analyze(
            text=text, language="en", entities=PRESIDIO_ENTITIES, score_threshold=self.threshold
        )
        out = []
        for r in results:
            if r.entity_type == "DATE_TIME" and not is_specific_date(text[r.start:r.end]):
                continue  # relative/clinical time ("next week", "two days") is not identifying
            out.append(Entity(r.start, r.end, r.entity_type, r.score))
        return out


def load_presidio_engine(spacy_model: str):
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        raise StageError(f"presidio-analyzer not installed: {exc}") from exc
    try:
        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        ).create_engine()
    except OSError as exc:  # spaCy raises OSError when the model package is missing
        raise StageError(
            f"spaCy model '{spacy_model}' is not installed. Run once: python -m spacy download {spacy_model}"
        ) from exc
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])


def build_redactor(spacy_model: str) -> Redactor:
    """Rules + Presidio. Raises StageError if Presidio or its spaCy model is unavailable."""
    return Redactor([rule_entities, PresidioAnalyzer(load_presidio_engine(spacy_model))])
