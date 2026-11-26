from array import array

import pytest

from worker.pipeline import merge, postprocess, wavio


def w(word, start, end, prob=0.95):
    return {"word": word, "start": start, "end": end, "prob": prob}


def seg(speaker, start, end, text, conf=0.95):
    return {"speaker": speaker, "start": start, "end": end, "text": text, "confidence": conf,
            "words": [w(t, start, end, conf) for t in text.split()]}


# ---------- merge ----------

def test_word_goes_to_speaker_with_most_overlap():
    turns = [{"start": 0, "end": 2.0, "speaker": "A"}, {"start": 2.0, "end": 5, "speaker": "B"}]
    out = merge.assign_speakers([w("hi", 0.5, 1.0), w("there", 1.8, 2.6), w("yes", 3, 3.5)], turns)
    assert [x["speaker"] for x in out] == ["A", "B", "B"]  # "there": 0.2 in A, 0.6 in B


def test_word_in_diarization_gap_gets_nearest_speaker():
    turns = [{"start": 0, "end": 1, "speaker": "A"}, {"start": 5, "end": 6, "speaker": "B"}]
    out = merge.assign_speakers([w("x", 1.2, 1.4), w("y", 4.5, 4.7)], turns)
    assert [x["speaker"] for x in out] == ["A", "B"]


def test_no_turns_means_single_speaker():
    out = merge.assign_speakers([w("a", 0, 1)], [])
    assert out[0]["speaker"] == merge.UNKNOWN_SPEAKER


def test_segments_split_on_speaker_change_and_long_gap():
    words = merge.assign_speakers(
        [w("Hello", 0, 0.5), w("doctor", 0.6, 1.0), w("Hi", 1.2, 1.5), w("again", 9.0, 9.5)],
        [{"start": 0, "end": 1.1, "speaker": "A"}, {"start": 1.1, "end": 10, "speaker": "B"}],
    )
    segs = merge.build_segments(words, max_gap=3.0)
    assert [(s["speaker"], s["text"]) for s in segs] == [("A", "Hello doctor"), ("B", "Hi"), ("B", "again")]
    assert segs[0]["start"] == 0 and segs[0]["end"] == 1.0
    assert segs[0]["confidence"] == pytest.approx(0.95)


def test_speakers_labelled_in_order_of_first_appearance():
    segs = [seg("SPEAKER_07", 0, 1, "a"), seg("SPEAKER_02", 1, 2, "b"), seg("SPEAKER_07", 2, 3, "c")]
    assert [s["speaker"] for s in merge.label_speakers(segs)] == ["Speaker 1", "Speaker 2", "Speaker 1"]


# ---------- postprocess ----------

def test_normalize_text():
    assert postprocess.normalize_text("  hello   there , doctor .") == "Hello there, doctor."


@pytest.mark.parametrize("text", ["", "...", "[MUSIC]", "(laughs)", "♪ la la ♪", "*cough*"])
def test_non_speech_dropped(text):
    assert postprocess.postprocess([seg("A", 0, 1, text or " ")]) == []


def test_repetition_loop_dropped_but_normal_repeat_kept():
    loop = "thank you " * 6
    assert postprocess.postprocess([seg("A", 0, 5, loop)]) == []
    assert len(postprocess.postprocess([seg("A", 0, 5, "no no no it is fine")])) == 1


def test_known_hallucination_dropped_only_when_suspicious():
    in_speech = [(0.0, 10.0)]
    # genuine, confident "Thank you." inside speech is kept
    assert len(postprocess.postprocess([seg("A", 1, 2, "Thank you.", 0.95)], speech_regions=in_speech)) == 1
    # low confidence -> dropped
    assert postprocess.postprocess([seg("A", 1, 2, "Thank you.", 0.3)], speech_regions=in_speech) == []
    # confident but over silence (outside speech regions) -> dropped
    assert postprocess.postprocess([seg("A", 20, 21, "Thank you.", 0.95)], speech_regions=in_speech) == []


def test_adjacent_same_speaker_segments_merge_after_drops():
    segs = [seg("A", 0, 1, "I think"), seg("A", 1.2, 1.5, "[noise]"), seg("A", 1.6, 2.5, "it hurts"),
            seg("B", 3, 4, "Where?")]
    out = postprocess.postprocess(segs)
    assert [(s["speaker"], s["text"]) for s in out] == [("A", "I think it hurts"), ("B", "Where?")]
    assert out[0]["start"] == 0 and out[0]["end"] == 2.5 and len(out[0]["words"]) == 4


def test_different_speakers_not_merged_even_if_close():
    out = postprocess.postprocess([seg("A", 0, 1, "Yes."), seg("B", 1.1, 2, "Okay.")])
    assert len(out) == 2


def test_filler_removal_keeps_raw_text():
    out = postprocess.postprocess([seg("A", 0, 3, "So um I I I want to uh go home.")], filler_removal=True)
    assert out[0]["text"] == "So I want to go home."
    assert out[0]["text_raw"] == "So um I I I want to uh go home."


def test_filler_removal_off_leaves_text_alone():
    out = postprocess.postprocess([seg("A", 0, 3, "um I want to go")])
    assert out[0]["text"] == "Um I want to go" and "text_raw" not in out[0]


def test_filler_removal_keeps_legit_double_words_and_drops_empty_segment():
    out = postprocess.postprocess([seg("A", 0, 1, "Um."), seg("B", 2, 3, "He said that that works.")],
                                  filler_removal=True)
    assert [s["text"] for s in out] == ["He said that that works."]


# ---------- wavio ----------

def test_wavio_roundtrip(tmp_path):
    p = tmp_path / "a.wav"
    wavio.write_pcm16(p, array("h", [0, 100, -100, 32767]), 16000)
    samples, rate = wavio.read_pcm16(p)
    assert list(samples) == [0, 100, -100, 32767] and rate == 16000
    assert wavio.duration_sec(p) == pytest.approx(4 / 16000)
