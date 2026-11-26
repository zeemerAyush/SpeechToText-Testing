"""Merge word timestamps (Whisper) with speaker turns (pyannote). Pure logic, no ML imports.

Word:  {"word": str, "start": float, "end": float, "prob": float}
Turn:  {"start": float, "end": float, "speaker": str}   (speaker ids are anonymous, e.g. SPEAKER_00)
"""
from __future__ import annotations

UNKNOWN_SPEAKER = "SPEAKER_00"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(words: list[dict], turns: list[dict]) -> list[dict]:
    """Give each word the speaker whose turn overlaps it most.

    Words overlapping no turn (e.g. inside a diarization gap) get the nearest turn's speaker.
    With no turns at all, everything is attributed to a single speaker.
    """
    out = []
    for w in words:
        scores: dict[str, float] = {}
        for t in turns:
            o = _overlap(w["start"], w["end"], t["start"], t["end"])
            if o > 0:
                scores[t["speaker"]] = scores.get(t["speaker"], 0.0) + o
        if scores:
            speaker = max(scores, key=scores.get)
        elif turns:
            mid = (w["start"] + w["end"]) / 2

            def distance(t: dict) -> float:
                if t["start"] <= mid <= t["end"]:
                    return 0.0
                return min(abs(mid - t["start"]), abs(mid - t["end"]))

            speaker = min(turns, key=distance)["speaker"]
        else:
            speaker = UNKNOWN_SPEAKER
        out.append({**w, "speaker": speaker})
    return out


def _make_segment(words: list[dict]) -> dict:
    probs = [w["prob"] for w in words if w.get("prob") is not None]
    return {
        "speaker": words[0]["speaker"],
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"].strip() for w in words if w["word"].strip()),
        "confidence": sum(probs) / len(probs) if probs else None,
        "words": [
            {"word": w["word"].strip(), "start": w["start"], "end": w["end"], "prob": w.get("prob")}
            for w in words
        ],
    }


def build_segments(words_with_speakers: list[dict], max_gap: float = 3.0) -> list[dict]:
    """Group consecutive words by the same speaker; also split on a long silence."""
    segments: list[dict] = []
    current: list[dict] = []
    for w in words_with_speakers:
        if current and (
            w["speaker"] != current[-1]["speaker"] or w["start"] - current[-1]["end"] > max_gap
        ):
            segments.append(_make_segment(current))
            current = []
        current.append(w)
    if current:
        segments.append(_make_segment(current))
    return segments


def label_speakers(segments: list[dict]) -> list[dict]:
    """Rename anonymous ids to 'Speaker 1', 'Speaker 2', ... in order of first appearance."""
    names: dict[str, str] = {}
    out = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        if seg["speaker"] not in names:
            names[seg["speaker"]] = f"Speaker {len(names) + 1}"
        out.append({**seg, "speaker": names[seg["speaker"]]})
    return out
