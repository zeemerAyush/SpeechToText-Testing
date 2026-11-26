"""What each role may see of a transcript. Pure functions, no I/O.

The redacted view only ever contains ``text_redacted``. If redaction did not succeed it is null, so the
view carries no text at all: original text is never shown to roles that may not see it (fail closed).
"""
from __future__ import annotations


def _redaction_state(t: dict) -> str:
    status = (t.get("redaction") or {}).get("status")
    if status == "ok" and t.get("full_text_redacted") is not None:
        return "available"
    return "failed" if status == "failed" else "pending"


def redacted_view(t: dict) -> dict:
    return {
        "recording_id": t["recording_id"],
        "language": t["language"],
        "speaker_count": t["speaker_count"],
        "redaction": _redaction_state(t),
        "segments": [
            {
                "speaker": s["speaker"],
                "start": s["start"],
                "end": s["end"],
                "text_redacted": s.get("text_redacted"),
                "confidence": s.get("confidence"),
            }
            for s in t["segments"]
        ],
        "full_text_redacted": t.get("full_text_redacted"),
        "pipeline": t["pipeline"],
    }


def original_view(t: dict) -> dict:
    out = {k: v for k, v in t.items() if k != "_id"}
    out["created_at"] = t["created_at"].isoformat()
    return out
