"""What each role may see of a transcript. Pure functions, no I/O.

PHI redaction arrives in Phase 5. Until then ``text_redacted`` is null, so the redacted view
carries no text at all: the original text is never shown to roles that may not see it (fail closed).
"""
from __future__ import annotations


def redacted_view(t: dict) -> dict:
    return {
        "recording_id": t["recording_id"],
        "language": t["language"],
        "speaker_count": t["speaker_count"],
        "redaction": "available" if t.get("full_text_redacted") is not None else "pending",
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
