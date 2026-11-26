"""(Re)run PHI redaction on stored transcripts whose redaction is missing or failed.

    python -m worker.redact_backfill

Needs only the stored transcript text (no audio, no GPU). Use it after installing Presidio/spaCy, after a
redaction failure, or for transcripts created before redaction existed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from worker import config  # noqa: E402,F401  (offline flags first)

from pymongo.database import Database  # noqa: E402

from app.gate import audit  # noqa: E402


def backfill(db: Database, redact: Callable[[str], str], engine: str) -> dict[str, int]:
    done = failed = 0
    for t in list(db.transcripts.find({"redaction.status": {"$ne": "ok"}})):
        try:
            segments = [{**s, "text_redacted": redact(s["text"])} for s in t["segments"]]
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"[:200]
            db.transcripts.update_one(
                {"_id": t["_id"]}, {"$set": {"redaction": {"status": "failed", "engine": engine, "error": msg}}}
            )
            audit.record(db, "redact", recording_id=t["recording_id"], detail=f"backfill failed: {msg}")
            failed += 1
            continue
        full = "\n".join(f'{s["speaker"]}: {s["text_redacted"]}' for s in segments)
        db.transcripts.update_one(
            {"_id": t["_id"]},
            {"$set": {"segments": segments, "full_text_redacted": full,
                      "redaction": {"status": "ok", "engine": engine}}},
        )
        audit.record(db, "redact", recording_id=t["recording_id"], detail="backfill ok")
        done += 1
    return {"redacted": done, "failed": failed}


def main() -> int:
    from app.config import get_settings  # noqa: F401
    from app.db.mongo import get_db
    from worker.pipeline.audio_clean import StageError
    from worker.pipeline.redact import build_redactor

    try:
        redactor = build_redactor(config.SPACY_MODEL)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result = backfill(get_db(), redactor.redact, "presidio+rules")
    print(f"Redacted {result['redacted']} transcript(s); {result['failed']} failed.")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
