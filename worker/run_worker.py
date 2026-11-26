"""Polls MongoDB for queued recordings and processes them one at a time.

Run from the project root:  python -m worker.run_worker [--once]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from worker import config  # noqa: E402,F401  (sets offline flags before anything else)

from pymongo.database import Database  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.db.mongo import get_db  # noqa: E402
from app.gate import audit  # noqa: E402
from worker.pipeline import noop  # noqa: E402
from worker.secure_workspace import decrypted_workspace  # noqa: E402


def claim_next(db: Database) -> dict | None:
    """Atomically take the oldest queued recording."""
    return db.recordings.find_one_and_update(
        {"status": "queued"},
        {"$set": {"status": "processing", "started_at": datetime.now(timezone.utc)}},
        sort=[("created_at", 1)],
        return_document=True,
    )


def process(db: Database, settings: Settings, rec: dict) -> None:
    rec_id = rec["_id"]
    audit.record(db, "process_start", recording_id=rec_id, detail="worker")
    try:
        with decrypted_workspace(settings, rec) as audio:
            noop.run(audio)  # Phase 2 replaces this with the real pipeline
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"[:300]
        db.recordings.update_one({"_id": rec_id}, {"$set": {"status": "failed", "error": msg}})
        audit.record(db, "process_fail", recording_id=rec_id, detail=msg)
        return
    db.recordings.update_one(
        {"_id": rec_id},
        {"$set": {"status": "completed", "completed_at": datetime.now(timezone.utc), "error": None}},
    )
    audit.record(db, "process_done", recording_id=rec_id)


def run_once(db: Database, settings: Settings) -> bool:
    rec = claim_next(db)
    if rec is None:
        return False
    process(db, settings, rec)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="drain the queue then exit")
    args = parser.parse_args()
    settings, db = get_settings(), get_db()
    while True:
        if run_once(db, settings):
            continue
        if args.once:
            return
        time.sleep(config.POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
