"""Polls MongoDB for queued recordings, runs the pipeline, and stores the transcript.

Run from the project root:  python -m worker.run_worker [--once]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from worker import config  # noqa: E402  (sets offline flags before anything else)

from pymongo.database import Database  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.db.mongo import get_db  # noqa: E402
from app.gate import audit  # noqa: E402
from app.gate.storage import wipe_dir  # noqa: E402
from app.gate.workspace import decrypted_workspace  # noqa: E402

# runner(decrypted_audio_path, recording_options, on_stage) -> transcript dict
Runner = Callable[[Path, dict, Callable[[str], None]], dict]


def make_default_runner(settings: Settings) -> Runner:
    """The real pipeline. Models are loaded once and reused for every recording."""
    from worker.pipeline.run import Options, PipelineConfig, default_stages, process_audio

    cfg = PipelineConfig()
    stages = default_stages(cfg)

    def runner(audio: Path, opts: dict, on_stage: Callable[[str], None]) -> dict:
        options = Options(
            denoise=opts.get("denoise", config.DENOISE),
            filler_removal=opts.get("filler_removal", False),
            min_speakers=opts.get("min_speakers"),
            max_speakers=opts.get("max_speakers"),
        )
        return process_audio(audio, options, stages, cfg, settings.tmp_dir, on_stage=on_stage)

    return runner


def claim_next(db: Database) -> dict | None:
    """Atomically take the oldest queued recording."""
    return db.recordings.find_one_and_update(
        {"status": "queued"},
        {"$set": {"status": "processing", "started_at": datetime.now(timezone.utc), "stage": "starting"}},
        sort=[("created_at", 1)],
        return_document=True,
    )


def store_transcript(db: Database, rec_id: str, result: dict) -> None:
    """Save the transcript. Redacted fields stay null until PHI redaction (Phase 5) exists."""
    doc = {
        "recording_id": rec_id,
        "language": result["language"],
        "speaker_count": result["speaker_count"],
        "segments": [{**seg, "text_redacted": None} for seg in result["segments"]],
        "full_text": result["full_text"],
        "full_text_redacted": None,
        "pipeline": result["pipeline"],
        "created_at": datetime.now(timezone.utc),
    }
    db.transcripts.replace_one({"recording_id": rec_id}, doc, upsert=True)


def process(db: Database, settings: Settings, rec: dict, runner: Runner) -> None:
    rec_id = rec["_id"]
    audit.record(db, "process_start", recording_id=rec_id, detail="worker")

    def on_stage(name: str) -> None:
        db.recordings.update_one({"_id": rec_id}, {"$set": {"stage": name}})

    try:
        with decrypted_workspace(settings, rec) as audio:
            result = runner(audio, rec.get("options") or {}, on_stage)
        store_transcript(db, rec_id, result)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"[:300]
        db.recordings.update_one(
            {"_id": rec_id}, {"$set": {"status": "failed", "error": msg, "stage": None}}
        )
        audit.record(db, "process_fail", recording_id=rec_id, detail=msg)
        return
    update = {"status": "completed", "completed_at": datetime.now(timezone.utc), "error": None, "stage": None}
    if rec.get("duration_sec") is None:  # e.g. WebM when ffprobe was unavailable at the gate
        update["duration_sec"] = result.get("duration_sec")
    db.recordings.update_one({"_id": rec_id}, {"$set": update})
    audit.record(db, "process_done", recording_id=rec_id)


def run_once(db: Database, settings: Settings, runner: Runner) -> bool:
    rec = claim_next(db)
    if rec is None:
        return False
    process(db, settings, rec, runner)
    return True


def recover_after_crash(db: Database, settings: Settings) -> int:
    """Single-worker startup recovery: requeue jobs stuck in 'processing' and wipe stale temp files."""
    stuck = list(db.recordings.find({"status": "processing"}))
    for rec in stuck:
        db.recordings.update_one({"_id": rec["_id"]}, {"$set": {"status": "queued", "stage": None}})
        audit.record(db, "queued", recording_id=rec["_id"], detail="requeued after worker restart")
    if settings.tmp_dir.exists():
        for child in settings.tmp_dir.iterdir():
            if child.name.startswith(("job_", "cli_")):
                wipe_dir(child)
    return len(stuck)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="drain the queue then exit")
    args = parser.parse_args()
    settings, db = get_settings(), get_db()
    requeued = recover_after_crash(db, settings)
    if requeued:
        print(f"Requeued {requeued} interrupted recording(s).", file=sys.stderr)
    runner = make_default_runner(settings)
    while True:
        if run_once(db, settings, runner):
            continue
        if args.once:
            return
        time.sleep(config.POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
