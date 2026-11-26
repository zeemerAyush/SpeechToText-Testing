"""The guardrail gate: stage -> validate -> hash -> encrypt -> queue.

Nothing downstream (the worker, and later any model) sees audio unless it passes through here.
Auth/RBAC is enforced by the API layer before this runs; every outcome is audited.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from pymongo.database import Database

from app.config import Settings
from app.gate import audit
from app.gate.crypto import encode_key, encrypt_file, generate_data_key
from app.gate.storage import wipe_dir
from app.gate.validation import ValidationError, validate_audio


def new_recording_id() -> str:
    return "rec_" + uuid.uuid4().hex[:16]


def safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    return re.sub(r"[^\w.\- ]", "_", base)[:120] or "unnamed"


def _copy_limited(src: BinaryIO, dst: Path, max_bytes: int) -> None:
    written = 0
    with open(dst, "wb") as out:
        while chunk := src.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise ValidationError(f"File exceeds the maximum size of {max_bytes // (1024 * 1024)} MB.", 413)
            out.write(chunk)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _set_status(db: Database, rec_id: str, status: str, **extra) -> None:
    db.recordings.update_one({"_id": rec_id}, {"$set": {"status": status, **extra}})


def run_gate(
    db: Database,
    settings: Settings,
    *,
    stream: BinaryIO,
    filename: str,
    user_id: str,
    source: str,
    reference: str | None,
    ip: str | None,
    options: dict | None = None,
) -> dict:
    """Run the gate on an uploaded stream. Returns the recording document on success.

    Raises ValidationError (after auditing and cleanup) if any check fails.
    """
    rec_id = new_recording_id()
    clean_name = safe_filename(filename)
    now = datetime.now(timezone.utc)
    db.recordings.insert_one(
        {
            "_id": rec_id,
            "uploaded_by": user_id,
            "source": source,
            "original_filename": clean_name,
            "reference": reference,
            "options": options or {},
            "stage": None,
            "attempts": 0,
            "audio_path": None,
            "data_key": None,
            "sha256": None,
            "duration_sec": None,
            "status": "uploaded",
            "error": None,
            "created_at": now,
            "completed_at": None,
        }
    )
    audit.record(db, "upload", user_id=user_id, recording_id=rec_id, detail=clean_name, ip=ip)

    stage = settings.tmp_dir / f"upload_{rec_id}"
    stage.mkdir(parents=True, exist_ok=True)
    staged = stage / "upload.bin"
    try:
        _copy_limited(stream, staged, settings.max_upload_bytes)
        result = validate_audio(
            staged,
            clean_name,
            max_bytes=settings.max_upload_bytes,
            max_duration_sec=settings.max_duration_sec,
            require_ffprobe=settings.require_ffprobe,
        )
        digest = sha256_file(staged)

        data_key = generate_data_key()
        enc_path = settings.audio_dir / f"{rec_id}.enc"
        encrypt_file(staged, enc_path, data_key)
    except ValidationError as exc:
        _set_status(db, rec_id, "failed", error=str(exc))
        audit.record(db, "gate_fail", user_id=user_id, recording_id=rec_id, detail=str(exc), ip=ip)
        raise
    except Exception as exc:  # unexpected failure: fail closed, keep nothing readable
        _set_status(db, rec_id, "failed", error="Internal error while securing the recording.")
        audit.record(db, "gate_fail", user_id=user_id, recording_id=rec_id,
                     detail=f"internal error: {type(exc).__name__}", ip=ip)
        (settings.audio_dir / f"{rec_id}.enc").unlink(missing_ok=True)
        raise
    finally:
        wipe_dir(stage)  # plaintext upload never survives the gate

    _set_status(
        db, rec_id, "gate_passed",
        audio_path=str(enc_path.relative_to(settings.storage_dir)),
        data_key=encode_key(data_key),
        sha256=digest,
        duration_sec=result.duration_sec,
    )
    audit.record(
        db, "gate_pass", user_id=user_id, recording_id=rec_id,
        detail=f"{result.container}, checked with {result.checked_with}", ip=ip,
    )
    _set_status(db, rec_id, "queued")
    audit.record(db, "queued", user_id=user_id, recording_id=rec_id, ip=ip)
    return db.recordings.find_one({"_id": rec_id})
