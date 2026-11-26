"""Decrypt a recording into a short-lived temp dir. Callers MUST wipe it afterwards.

Used by the worker (context manager, always wiped) and by the admin audio endpoint
(wiped by a background task once the response has been sent).
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import Settings
from app.gate.crypto import decode_key, decrypt_file
from app.gate.service import sha256_file
from app.gate.storage import wipe_dir


class IntegrityError(Exception):
    """Decrypted audio does not match the SHA-256 recorded at upload time."""


def decrypt_to_temp(settings: Settings, recording: dict) -> tuple[Path, Path]:
    """Return (decrypted_file, job_dir). On any failure the job_dir is wiped before raising."""
    job_dir = settings.tmp_dir / f"job_{uuid.uuid4().hex[:12]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        work = job_dir / "audio.bin"
        decrypt_file(settings.storage_dir / recording["audio_path"], work, decode_key(recording["data_key"]))
        if sha256_file(work) != recording["sha256"]:
            raise IntegrityError("Integrity check failed: audio hash mismatch.")
        return work, job_dir
    except BaseException:
        wipe_dir(job_dir)
        raise


@contextmanager
def decrypted_workspace(settings: Settings, recording: dict) -> Iterator[Path]:
    """Yield the path of a decrypted working copy; wipe it on exit, even on failure."""
    work, job_dir = decrypt_to_temp(settings, recording)
    try:
        yield work
    finally:
        wipe_dir(job_dir)
