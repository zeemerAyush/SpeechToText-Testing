"""Decrypt a recording into a short-lived temp dir that is ALWAYS wiped afterwards."""
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


@contextmanager
def decrypted_workspace(settings: Settings, recording: dict) -> Iterator[Path]:
    """Yield the path of a decrypted working copy; wipe it on exit, even on failure."""
    job_dir = settings.tmp_dir / f"job_{uuid.uuid4().hex[:12]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        data_key = decode_key(recording["data_key"])
        work = job_dir / "audio.bin"
        decrypt_file(settings.storage_dir / recording["audio_path"], work, data_key)
        if sha256_file(work) != recording["sha256"]:
            raise IntegrityError("Integrity check failed: audio hash mismatch.")
        yield work
    finally:
        wipe_dir(job_dir)
