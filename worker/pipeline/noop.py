"""Phase 1 placeholder 'model': proves the gate -> decrypt -> process -> cleanup flow.

It reads the decrypted file but runs no AI. Replaced by the real pipeline in Phase 2.
"""
from __future__ import annotations

from pathlib import Path


def run(audio_path: Path) -> dict:
    return {"bytes_read": audio_path.stat().st_size}
