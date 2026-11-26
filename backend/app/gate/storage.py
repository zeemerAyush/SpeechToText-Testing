"""Temporary-file handling for plaintext audio: staging, decrypting, and wiping."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def wipe_file(path: Path) -> None:
    """Overwrite a file with zeros, then delete it. Best effort (SSD wear-levelling and
    journaling filesystems can keep old blocks, so disk-level encryption is still advised)."""
    try:
        size = path.stat().st_size
        with open(path, "r+b") as fh:
            remaining = size
            block = b"\x00" * min(1024 * 1024, max(size, 1))
            while remaining > 0:
                n = min(len(block), remaining)
                fh.write(block[:n])
                remaining -= n
            fh.flush()
            os.fsync(fh.fileno())
    except FileNotFoundError:
        return
    except OSError:
        pass  # still try to delete below
    path.unlink(missing_ok=True)


def wipe_dir(path: Path) -> None:
    """Wipe every file under ``path`` and remove the directory."""
    if not path.exists():
        return
    for p in path.rglob("*"):
        if p.is_file():
            wipe_file(p)
    shutil.rmtree(path, ignore_errors=True)
