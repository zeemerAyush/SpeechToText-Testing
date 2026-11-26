"""Worker configuration. Models must run offline: set the flags BEFORE any ML import."""
from __future__ import annotations

import os

# Offline flags: model libraries cannot download or call out while handling audio.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = os.environ.get("DEVICE", "cpu")  # cuda | cpu
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
POLL_INTERVAL_SEC = float(os.environ.get("WORKER_POLL_SEC", "3"))
