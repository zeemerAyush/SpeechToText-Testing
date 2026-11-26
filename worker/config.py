"""Worker configuration. Models must run offline: set the flags BEFORE any ML import."""
from __future__ import annotations

import os

# Offline flags: model libraries cannot download or call out while handling audio.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = os.environ.get("DEVICE", "cpu")  # cuda | cpu
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # e.g. large-v3 on a GPU
# float16 on GPU, int8 on CPU unless overridden.
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
# Optional local folder with a pre-downloaded faster-whisper model (otherwise the HF cache is used).
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR") or None
# Path to a local pyannote pipeline config.yaml (downloaded once, then used offline).
PYANNOTE_CONFIG = os.environ.get("PYANNOTE_CONFIG", "models/pyannote/config.yaml")
# spaCy model used by Presidio for names/places (install once: python -m spacy download en_core_web_lg).
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_lg")
DENOISE = os.environ.get("DENOISE", "1") not in {"0", "false", "False"}
POLL_INTERVAL_SEC = float(os.environ.get("WORKER_POLL_SEC", "3"))
