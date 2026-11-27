"""Check that every model loads and runs OFFLINE, using the same code paths as the worker.

    python scripts/check_models.py

Reads DEVICE, WHISPER_MODEL, WHISPER_MODEL_DIR, PYANNOTE_CONFIG and SPACY_MODEL from .env (or the shell). The model
libraries are put in offline mode first, so a missing download shows up here as a failure instead of at the first
real recording. Each check is independent and reports PASS or FAIL with a hint.
"""
from __future__ import annotations

import math
import shutil
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from worker import config  # noqa: E402  (loads .env and sets the offline flags before any model import)

results: list[tuple[str, bool]] = []


def check(name: str, hint: str):
    """Decorator: run the function, print PASS/FAIL with timing, remember the result."""
    def wrap(fn):
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            results.append((name, True))
            print(f"[PASS] {name} ({time.perf_counter() - t0:.1f}s) {detail}")
        except Exception as exc:
            results.append((name, False))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}\n       hint: {hint}")
        return fn
    return wrap


def make_test_wav(path: Path, seconds: float = 3.0, rate: int = 16000) -> None:
    """A plain tone, not speech: enough to prove a model loads and runs, without any patient audio."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / rate)))
                               for i in range(int(seconds * rate))))


def main() -> int:
    print(f"device={config.DEVICE}  whisper={config.WHISPER_MODEL}  whisper_dir={config.WHISPER_MODEL_DIR}")
    print(f"pyannote_config={config.PYANNOTE_CONFIG}  spacy={config.SPACY_MODEL}  denoise={config.DENOISE}")
    import os
    print(f"offline: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}  "
          f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}\n")

    tmp = Path(tempfile.mkdtemp(prefix="check_models_"))
    wav = tmp / "tone.wav"
    make_test_wav(wav)
    try:
        @check("ffmpeg + ffprobe on PATH", "install FFmpeg (apt install ffmpeg)")
        def _ffmpeg():
            missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
            if missing:
                raise RuntimeError(f"not found: {', '.join(missing)}")

        @check("GPU available to PyTorch", "install the CUDA build of torch, or set DEVICE=cpu")
        def _gpu():
            import torch

            if config.DEVICE == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("DEVICE=cuda but torch.cuda.is_available() is False")
            return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"

        @check("PHI redaction (Presidio + spaCy)", "python scripts/download_models.py (spacy step)")
        def _redact():
            from worker.pipeline.redact import build_redactor

            r = build_redactor(config.SPACY_MODEL)
            a = r.redact("My name is John Smith and my phone number is 555-014-2237.")
            b = r.redact("Dr. Smith will see you next week.")
            if "[PERSON]" not in a or "[PHONE_NUMBER]" not in a:
                raise RuntimeError(f"identifiers were not redacted: {a!r}")
            if "next week" not in b:
                raise RuntimeError(f"relative time was wrongly redacted: {b!r}")
            return f"-> {a}"

        @check("Silero voice-activity detection", "pip install silero-vad")
        def _vad():
            from worker.pipeline import audio_clean

            audio_clean.speech_regions(wav)

        @check("DeepFilterNet denoising", "optional: set DENOISE=0, or re-run download_models.py (deepfilternet step)")
        def _denoise():
            from worker.pipeline import audio_clean

            out = tmp / "denoised.wav"
            audio_clean.denoise(wav, out, config.DEVICE)
            if not out.is_file():
                raise RuntimeError("no output file produced")

        @check("Whisper transcription", "re-run download_models.py (whisper step) and set WHISPER_MODEL_DIR in .env")
        def _whisper():
            from worker.pipeline import transcribe

            model = transcribe.load_model(config.WHISPER_MODEL, config.DEVICE, config.COMPUTE_TYPE,
                                          config.WHISPER_MODEL_DIR)
            transcribe.transcribe(model, wav)  # a tone has no words; this proves load + decode + inference

        @check("pyannote speaker diarization", "re-run download_models.py (pyannote step) and set PYANNOTE_CONFIG in .env")
        def _diarize():
            from worker.pipeline import diarize

            pipeline = diarize.load_pipeline(config.PYANNOTE_CONFIG, config.DEVICE)
            diarize.diarize(pipeline, wav)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("Failed: " + ", ".join(failed))
        return 1
    print("All models load offline. Next: python -m worker.cli <a short speech recording>.wav -o out.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
