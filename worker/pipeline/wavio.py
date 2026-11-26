"""Tiny stdlib WAV helpers (16-bit PCM mono), so ML stages need no torchaudio/soundfile."""
from __future__ import annotations

import wave
from array import array
from pathlib import Path


def read_pcm16(path: Path) -> tuple[array, int]:
    """Return (samples, sample_rate) for a mono 16-bit WAV."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit PCM WAV (run the convert stage first).")
        rate = w.getframerate()
        samples = array("h")
        samples.frombytes(w.readframes(w.getnframes()))
    return samples, rate


def write_pcm16(path: Path, samples: array, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())
