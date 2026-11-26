"""Audio cleaning stages: convert/normalize (FFmpeg), denoise (DeepFilterNet), VAD (Silero).

ML libraries are imported lazily so the rest of the pipeline works without them installed.
"""
from __future__ import annotations

import shutil
import subprocess
from array import array
from pathlib import Path

from worker.pipeline import wavio


class StageError(RuntimeError):
    """A pipeline stage failed or its dependency is missing."""


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise StageError("FFmpeg is not installed or not on PATH.")
    return exe


def convert_to_wav16k(src: Path, dst: Path, normalize: bool = True) -> None:
    """Convert any supported audio to 16 kHz mono 16-bit WAV, optionally loudness-normalised."""
    cmd = [_ffmpeg(), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000"]
    if normalize:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise StageError(f"FFmpeg conversion failed: {proc.stderr.strip()[:300]}")


def denoise(src: Path, dst: Path, device: str = "cpu") -> None:
    """Remove steady background noise with DeepFilterNet (works at 48 kHz internally)."""
    try:
        import numpy as np
        import torch
        from df.enhance import enhance, init_df
    except ImportError as exc:
        raise StageError(f"DeepFilterNet/torch not installed: {exc}") from exc

    ffmpeg = _ffmpeg()
    up, out48 = src.with_name(src.stem + "_48k.wav"), src.with_name(src.stem + "_48k_dn.wav")
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(src), "-ar", "48000", "-ac", "1",
                    "-c:a", "pcm_s16le", str(up)], check=True)
    samples, rate = wavio.read_pcm16(up)
    audio = torch.from_numpy(np.asarray(samples, dtype=np.float32) / 32768.0).unsqueeze(0)

    model, df_state, _ = init_df()
    enhanced = enhance(model, df_state, audio).squeeze(0).cpu().numpy()
    pcm = array("h", (np.clip(enhanced, -1.0, 1.0) * 32767).astype(np.int16).tolist())
    wavio.write_pcm16(out48, pcm, rate)
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(out48), "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le", str(dst)], check=True)


def speech_regions(wav: Path) -> list[tuple[float, float]]:
    """Silero VAD: return (start, end) seconds of detected speech."""
    try:
        import numpy as np
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise StageError(f"silero-vad/torch not installed: {exc}") from exc

    samples, rate = wavio.read_pcm16(wav)
    audio = torch.from_numpy(np.asarray(samples, dtype=np.float32) / 32768.0)
    stamps = get_speech_timestamps(audio, load_silero_vad(), sampling_rate=rate, return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in stamps]
