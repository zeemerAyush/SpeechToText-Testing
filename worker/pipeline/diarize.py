"""Speaker diarization with pyannote.audio 3.1 (runs from a local, pre-downloaded config)."""
from __future__ import annotations

from pathlib import Path

from worker.pipeline import wavio
from worker.pipeline.audio_clean import StageError


def load_pipeline(config_path: str, device: str):
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise StageError(f"pyannote.audio/torch not installed: {exc}") from exc
    if not Path(config_path).exists():
        raise StageError(
            f"pyannote config not found at '{config_path}'. Download the pipeline once "
            "(needs a Hugging Face token) and set PYANNOTE_CONFIG."
        )
    pipeline = Pipeline.from_pretrained(config_path)
    pipeline.to(torch.device(device))
    return pipeline


def diarize(
    pipeline, wav: Path, min_speakers: int | None = None, max_speakers: int | None = None
) -> list[dict]:
    """Return speaker turns: {"start", "end", "speaker"} with anonymous ids."""
    import numpy as np
    import torch

    samples, rate = wavio.read_pcm16(wav)  # in-memory, so pyannote needs no audio backend
    waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32) / 32768.0).unsqueeze(0)
    result = pipeline(
        {"waveform": waveform, "sample_rate": rate},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    annotation = getattr(result, "speaker_diarization", result)  # newer versions wrap the result
    return [
        {"start": round(t.start, 2), "end": round(t.end, 2), "speaker": spk}
        for t, _, spk in annotation.itertracks(yield_label=True)
    ]
