"""Speech-to-text with faster-whisper (English, word timestamps)."""
from __future__ import annotations

from pathlib import Path

from worker.pipeline.audio_clean import StageError


def load_model(model_name: str, device: str, compute_type: str, model_dir: str | None = None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise StageError(f"faster-whisper not installed: {exc}") from exc
    return WhisperModel(model_dir or model_name, device=device, compute_type=compute_type)


def transcribe(model, wav: Path, initial_prompt: str | None = None) -> list[dict]:
    """Return a flat list of words: {"word", "start", "end", "prob"}."""
    segments, _info = model.transcribe(
        str(wav),
        language="en",
        word_timestamps=True,
        vad_filter=True,                    # avoid inventing text over silence
        condition_on_previous_text=False,   # prevents repeated-text loops
        initial_prompt=initial_prompt,
        beam_size=5,
    )
    words: list[dict] = []
    for seg in segments:  # generator: decoding happens here
        for w in seg.words or []:
            words.append(
                {"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2),
                 "prob": round(w.probability, 3)}
            )
    return words
