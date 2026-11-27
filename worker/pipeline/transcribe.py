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


def read_wav_16k_mono(wav: Path):
    """Decode the pipeline's own 16 kHz mono 16-bit WAV into float32 samples, or return None for any other
    format. Handing Whisper the samples means faster-whisper never has to open the file through PyAV, whose
    ``av.open`` signature changes between releases (faster-whisper 1.2 breaks with PyAV 19)."""
    import wave

    import numpy as np

    try:
        with wave.open(str(wav), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                return None
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return None
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def transcribe(model, wav: Path, initial_prompt: str | None = None) -> list[dict]:
    """Return a flat list of words: {"word", "start", "end", "prob"}."""
    samples = read_wav_16k_mono(wav)
    segments, _info = model.transcribe(
        samples if samples is not None else str(wav),
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
