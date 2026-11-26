"""Pipeline orchestrator: audio file in -> cleaned, speaker-separated transcript dict out.

Stages are injectable (``Stages``) so the flow can be tested without any ML models.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from worker import config  # noqa: F401  (must run first: sets the offline flags)
from worker.pipeline import merge, postprocess, wavio
from app.gate.storage import wipe_dir


@dataclass
class Options:
    denoise: bool = True
    normalize: bool = True
    filler_removal: bool = False
    min_speakers: int | None = None
    max_speakers: int | None = None
    initial_prompt: str | None = None


@dataclass
class PipelineConfig:
    device: str = config.DEVICE
    whisper_model: str = config.WHISPER_MODEL
    compute_type: str = config.COMPUTE_TYPE
    whisper_model_dir: str | None = config.WHISPER_MODEL_DIR
    pyannote_config: str = config.PYANNOTE_CONFIG


@dataclass
class Stages:
    convert: Callable[[Path, Path, bool], None]            # (src, dst, normalize)
    denoise: Callable[[Path, Path], None]                  # (src, dst)
    vad: Callable[[Path], list[tuple[float, float]]]       # wav -> speech regions
    transcribe: Callable[[Path, str | None], list[dict]]   # (wav, prompt) -> words
    diarize: Callable[[Path, int | None, int | None], list[dict]]  # (wav, min, max) -> turns
    timings: dict[str, float] = field(default_factory=dict)


def default_stages(cfg: PipelineConfig) -> Stages:
    """Real stages. Models load on first use and are reused across recordings."""
    from worker.pipeline import audio_clean, diarize, transcribe

    cache: dict = {}

    def transcribe_stage(wav: Path, prompt: str | None) -> list[dict]:
        if "whisper" not in cache:
            cache["whisper"] = transcribe.load_model(
                cfg.whisper_model, cfg.device, cfg.compute_type, cfg.whisper_model_dir
            )
        return transcribe.transcribe(cache["whisper"], wav, prompt)

    def diarize_stage(wav: Path, lo: int | None, hi: int | None) -> list[dict]:
        if "pyannote" not in cache:
            cache["pyannote"] = diarize.load_pipeline(cfg.pyannote_config, cfg.device)
        return diarize.diarize(cache["pyannote"], wav, lo, hi)

    return Stages(
        convert=audio_clean.convert_to_wav16k,
        denoise=lambda src, dst: audio_clean.denoise(src, dst, cfg.device),
        vad=audio_clean.speech_regions,
        transcribe=transcribe_stage,
        diarize=diarize_stage,
    )


def _timed(stages: Stages, name: str, fn, *args):
    t0 = time.perf_counter()
    try:
        return fn(*args)
    finally:
        stages.timings[name] = round(time.perf_counter() - t0, 2)


def process_audio(
    src: Path, options: Options, stages: Stages, cfg: PipelineConfig, work_root: Path
) -> dict:
    """Run the full pipeline on one audio file. Temporary files are always wiped."""
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cli_", dir=work_root))
    try:
        wav = work / "clean.wav"
        _timed(stages, "convert", stages.convert, src, wav, options.normalize)
        duration = wavio.duration_sec(wav)

        if options.denoise:
            denoised = work / "denoised.wav"
            _timed(stages, "denoise", stages.denoise, wav, denoised)
            wav = denoised

        regions = _timed(stages, "vad", stages.vad, wav)
        words = _timed(stages, "transcribe", stages.transcribe, wav, options.initial_prompt)
        turns = _timed(stages, "diarize", stages.diarize, wav, options.min_speakers, options.max_speakers)

        segments = merge.build_segments(merge.assign_speakers(words, turns))
        segments = postprocess.postprocess(
            segments, speech_regions=regions, filler_removal=options.filler_removal
        )
        segments = merge.label_speakers(segments)  # after cleanup, so dropped junk never takes "Speaker 1"
    finally:
        wipe_dir(work)

    return {
        "language": "en",
        "duration_sec": round(duration, 2),
        "speaker_count": len({s["speaker"] for s in segments}),
        "segments": [_public_segment(s) for s in segments],
        "full_text": "\n".join(f'{s["speaker"]}: {s["text"]}' for s in segments),
        "pipeline": {
            "whisper_model": cfg.whisper_model,
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "device": cfg.device,
            "denoise": options.denoise,
            "normalize": options.normalize,
            "filler_removal": options.filler_removal,
            "timings_sec": stages.timings,
        },
    }


def _public_segment(s: dict) -> dict:
    out = {
        "speaker": s["speaker"],
        "start": round(s["start"], 2),
        "end": round(s["end"], 2),
        "text": s["text"],
        "confidence": round(s["confidence"], 3) if s.get("confidence") is not None else None,
        "words": s["words"],
    }
    if "text_raw" in s:
        out["text_raw"] = s["text_raw"]
    return out
