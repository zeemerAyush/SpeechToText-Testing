"""Run every variant over every recording and score it against the reference."""
from __future__ import annotations

import dataclasses
import gc
import json
import time
from pathlib import Path
from typing import Callable

from evaluation.dataset import Recording
from evaluation.metrics import (
    MetricsUnavailable,
    diarization_error,
    speaker_count,
    term_recall,
    word_errors,
)
from evaluation.variants import Variant
from worker.pipeline.run import Options, PipelineConfig, Stages, process_audio

StagesFactory = Callable[[PipelineConfig], Stages]


def release_models() -> None:
    """Free GPU memory between model sizes (best effort; torch may not be installed)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def run_one(
    rec: Recording,
    variant: Variant,
    stages: Stages,
    cfg: PipelineConfig,
    work_root: Path,
    *,
    terms: list[str],
    collar: float,
    strip_fillers: bool,
    der_fn: Callable,
    save_dir: Path | None,
) -> dict:
    run: dict = {"recording": rec.name, "variant": variant.name, "ok": False}
    options = Options(
        denoise=variant.denoise, normalize=variant.normalize, filler_removal=variant.filler_removal,
        min_speakers=variant.min_speakers, max_speakers=variant.max_speakers,
        initial_prompt=variant.initial_prompt,
    )
    t0 = time.perf_counter()
    try:
        out = process_audio(rec.audio, options, stages, cfg, work_root)
    except Exception as exc:
        run["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return run
    elapsed = time.perf_counter() - t0

    hyp_text = " ".join(s["text"] for s in out["segments"])
    wer = word_errors(rec.reference.text, hyp_text, strip_fillers)
    run.update(
        ok=True,
        audio_sec=out["duration_sec"],
        elapsed_sec=round(elapsed, 2),
        rtf=round(elapsed / out["duration_sec"], 3) if out["duration_sec"] else None,
        stage_sec=out["pipeline"].get("timings_sec", {}),
        wer={"substitutions": wer.substitutions, "deletions": wer.deletions, "insertions": wer.insertions,
             "hits": wer.hits},
    )
    if terms:
        t = term_recall(rec.reference.text, hyp_text, terms, strip_fillers)
        run["terms"] = {"found": t.found, "total": t.total, "missed": t.missed}
    if rec.reference.has_diarization:
        run["speakers"] = {"reference": speaker_count(rec.reference.segments),
                           "hypothesis": speaker_count(out["segments"])}
        try:
            d = der_fn(rec.reference.segments, out["segments"], collar)
            run["der"] = {"total": d.total, "missed": d.missed, "false_alarm": d.false_alarm,
                          "confusion": d.confusion}
        except MetricsUnavailable as exc:
            run["der_error"] = str(exc)
    if save_dir is not None:  # transcripts contain patient speech: only written when explicitly requested
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{_safe_name(variant.name)}__{_safe_name(rec.name)}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return run


def evaluate(
    recordings: list[Recording],
    variants: list[Variant],
    stages_factory: StagesFactory,
    base_cfg: PipelineConfig,
    work_root: Path,
    *,
    terms: list[str] | None = None,
    collar: float = 0.25,
    strip_fillers: bool = True,
    der_fn: Callable = diarization_error,
    save_dir: Path | None = None,
    progress: Callable[[str], None] = lambda msg: None,
) -> list[dict]:
    """Return one result dict per (variant, recording). A failing run is recorded and does not stop the rest.

    Variants are grouped by Whisper model so each model is loaded once and freed before the next one.
    """
    groups: dict[str, list[Variant]] = {}
    for v in variants:
        groups.setdefault(v.whisper_model or base_cfg.whisper_model, []).append(v)

    runs: list[dict] = []
    total, done = len(variants) * len(recordings), 0
    for model, group in groups.items():
        cfg = dataclasses.replace(
            base_cfg, whisper_model=model,
            # a pinned local model folder only applies to the default model
            whisper_model_dir=base_cfg.whisper_model_dir if model == base_cfg.whisper_model else None,
        )
        stages = stages_factory(cfg)
        stages.redact = None  # redaction is not being measured; skip loading spaCy
        try:
            for variant in group:
                for rec in recordings:
                    done += 1
                    progress(f"[{done}/{total}] {variant.name} / {rec.name} ({model})")
                    runs.append(run_one(
                        rec, variant, stages, cfg, work_root, terms=terms or [], collar=collar,
                        strip_fillers=strip_fillers, der_fn=der_fn, save_dir=save_dir,
                    ))
        finally:
            del stages
            release_models()
    return runs
