"""Command line: score the pipeline against hand-corrected references and compare settings.

    python -m evaluation --data eval_data --out eval_runs/run1 [--variants variants.json] [--terms terms.txt]

See evaluation/README.md for how to prepare the data. Reports contain only scores unless --save-transcripts is
given; the dataset and the output folder hold patient speech, so keep them out of git (they are git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from worker import config  # noqa: E402  (sets offline flags before any ML import)
from worker.pipeline.audio_clean import StageError  # noqa: E402
from worker.pipeline.run import PipelineConfig  # noqa: E402

from evaluation import report  # noqa: E402
from evaluation.dataset import DatasetError, load_dataset, load_terms  # noqa: E402
from evaluation.metrics import diarization_error  # noqa: E402
from evaluation.runner import evaluate  # noqa: E402
from evaluation.variants import DEFAULT_VARIANTS, VariantError, load_variants  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="folder with audio files and <name>.ref.json/.ref.txt")
    p.add_argument("--out", type=Path, help="output folder (default: eval_runs/<timestamp>)")
    p.add_argument("--variants", type=Path, help="JSON list of settings to compare (default: baseline vs no-denoise)")
    p.add_argument("--terms", type=Path, help="text file of medical terms, one per line, for term recall")
    p.add_argument("--collar", type=float, default=0.25, help="DER forgiveness around boundaries, seconds")
    p.add_argument("--count-fillers", action="store_true", help="count um/uh as words in WER (default: ignore)")
    p.add_argument("--save-transcripts", action="store_true", help="also save each hypothesis transcript (sensitive)")
    p.add_argument("--device", default=config.DEVICE, choices=["cpu", "cuda"])
    p.add_argument("--model", default=config.WHISPER_MODEL, help="default Whisper model for variants that set none")
    return p


def real_stages_factory(cfg: PipelineConfig):
    from worker.pipeline.run import default_stages

    return default_stages(cfg)


def main(argv: list[str] | None = None, stages_factory=real_stages_factory, der_fn=diarization_error) -> int:
    args = build_parser().parse_args(argv)
    try:
        recordings, skipped = load_dataset(args.data)
        variants = load_variants(args.variants) if args.variants else DEFAULT_VARIANTS
        terms = load_terms(args.terms) if args.terms else []
    except (DatasetError, VariantError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = args.out or Path("eval_runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(
        device=args.device, whisper_model=args.model,
        compute_type="float16" if args.device == "cuda" else "int8",
        whisper_model_dir=config.WHISPER_MODEL_DIR, pyannote_config=config.PYANNOTE_CONFIG,
    )

    from app.config import get_settings

    try:
        runs = evaluate(
            recordings, variants, stages_factory, cfg, get_settings().tmp_dir,
            terms=terms, collar=args.collar, strip_fillers=not args.count_fillers, der_fn=der_fn,
            save_dir=out_dir / "transcripts" if args.save_transcripts else None,
            progress=lambda m: print(m, file=sys.stderr),
        )
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ok_audio = {r["recording"]: r["audio_sec"] for r in runs if r["ok"]}
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"), "recordings": len(recordings),
        "audio_sec": sum(ok_audio.values()), "skipped": skipped, "device": args.device,
        "default_model": args.model, "collar": args.collar, "strip_fillers": not args.count_fillers,
        "variants": [v.__dict__ for v in variants],
    }
    results = report.build_results(runs, [v.name for v in variants], meta)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report.render_markdown(results), encoding="utf-8")
    (out_dir / "runs.csv").write_text(report.render_csv(runs), encoding="utf-8", newline="")
    print(f"Wrote {out_dir / 'report.md'}, results.json and runs.csv", file=sys.stderr)
    return 0 if all(r["ok"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
