"""Command-line pipeline: audio file in, transcript JSON out.

    python -m worker.cli recording.wav -o transcript.json [--no-denoise] [--filler-removal]

The output contains the transcript text (sensitive). This tool reads a plain audio file directly and
does NOT go through the guardrail gate: use it for development on approved test audio only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from worker import config  # noqa: E402  (sets offline flags before any ML import)
from worker.pipeline.audio_clean import StageError  # noqa: E402
from worker.pipeline.run import Options, PipelineConfig, default_stages, process_audio  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="input audio file (wav, mp3, m4a, flac, ogg, webm)")
    p.add_argument("-o", "--output", type=Path, help="output JSON file (default: stdout)")
    p.add_argument("--no-denoise", action="store_true", help="skip DeepFilterNet denoising")
    p.add_argument("--no-normalize", action="store_true", help="skip loudness normalisation")
    p.add_argument("--filler-removal", action="store_true", help="remove um/uh and stutter repeats")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--initial-prompt", default=config.INITIAL_PROMPT,
                   help="medical terms to bias Whisper spelling (default: INITIAL_PROMPT / INITIAL_PROMPT_FILE)")
    p.add_argument("--device", default=config.DEVICE, choices=["cpu", "cuda"])
    p.add_argument("--model", default=config.WHISPER_MODEL, help="Whisper model (small, medium, large-v3)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.audio.is_file():
        print(f"error: file not found: {args.audio}", file=sys.stderr)
        return 2

    cfg = PipelineConfig(
        device=args.device,
        whisper_model=args.model,
        compute_type="float16" if args.device == "cuda" else "int8",
        whisper_model_dir=config.WHISPER_MODEL_DIR,
        pyannote_config=config.PYANNOTE_CONFIG,
    )
    options = Options(
        denoise=config.DENOISE and not args.no_denoise,
        normalize=not args.no_normalize,
        filler_removal=args.filler_removal,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        initial_prompt=args.initial_prompt,
    )

    from app.config import get_settings

    try:
        result = process_audio(args.audio, options, default_stages(cfg), cfg, get_settings().tmp_dir)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output} ({result['speaker_count']} speakers, "
              f"{len(result['segments'])} segments, {result['duration_sec']}s audio)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
