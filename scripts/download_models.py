"""Download every model once, with internet access. Afterwards the worker runs fully offline.

    HF_TOKEN=hf_xxx python scripts/download_models.py [--whisper large-v3] [--skip pyannote ...]

What it fetches (into ./models unless --models-dir says otherwise):
  whisper       faster-whisper model         -> models/whisper/<name>            (set WHISPER_MODEL_DIR)
  pyannote      diarization 3.1 + 2 submodels -> models/pyannote/config.yaml      (set PYANNOTE_CONFIG)
                (needs a Hugging Face token whose owner has accepted the model conditions, see below)
  deepfilternet noise-removal model          -> its own cache (small)
  spacy         en_core_web_lg for Presidio  -> installed into the venv
  silero        voice-activity model is bundled with the silero-vad package (only checked)

Each step reports OK or FAILED on its own, so one failure does not stop the rest. Re-run to retry.
The token is read from the HF_TOKEN environment variable (never from the command line), or from a previous
`huggingface-cli login`. Run `python scripts/check_models.py` afterwards to verify everything loads offline.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PYANNOTE_REPOS = {
    "pipeline": "pyannote/speaker-diarization-3.1",
    "segmentation": "pyannote/segmentation-3.0",
    "embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
}
STEPS = ["whisper", "pyannote", "deepfilternet", "spacy", "silero"]


def rewrite_pyannote_config(text: str, segmentation: str, embedding: str) -> str:
    """Point the pipeline config at local checkpoint files instead of Hugging Face repo names.

    Only the two lines that carry a value are changed (``segmentation: pyannote/segmentation-3.0`` and
    ``embedding: ...``); the nested ``segmentation:`` block under ``params`` has no value and is left alone.
    """
    out = text
    for key, value in (("segmentation", segmentation), ("embedding", embedding)):
        pattern = re.compile(rf"^(\s*{key}:\s+)\S+[ \t]*$", re.MULTILINE)
        out, n = pattern.subn(lambda m, v=value: f'{m.group(1)}"{v}"', out)
        if n != 1:
            raise ValueError(
                f"Expected exactly one '{key}: <repo>' line in the pipeline config but found {n}; "
                "the file format may have changed."
            )
    return out


def step_whisper(name: str, models_dir: Path) -> str:
    from faster_whisper import download_model

    target = models_dir / "whisper" / name
    path = download_model(name, output_dir=str(target))
    return f"saved to {path}"


def _gated_error():
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import GatedRepoError
    return GatedRepoError


def step_pyannote(models_dir: Path, token: str | None) -> str:
    from huggingface_hub import snapshot_download

    gated = _gated_error()
    base = models_dir / "pyannote"
    dirs: dict[str, Path] = {}
    for key, repo in PYANNOTE_REPOS.items():
        dest = base / key
        try:
            snapshot_download(repo_id=repo, local_dir=str(dest), token=token)
        except gated as exc:
            raise RuntimeError(
                f"Access to {repo} is gated. Sign in at https://huggingface.co/{repo}, accept the conditions with "
                "the account that owns your token, then run this script again."
            ) from exc
        dirs[key] = dest

    source = dirs["pipeline"] / "config.yaml"
    if not source.is_file():
        raise RuntimeError(f"{source} was not downloaded; check access to {PYANNOTE_REPOS['pipeline']}.")
    checkpoints = {}
    for key in ("segmentation", "embedding"):
        ckpt = (dirs[key] / "pytorch_model.bin").resolve()
        if not ckpt.is_file():
            raise RuntimeError(f"{ckpt} is missing; the {PYANNOTE_REPOS[key]} download is incomplete.")
        checkpoints[key] = ckpt.as_posix()

    out = base / "config.yaml"
    out.write_text(
        rewrite_pyannote_config(source.read_text(encoding="utf-8"), checkpoints["segmentation"],
                                checkpoints["embedding"]),
        encoding="utf-8",
    )
    return f"offline pipeline config written to {out}"


def step_deepfilternet() -> str:
    from df.enhance import init_df

    init_df()  # downloads the default model into DeepFilterNet's cache on first use
    return "model cached"


def step_spacy(model: str) -> str:
    subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
    return f"{model} installed"


def step_silero() -> str:
    from silero_vad import load_silero_vad

    load_silero_vad()
    return "bundled model loads"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--whisper", default="large-v3", help="Whisper model size or Hugging Face repo")
    p.add_argument("--spacy-model", default="en_core_web_lg")
    p.add_argument("--skip", nargs="*", default=[], choices=STEPS, help="steps to skip")
    args = p.parse_args(argv)

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("note: HF_HUB_OFFLINE=1 is set in your shell; clearing it for this download.")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    token = os.environ.get("HF_TOKEN") or None

    actions = {
        "whisper": lambda: step_whisper(args.whisper, args.models_dir),
        "pyannote": lambda: step_pyannote(args.models_dir, token),
        "deepfilternet": step_deepfilternet,
        "spacy": lambda: step_spacy(args.spacy_model),
        "silero": step_silero,
    }
    failed = []
    for name in STEPS:
        if name in args.skip:
            print(f"[skip]   {name}")
            continue
        print(f"[run]    {name} ...", flush=True)
        try:
            print(f"[OK]     {name}: {actions[name]()}")
        except Exception as exc:  # report and carry on with the other models
            failed.append(name)
            print(f"[FAILED] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    whisper_dir = (args.models_dir / "whisper" / args.whisper).resolve().as_posix()
    pyannote_cfg = (args.models_dir / "pyannote" / "config.yaml").resolve().as_posix()
    print("\nAdd these to .env (the worker reads them from there):\n")
    print(f"  WHISPER_MODEL={args.whisper}\n  WHISPER_MODEL_DIR={whisper_dir}\n  PYANNOTE_CONFIG={pyannote_cfg}")
    print(f"  SPACY_MODEL={args.spacy_model}\n  DEVICE=cuda")
    if failed:
        print(f"\nFailed steps: {', '.join(failed)}. Fix the error above and re-run (finished steps are quick).",
              file=sys.stderr)
        if "deepfilternet" in failed:
            print("DeepFilterNet is optional: set DENOISE=0 in .env to run without it.", file=sys.stderr)
        return 1
    print("\nNext: python scripts/check_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
