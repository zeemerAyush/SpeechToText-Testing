"""Load an evaluation dataset: audio files with hand-corrected reference transcripts.

Layout (one pair per recording, same base name):

    eval_data/
      visit01.wav
      visit01.ref.json      # segments with speaker/start/end/text  -> WER and DER
      visit02.mp3
      visit02.ref.txt       # plain text only                       -> WER only
      terms.txt             # optional vocabulary for term recall (one term per line)

A ``.ref.json`` has the same shape as the pipeline's own output, so the usual workflow is: run the pipeline,
save its JSON, then correct the text, speaker labels and times by hand while listening.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm"}


class DatasetError(ValueError):
    """The dataset or one of its reference files is unusable. The message says what to fix."""


@dataclass
class Reference:
    text: str                             # whole transcript in time order, for WER
    segments: list[dict] | None = None    # {speaker, start, end, text}; None when only text is available

    @property
    def has_diarization(self) -> bool:
        return bool(self.segments)


@dataclass
class Recording:
    name: str
    audio: Path
    reference: Reference


def load_reference_json(path: Path) -> Reference:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{path.name}: cannot read as JSON ({exc}).") from exc
    raw = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise DatasetError(f"{path.name}: expected an object with a non-empty 'segments' list.")

    segments = []
    for i, s in enumerate(raw):
        where = f"{path.name} segment {i + 1}"
        if not isinstance(s, dict) or not isinstance(s.get("text"), str):
            raise DatasetError(f"{where}: every segment needs a 'text' string.")
        missing = [k for k in ("speaker", "start", "end") if k not in s]
        if missing:
            raise DatasetError(f"{where}: missing {', '.join(missing)} (needed to measure speaker separation).")
        try:
            start, end = float(s["start"]), float(s["end"])
        except (TypeError, ValueError):
            raise DatasetError(f"{where}: start/end must be numbers (seconds).") from None
        if start < 0 or end < start:
            raise DatasetError(f"{where}: invalid times start={start}, end={end}.")
        segments.append({"speaker": str(s["speaker"]), "start": start, "end": end, "text": s["text"]})

    segments.sort(key=lambda s: (s["start"], s["end"]))
    return Reference(text=" ".join(s["text"].strip() for s in segments), segments=segments)


def load_reference_text(path: Path) -> Reference:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DatasetError(f"{path.name}: cannot read ({exc}).") from exc
    if not text:
        raise DatasetError(f"{path.name}: the reference transcript is empty.")
    return Reference(text=text)


def load_dataset(folder: Path) -> tuple[list[Recording], list[str]]:
    """Return (recordings, names of audio files skipped because they have no reference)."""
    if not folder.is_dir():
        raise DatasetError(f"Dataset folder not found: {folder}")
    recordings: list[Recording] = []
    skipped: list[str] = []
    for audio in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS):
        json_ref = audio.with_name(audio.stem + ".ref.json")
        text_ref = audio.with_name(audio.stem + ".ref.txt")
        if json_ref.is_file():
            ref = load_reference_json(json_ref)
        elif text_ref.is_file():
            ref = load_reference_text(text_ref)
        else:
            skipped.append(audio.name)
            continue
        recordings.append(Recording(name=audio.stem, audio=audio, reference=ref))
    if not recordings:
        raise DatasetError(
            f"No recordings with a reference found in {folder}. Each audio file needs "
            "<name>.ref.json or <name>.ref.txt next to it."
        )
    return recordings, skipped


def load_terms(path: Path) -> list[str]:
    """One term or phrase per line; blank lines and lines starting with # are ignored."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Cannot read terms file {path}: {exc}") from exc
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
