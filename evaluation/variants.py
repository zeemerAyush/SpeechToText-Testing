"""Experiment variants: the pipeline settings to compare. The first variant is the baseline."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


class VariantError(ValueError):
    """The variants file is invalid. The message says what to fix."""


@dataclass(frozen=True)
class Variant:
    name: str
    denoise: bool = True
    normalize: bool = True
    filler_removal: bool = False
    min_speakers: int | None = None
    max_speakers: int | None = None
    initial_prompt: str | None = None
    whisper_model: str | None = None  # None = the default model


DEFAULT_VARIANTS = [
    Variant("baseline"),
    Variant("no-denoise", denoise=False),
]

_ALLOWED = {f.name for f in fields(Variant)}


def parse_variants(data: object) -> list[Variant]:
    if not isinstance(data, list) or not data:
        raise VariantError("The variants file must contain a non-empty JSON list.")
    variants: list[Variant] = []
    seen: set[str] = set()
    for i, item in enumerate(data):
        where = f"variant {i + 1}"
        if not isinstance(item, dict):
            raise VariantError(f"{where}: expected an object.")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise VariantError(f"{where}: 'name' is required.")
        if name in seen:
            raise VariantError(f"Duplicate variant name '{name}'.")
        unknown = set(item) - _ALLOWED
        if unknown:
            raise VariantError(f"Variant '{name}': unknown option(s) {sorted(unknown)}. Allowed: {sorted(_ALLOWED)}.")
        for key in ("denoise", "normalize", "filler_removal"):
            if key in item and not isinstance(item[key], bool):
                raise VariantError(f"Variant '{name}': '{key}' must be true or false.")
        for key in ("min_speakers", "max_speakers"):
            v = item.get(key)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 10):
                raise VariantError(f"Variant '{name}': '{key}' must be a whole number from 1 to 10.")
        lo, hi = item.get("min_speakers"), item.get("max_speakers")
        if lo and hi and lo > hi:
            raise VariantError(f"Variant '{name}': min_speakers cannot exceed max_speakers.")
        seen.add(name)
        variants.append(Variant(**item))
    return variants


def load_variants(path: Path) -> list[Variant]:
    try:
        return parse_variants(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise VariantError(f"Cannot read variants file {path}: {exc}") from exc
