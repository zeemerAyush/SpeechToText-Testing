"""File validation: signature (magic bytes) vs extension, size, duration, decodability."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mutagen

# extension -> canonical container type
EXTENSION_TYPES = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".m4a": "m4a",
    ".flac": "flac",
    ".ogg": "ogg",
    ".opus": "ogg",
    ".webm": "webm",
}


class ValidationError(Exception):
    """The file failed a gate check. The message is safe to show to the user."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ValidationResult:
    container: str
    duration_sec: float | None
    checked_with: str  # "ffprobe" or "python-fallback"


def detect_type(head: bytes) -> str | None:
    """Identify the container from the first bytes of the file."""
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "m4a"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    return None


def check_signature(path: Path, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    expected = EXTENSION_TYPES.get(ext)
    if expected is None:
        raise ValidationError(f"Unsupported file extension '{ext or '(none)'}'.")
    with open(path, "rb") as fh:
        detected = detect_type(fh.read(16))
    if detected is None:
        raise ValidationError("File content is not a recognised audio format.")
    if detected != expected:
        raise ValidationError(f"File extension '{ext}' does not match its content ({detected}).")
    return detected


def _probe_with_ffprobe(path: Path, ffprobe: str) -> float | None:
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise ValidationError("File could not be read as audio.")
    info = json.loads(proc.stdout or "{}")
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise ValidationError("File contains no audio stream.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:  # full decode proves the file is not malformed
        dec = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
            capture_output=True, text=True, timeout=600,
        )
        if dec.returncode != 0 or dec.stderr.strip():
            raise ValidationError("File is malformed and could not be decoded.")
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return None


def _probe_with_mutagen(path: Path) -> float | None:
    try:
        audio = mutagen.File(path)
    except Exception as exc:  # mutagen raises many parser-specific errors
        raise ValidationError("File could not be read as audio.") from exc
    if audio is None or audio.info is None:
        raise ValidationError("File could not be read as audio.")
    length = getattr(audio.info, "length", None)
    if length is None or length <= 0:
        raise ValidationError("File contains no audio.")
    return float(length)


def validate_audio(
    path: Path,
    filename: str,
    *,
    max_bytes: int,
    max_duration_sec: int,
    require_ffprobe: bool,
) -> ValidationResult:
    size = path.stat().st_size
    if size == 0:
        raise ValidationError("File is empty.")
    if size > max_bytes:
        raise ValidationError(f"File exceeds the maximum size of {max_bytes // (1024 * 1024)} MB.", 413)

    container = check_signature(path, filename)

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        duration, tool = _probe_with_ffprobe(path, ffprobe), "ffprobe"
    elif require_ffprobe:
        raise ValidationError("Server is misconfigured: ffprobe is required but not installed.")
    elif container == "webm":
        # No pure-Python WebM parser; duration cannot be checked without ffprobe.
        duration, tool = None, "python-fallback"
    else:
        duration, tool = _probe_with_mutagen(path), "python-fallback"

    if duration is not None and duration > max_duration_sec:
        raise ValidationError(f"Recording exceeds the maximum duration of {max_duration_sec // 60} minutes.")
    return ValidationResult(container=container, duration_sec=duration, checked_with=tool)
