"""Clean the transcript text (DESIGN.md 6.5). Pure logic, no ML imports."""
from __future__ import annotations

import re

# Phrases Whisper tends to invent over silence/noise. Dropped only when the segment is also
# low-confidence or lies outside detected speech, so a genuine "Thank you." is kept.
KNOWN_HALLUCINATIONS = {
    "thank you", "thanks", "thank you for watching", "thanks for watching", "bye",
    "you", "subtitles by the amaraorg community",
}
LOW_CONFIDENCE = 0.6

_NON_SPEECH = re.compile(r"^[\s\[\(\*♪♫<]+.*?[\]\)\*♪♫>\s]*$")
_FILLER = re.compile(r"\b(?:um+|uh+|uhm+|erm|er)\b[,.]?\s*", re.IGNORECASE)
_STUTTER_WORDS = {"i", "a", "the", "and", "to", "we", "it", "so", "but", "you", "they", "he", "she", "is", "in", "of"}


def capitalize_first(text: str) -> str:
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def normalize_text(text: str, capitalize: bool = True) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return capitalize_first(text) if capitalize else text


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def is_non_speech(text: str) -> bool:
    stripped = text.strip()
    if not _tokens(stripped):
        return True  # empty or punctuation only
    # Whole segment is a bracketed marker such as [MUSIC], (laughs), ♪ ... ♪
    return bool(_NON_SPEECH.match(stripped)) and bool(re.match(r"^[\[\(\*♪♫<]", stripped))


def is_repetition_loop(text: str) -> bool:
    """True if the segment is the same 1-4 word phrase repeated 5+ times (Whisper loop)."""
    toks = _tokens(text)
    for n in range(1, 5):
        if len(toks) >= 5 * n and all(toks[i] == toks[i - n] for i in range(n, len(toks))):
            return True
    return False


def _in_speech(seg: dict, regions: list[tuple[float, float]]) -> bool:
    return any(min(seg["end"], r1) - max(seg["start"], r0) > 0 for r0, r1 in regions)


def is_hallucination(seg: dict, regions: list[tuple[float, float]] | None) -> bool:
    if is_repetition_loop(seg["text"]):
        return True
    phrase = " ".join(_tokens(seg["text"].replace(".", "")))
    if phrase not in KNOWN_HALLUCINATIONS:
        return False
    low = seg.get("confidence") is not None and seg["confidence"] < LOW_CONFIDENCE
    outside = regions is not None and not _in_speech(seg, regions)
    return low or outside


def remove_fillers(text: str) -> str:
    text = _FILLER.sub("", text)

    def collapse(m: re.Match) -> str:
        word = m.group(1)
        run = len(re.findall(r"[A-Za-z']+", m.group(0)))
        if run >= 3 or word.lower() in _STUTTER_WORDS:
            return word
        return m.group(0)

    text = re.sub(r"\b([A-Za-z']+)(?:[\s,]+\1\b)+", collapse, text, flags=re.IGNORECASE)
    return text


def _merge_adjacent(segments: list[dict], max_gap: float) -> list[dict]:
    merged: list[dict] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if prev and prev["speaker"] == seg["speaker"] and seg["start"] - prev["end"] <= max_gap:
            words = prev["words"] + seg["words"]
            probs = [w["prob"] for w in words if w.get("prob") is not None]
            has_raw = "text_raw" in prev or "text_raw" in seg
            raw = prev.get("text_raw", prev["text"]) + " " + seg.get("text_raw", seg["text"])
            prev.update(
                end=seg["end"],
                text=normalize_text(prev["text"] + " " + seg["text"], capitalize=False),
                words=words,
                confidence=sum(probs) / len(probs) if probs else None,
            )
            if has_raw:
                prev["text_raw"] = raw.strip()
        else:
            merged.append(dict(seg))
    return merged


def postprocess(
    segments: list[dict],
    *,
    speech_regions: list[tuple[float, float]] | None = None,
    filler_removal: bool = False,
    merge_gap: float = 1.0,
) -> list[dict]:
    kept: list[dict] = []
    for seg in segments:
        text = normalize_text(seg["text"], capitalize=False)  # capitalised once, after merging
        candidate = {**seg, "text": text}
        if is_non_speech(text) or is_hallucination(candidate, speech_regions):
            continue
        if filler_removal:
            cleaned = normalize_text(remove_fillers(text), capitalize=False)
            if not _tokens(cleaned):
                continue
            if cleaned != text:
                candidate["text_raw"] = text  # raw text is kept when removal changes it
                candidate["text"] = cleaned
        kept.append(candidate)
    merged = _merge_adjacent(kept, merge_gap)
    for seg in merged:
        seg["text"] = capitalize_first(seg["text"])
        if "text_raw" in seg:
            seg["text_raw"] = capitalize_first(seg["text_raw"])
    return merged
