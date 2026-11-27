"""Quality metrics: word error rate (jiwer), diarization error rate (pyannote.metrics), medical-term recall.

All results keep raw counts, not just percentages, so many recordings can be combined correctly
(total errors / total reference words, rather than an average of per-file percentages).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class MetricsUnavailable(RuntimeError):
    """A metric library is not installed."""


# ---------------------------------------------------------------- text normalisation

_FILLERS = re.compile(r"\b(?:um+|uh+|uhm+|erm|er)\b", re.IGNORECASE)


def normalize_text(text: str, strip_fillers: bool = True) -> str:
    """Lower-case, drop punctuation and (optionally) fillers, so only the words are compared."""
    t = text.lower().replace("’", "'")
    if strip_fillers:
        t = _FILLERS.sub(" ", t)
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", t)  # stray quote marks, keep "don't"
    return " ".join(t.split())


# ---------------------------------------------------------------- word error rate

@dataclass(frozen=True)
class WerResult:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def ref_words(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float | None:
        """None when the reference has no words (the rate is undefined)."""
        return self.errors / self.ref_words if self.ref_words else None

    def __add__(self, other: "WerResult") -> "WerResult":
        return WerResult(
            self.substitutions + other.substitutions, self.deletions + other.deletions,
            self.insertions + other.insertions, self.hits + other.hits,
        )


def word_errors(reference: str, hypothesis: str, strip_fillers: bool = True) -> WerResult:
    ref = normalize_text(reference, strip_fillers)
    hyp = normalize_text(hypothesis, strip_fillers)
    if not ref:
        return WerResult(insertions=len(hyp.split()))
    if not hyp:
        return WerResult(deletions=len(ref.split()))
    try:
        import jiwer
    except ImportError as exc:
        raise MetricsUnavailable("jiwer is not installed (pip install jiwer).") from exc
    out = jiwer.process_words(ref, hyp)
    return WerResult(out.substitutions, out.deletions, out.insertions, out.hits)


# ---------------------------------------------------------------- medical term recall

@dataclass
class TermResult:
    found: int = 0
    total: int = 0
    missed: dict[str, int] = field(default_factory=dict)

    @property
    def recall(self) -> float | None:
        return self.found / self.total if self.total else None

    def __add__(self, other: "TermResult") -> "TermResult":
        missed = dict(self.missed)
        for term, n in other.missed.items():
            missed[term] = missed.get(term, 0) + n
        return TermResult(self.found + other.found, self.total + other.total, missed)


def term_recall(reference: str, hypothesis: str, terms: list[str], strip_fillers: bool = True) -> TermResult:
    """Of the vocabulary terms spoken in the reference, how many did the hypothesis get right?

    A term is counted once per occurrence in the reference; the hypothesis earns credit for at most as many
    occurrences as it contains (exact word or phrase match after normalisation).
    """
    ref = normalize_text(reference, strip_fillers)
    hyp = normalize_text(hypothesis, strip_fillers)
    result = TermResult()
    for raw in terms:
        term = normalize_text(raw, strip_fillers=False)
        if not term:
            continue
        pattern = rf"(?<!\S){re.escape(term)}(?!\S)"
        in_ref = len(re.findall(pattern, ref))
        if not in_ref:
            continue
        hit = min(in_ref, len(re.findall(pattern, hyp)))
        result.total += in_ref
        result.found += hit
        if hit < in_ref:
            result.missed[term] = in_ref - hit
    return result


# ---------------------------------------------------------------- diarization error rate

@dataclass(frozen=True)
class DerResult:
    total: float = 0.0          # seconds of reference speech scored
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0

    @property
    def errors(self) -> float:
        return self.missed + self.false_alarm + self.confusion

    @property
    def der(self) -> float | None:
        return self.errors / self.total if self.total else None

    def __add__(self, other: "DerResult") -> "DerResult":
        return DerResult(self.total + other.total, self.missed + other.missed,
                         self.false_alarm + other.false_alarm, self.confusion + other.confusion)


def diarization_error(
    reference: list[dict], hypothesis: list[dict], collar: float = 0.25, skip_overlap: bool = False
) -> DerResult:
    """DER with the optimal speaker mapping (labels need not match). Segments: {start, end, speaker}.

    ``collar`` forgives errors within this many seconds of each reference boundary (0.25 is the common choice).
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError as exc:
        raise MetricsUnavailable("pyannote.metrics is not installed (it comes with pyannote.audio).") from exc

    def to_annotation(segments: list[dict]) -> "Annotation":
        ann = Annotation()
        for s in segments:
            if s["end"] > s["start"]:  # zero-length segments are invalid
                ann[Segment(s["start"], s["end"])] = str(s["speaker"])
        return ann

    detail = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)(
        to_annotation(reference), to_annotation(hypothesis), detailed=True
    )
    return DerResult(
        total=float(detail["total"]), missed=float(detail["missed detection"]),
        false_alarm=float(detail["false alarm"]), confusion=float(detail["confusion"]),
    )


def speaker_count(segments: list[dict]) -> int:
    return len({s["speaker"] for s in segments})
