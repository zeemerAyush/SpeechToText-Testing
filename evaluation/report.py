"""Combine per-recording results into per-variant scores, paired comparisons and a readable report."""
from __future__ import annotations

import csv
import io
import statistics
from collections import defaultdict
from typing import Callable

from evaluation.metrics import DerResult, TermResult, WerResult

SMALL_SAMPLE_RECORDINGS = 5
SMALL_SAMPLE_SECONDS = 600


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def delta_points(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.1f}"


def _wer(run: dict) -> WerResult:
    return WerResult(**run["wer"])


def _der(run: dict) -> DerResult | None:
    return DerResult(**run["der"]) if "der" in run else None


# ---------------------------------------------------------------- aggregation

def aggregate(runs: list[dict], variant_names: list[str]) -> dict[str, dict]:
    """Per variant: totals across recordings (errors / reference words, not an average of percentages)."""
    out: dict[str, dict] = {}
    for name in variant_names:
        mine = [r for r in runs if r["variant"] == name]
        ok = [r for r in mine if r["ok"]]
        wer = sum((_wer(r) for r in ok), WerResult())
        ders = [d for d in (_der(r) for r in ok) if d is not None]
        terms = [TermResult(**r["terms"]) for r in ok if "terms" in r]
        spk = [r["speakers"] for r in ok if "speakers" in r]
        rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
        out[name] = {
            "recordings": len(ok),
            "failed": len(mine) - len(ok),
            "audio_sec": sum(r["audio_sec"] for r in ok),
            "wer": wer.wer,
            "wer_counts": {"substitutions": wer.substitutions, "deletions": wer.deletions,
                           "insertions": wer.insertions, "ref_words": wer.ref_words},
            "der": sum(ders, DerResult()).der if ders else None,
            "term_recall": sum(terms, TermResult()).recall if terms else None,
            "speaker_count_accuracy": (
                sum(1 for s in spk if s["reference"] == s["hypothesis"]) / len(spk) if spk else None
            ),
            "rtf_median": statistics.median(rtfs) if rtfs else None,
        }
    return out


def paired_comparison(runs: list[dict], baseline: str, other: str,
                      metric: Callable[[dict], float | None]) -> dict | None:
    """Per-recording differences (other - baseline); negative means ``other`` is better (fewer errors)."""
    base = {r["recording"]: r for r in runs if r["variant"] == baseline and r["ok"]}
    deltas = []
    for r in runs:
        if r["variant"] != other or not r["ok"] or r["recording"] not in base:
            continue
        a, b = metric(base[r["recording"]]), metric(r)
        if a is not None and b is not None:
            deltas.append(b - a)
    if not deltas:
        return None
    return {
        "n": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "better": sum(1 for d in deltas if d < 0),
        "worse": sum(1 for d in deltas if d > 0),
        "same": sum(1 for d in deltas if d == 0),
    }


def _wer_metric(run: dict) -> float | None:
    return _wer(run).wer


def _der_metric(run: dict) -> float | None:
    d = _der(run)
    return d.der if d else None


def build_results(runs: list[dict], variant_names: list[str], meta: dict) -> dict:
    baseline = variant_names[0]
    comparisons = {}
    for name in variant_names[1:]:
        comparisons[name] = {
            "wer": paired_comparison(runs, baseline, name, _wer_metric),
            "der": paired_comparison(runs, baseline, name, _der_metric),
        }
    return {
        "meta": meta, "baseline": baseline, "variants": aggregate(runs, variant_names),
        "comparison_vs_baseline": comparisons, "runs": runs,
    }


# ---------------------------------------------------------------- rendering

def _warnings(results: dict) -> list[str]:
    meta, variants = results["meta"], results["variants"]
    notes = []
    n, secs = meta["recordings"], meta["audio_sec"]
    if n < SMALL_SAMPLE_RECORDINGS or secs < SMALL_SAMPLE_SECONDS:
        notes.append(
            f"Small sample ({n} recording(s), {secs / 60:.1f} min of audio). Treat differences of a point or two "
            "as noise; add more recordings before changing settings."
        )
    if meta.get("skipped"):
        notes.append("Skipped (no reference file): " + ", ".join(meta["skipped"]) + ".")
    for name, v in variants.items():
        if v["failed"]:
            notes.append(f"Variant '{name}': {v['failed']} run(s) failed and are excluded from its scores (see runs.csv).")
    errs = {r["der_error"] for r in results["runs"] if "der_error" in r}
    if errs:
        notes.append("Diarization error rate was not computed: " + "; ".join(sorted(errs)))
    if any(r["ok"] and "der" not in r and "der_error" not in r for r in results["runs"]):
        notes.append("Some recordings have a text-only reference, so DER excludes them.")
    return notes


def render_markdown(results: dict) -> str:
    meta, variants, base = results["meta"], results["variants"], results["baseline"]
    names = list(variants)
    scored = [n for n in names if variants[n]["wer"] is not None and not variants[n]["failed"]]
    best = min(scored, key=lambda n: variants[n]["wer"]) if len(scored) > 1 else None

    lines = [
        "# Evaluation report",
        "",
        f"- Recordings: {meta['recordings']} ({meta['audio_sec'] / 60:.1f} min of audio)",
        f"- Baseline: **{base}**; device: {meta['device']}; collar: {meta['collar']} s; "
        f"fillers {'ignored' if meta['strip_fillers'] else 'counted'} in WER",
        "",
    ]
    notes = _warnings(results)
    if notes:
        lines += ["## Read first", ""] + [f"- {n}" for n in notes] + [""]

    lines += ["## Summary", "",
              "| Variant | WER | Change vs baseline | DER | Term recall | Speaker count right | Median RTF |",
              "|---|---|---|---|---|---|---|"]
    for n in names:
        v = variants[n]
        base_wer = variants[base]["wer"]
        label = f"{n} (baseline)" if n == base else (f"{n} (lowest WER)" if n == best else n)
        # Same pooled basis as the WER column, so the two always agree.
        change = "-" if n == base or v["wer"] is None or base_wer is None else f"{delta_points(v['wer'] - base_wer)} pts"
        rtf = "n/a" if v["rtf_median"] is None else f"{v['rtf_median']:.2f}"
        lines.append(
            f"| {label} | {pct(v['wer'])} | {change} | {pct(v['der'])} | {pct(v['term_recall'])} | "
            f"{pct(v['speaker_count_accuracy'])} | {rtf} |"
        )
    lines += ["",
              "WER = word error rate and DER = diarization error rate (lower is better). Scores pool every word "
              "(or second of speech) across all recordings. RTF = processing time / audio "
              "time (below 1 is faster than real time; the first file per model includes model loading, so the median "
              "is reported).", ""]

    if results["comparison_vs_baseline"]:
        lines += ["## Paired comparison with the baseline", "",
                  "Differences are computed per recording (variant minus baseline), so a negative number means the "
                  "variant made fewer errors. Here every recording counts equally, whatever its length, so this can "
                  "disagree with the pooled summary when recordings differ in length; look at Better / Worse to see "
                  "whether a change helps consistently.", "",
                  "| Variant | Metric | Recordings | Mean per-recording change (pts) | Better | Worse | Same |", "|---|---|---|---|---|---|---|"]
        for n, c in results["comparison_vs_baseline"].items():
            for key, label in (("wer", "WER"), ("der", "DER")):
                if c[key]:
                    x = c[key]
                    lines.append(f"| {n} | {label} | {x['n']} | {delta_points(x['mean_delta'])} | "
                                 f"{x['better']} | {x['worse']} | {x['same']} |")
        lines.append("")

    by_rec: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results["runs"]:
        by_rec[r["recording"]][r["variant"]] = r
    lines += ["## WER by recording", "", "| Recording | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for rec, row in by_rec.items():
        cells = []
        for n in names:
            r = row.get(n)
            cells.append("-" if r is None else (pct(_wer(r).wer) if r["ok"] else "failed"))
        lines.append(f"| {rec} | " + " | ".join(cells) + " |")
    lines.append("")

    for n in names:
        missed: dict[str, int] = defaultdict(int)
        for r in results["runs"]:
            if r["variant"] == n and r["ok"]:
                for term, k in r.get("terms", {}).get("missed", {}).items():
                    missed[term] += k
        if missed:
            top = sorted(missed.items(), key=lambda kv: -kv[1])[:10]
            lines += [f"### Most-missed terms: {n}", ""] + [f"- {t} ({k}x)" for t, k in top] + [""]

    lines += [
        "## How to use this",
        "",
        "- **Denoise:** keep whichever of baseline / no-denoise has the lower WER (the paired table shows whether it "
        "wins consistently). Set the default with `DENOISE=0|1`; users can still override it per upload.",
        "- **Model size:** weigh the WER gain of a larger Whisper model against its RTF. Set `WHISPER_MODEL`.",
        "- **Medical prompt:** if a prompted variant raises term recall without hurting WER, save the prompt text in a "
        "file and set `INITIAL_PROMPT_FILE` (or `INITIAL_PROMPT`).",
        "- **Speaker hint:** if DER improves with a fixed number of speakers, tell uploaders to use the *Expected "
        "number of speakers* option for such recordings.",
    ]
    return "\n".join(lines) + "\n"


def render_csv(runs: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["variant", "recording", "ok", "error", "audio_sec", "elapsed_sec", "rtf", "wer", "substitutions",
                "deletions", "insertions", "ref_words", "der", "speakers_ref", "speakers_hyp", "term_recall"])
    for r in runs:
        if not r["ok"]:
            w.writerow([r["variant"], r["recording"], False, r.get("error", "")])
            continue
        wer = _wer(r)
        d = _der(r)
        t = TermResult(**r["terms"]) if "terms" in r else None
        w.writerow([
            r["variant"], r["recording"], True, "", r["audio_sec"], r["elapsed_sec"], r.get("rtf"),
            "" if wer.wer is None else round(wer.wer, 4), wer.substitutions, wer.deletions, wer.insertions,
            wer.ref_words, "" if not d or d.der is None else round(d.der, 4),
            r.get("speakers", {}).get("reference", ""), r.get("speakers", {}).get("hypothesis", ""),
            "" if not t or t.recall is None else round(t.recall, 4),
        ])
    return buf.getvalue()
