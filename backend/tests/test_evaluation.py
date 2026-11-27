import json
from pathlib import Path

import pytest

from evaluation import report
from evaluation.cli import main as eval_main
from evaluation.dataset import DatasetError, Recording, Reference, load_dataset, load_reference_json, load_terms
from evaluation.metrics import (
    DerResult,
    MetricsUnavailable,
    TermResult,
    WerResult,
    diarization_error,
    normalize_text,
    speaker_count,
    term_recall,
    word_errors,
)
from evaluation.runner import evaluate
from evaluation.variants import DEFAULT_VARIANTS, Variant, VariantError, load_variants, parse_variants
from tests.test_pipeline_run import fake_stages
from worker.pipeline.run import PipelineConfig


# ====================================================================== metrics

def test_normalize_text():
    assert normalize_text("Hello, Doctor!  It's 3:30 -- don't worry.") == "hello doctor it's 3 30 don't worry"
    assert normalize_text("Um, I uh think so") == "i think so"
    assert normalize_text("Um, I uh think so", strip_fillers=False) == "um i uh think so"
    assert normalize_text("‘quoted’ word") == "quoted word"


def test_word_errors_counts_and_rate():
    r = word_errors("the cat sat on the mat", "the cat sit on mat today")
    assert (r.ref_words, r.errors) == (6, 3)
    assert r.wer == pytest.approx(0.5)
    assert r.substitutions + r.deletions + r.hits == 6


def test_word_errors_perfect_and_ignores_case_and_punctuation():
    r = word_errors("Good morning, Doctor.", "good morning doctor")
    assert r.errors == 0 and r.wer == 0.0


def test_word_errors_empty_sides():
    assert word_errors("a b c", "").deletions == 3
    assert word_errors("a b c", "").wer == 1.0
    empty_ref = word_errors("", "x y")
    assert empty_ref.insertions == 2 and empty_ref.wer is None  # rate undefined without reference words
    assert word_errors("", "").errors == 0


def test_fillers_ignored_by_default_but_can_be_counted():
    assert word_errors("so I went home", "so um I went home").errors == 0
    assert word_errors("so I went home", "so um I went home", strip_fillers=False).insertions == 1


def test_wer_combines_by_totals_not_by_averaging_percentages():
    small = WerResult(substitutions=1, hits=9)      # 10% of 10 words
    big = WerResult(hits=90)                        # 0% of 90 words
    assert (small + big).wer == pytest.approx(0.01)  # 1 error in 100 words, not the 5% average


def test_term_recall_counts_occurrences():
    ref = "cholecystectomy was done, then the second cholecystectomy and sutures"
    hyp = "colic is tech to me was done then the second cholecystectomy and sutures"
    t = term_recall(ref, hyp, ["cholecystectomy", "sutures", "hernia"])
    assert (t.found, t.total) == (2, 3)                    # hernia is not in the reference: ignored
    assert t.missed == {"cholecystectomy": 1}
    assert t.recall == pytest.approx(2 / 3)


def test_term_recall_phrases_and_no_partial_word_matches():
    t = term_recall("a bile duct injury", "a bile duct injury", ["bile duct"])
    assert t.recall == 1.0
    assert term_recall("sutures", "suture", ["sutures"]).found == 0   # exact word required
    assert term_recall("nothing relevant", "x", ["hernia"]).recall is None


def test_term_result_addition_merges_missed():
    a = TermResult(1, 2, {"x": 1})
    b = TermResult(0, 1, {"x": 1, "y": 1})
    c = a + b
    assert (c.found, c.total, c.missed) == (1, 3, {"x": 2, "y": 1})


def test_der_result_arithmetic():
    a = DerResult(total=10, missed=1, false_alarm=0.5, confusion=0.5)
    assert a.der == pytest.approx(0.2)
    b = a + DerResult(total=90)
    assert b.der == pytest.approx(0.02)
    assert DerResult().der is None


def test_speaker_count():
    assert speaker_count([{"speaker": "A"}, {"speaker": "B"}, {"speaker": "A"}]) == 2


def _pyannote_metrics_installed() -> bool:
    try:
        import pyannote.metrics  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(_pyannote_metrics_installed(), reason="only meaningful when pyannote.metrics is absent")
def test_der_without_pyannote_gives_a_clear_error():
    with pytest.raises(MetricsUnavailable, match="pyannote.metrics"):
        diarization_error([{"speaker": "A", "start": 0, "end": 1}], [{"speaker": "A", "start": 0, "end": 1}])


@pytest.mark.skipif(not _pyannote_metrics_installed(), reason="needs pyannote.metrics (installed on the GPU server)")
class TestDerWithPyannote:
    ref = [{"speaker": "A", "start": 0, "end": 10}, {"speaker": "B", "start": 10, "end": 20}]

    def test_label_names_do_not_matter(self):
        hyp = [{"speaker": "Speaker 2", "start": 0, "end": 10}, {"speaker": "Speaker 1", "start": 10, "end": 20}]
        assert diarization_error(self.ref, hyp, collar=0.0).der == pytest.approx(0.0)

    def test_boundary_error_is_measured(self):
        hyp = [{"speaker": "A", "start": 0, "end": 15}, {"speaker": "B", "start": 15, "end": 20}]
        r = diarization_error(self.ref, hyp, collar=0.0)
        assert r.total == pytest.approx(20) and r.confusion == pytest.approx(5)
        assert r.der == pytest.approx(0.25)

    def test_missed_speech_and_collar(self):
        hyp = [{"speaker": "A", "start": 0, "end": 10}]
        assert diarization_error(self.ref, hyp, collar=0.0).missed == pytest.approx(10)
        assert diarization_error(self.ref, self.ref, collar=0.25).der == pytest.approx(0.0)


# ====================================================================== dataset

def write_ref(path: Path, segments):
    path.write_text(json.dumps({"segments": segments}), encoding="utf-8")


SEGS = [
    {"speaker": "B", "start": 5.0, "end": 8.0, "text": "Hi, come in."},
    {"speaker": "A", "start": 0.0, "end": 4.0, "text": "Hello doctor."},
]


def test_load_reference_json_sorts_by_time_and_joins_text(tmp_path):
    p = tmp_path / "v.ref.json"
    write_ref(p, SEGS)
    ref = load_reference_json(p)
    assert ref.text == "Hello doctor. Hi, come in."
    assert [s["speaker"] for s in ref.segments] == ["A", "B"] and ref.has_diarization


def test_pipeline_output_can_be_used_as_a_reference(tmp_path):
    p = tmp_path / "v.ref.json"
    out = {"language": "en", "segments": [{**SEGS[1], "confidence": 0.9, "words": [], "text_redacted": "x"}],
           "full_text": "..."}
    p.write_text(json.dumps(out), encoding="utf-8")
    assert load_reference_json(p).text == "Hello doctor."


@pytest.mark.parametrize(
    "segments, message",
    [
        ([], "non-empty"),
        ([{"speaker": "A", "start": 0, "end": 1}], "'text'"),
        ([{"text": "x", "start": 0, "end": 1}], "missing speaker"),
        ([{"text": "x", "speaker": "A", "start": "a", "end": 1}], "numbers"),
        ([{"text": "x", "speaker": "A", "start": 5, "end": 1}], "invalid times"),
    ],
)
def test_bad_reference_json_is_explained(tmp_path, segments, message):
    p = tmp_path / "bad.ref.json"
    write_ref(p, segments)
    with pytest.raises(DatasetError, match=message):
        load_reference_json(p)


def test_invalid_json_and_missing_folder(tmp_path):
    p = tmp_path / "x.ref.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="cannot read as JSON"):
        load_reference_json(p)
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope")


def test_load_dataset_pairs_files_and_reports_unreferenced_audio(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    write_ref(tmp_path / "a.ref.json", SEGS)
    (tmp_path / "b.mp3").write_bytes(b"x")
    (tmp_path / "b.ref.txt").write_text("plain text reference", encoding="utf-8")
    (tmp_path / "c.wav").write_bytes(b"x")                      # no reference
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    recs, skipped = load_dataset(tmp_path)
    assert [r.name for r in recs] == ["a", "b"] and skipped == ["c.wav"]
    assert recs[0].reference.has_diarization and not recs[1].reference.has_diarization


def test_dataset_without_any_reference_is_an_error(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    with pytest.raises(DatasetError, match="No recordings with a reference"):
        load_dataset(tmp_path)


def test_empty_text_reference_is_rejected(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "a.ref.txt").write_text("  \n", encoding="utf-8")
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(tmp_path)


def test_load_terms_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("# heading\nhernia\n\n  bile duct  \n#x\n", encoding="utf-8")
    assert load_terms(p) == ["hernia", "bile duct"]


# ====================================================================== variants

def test_parse_variants_defaults_and_options():
    vs = parse_variants([{"name": "a"}, {"name": "b", "denoise": False, "min_speakers": 2, "max_speakers": 3,
                                          "whisper_model": "small", "initial_prompt": "terms"}])
    assert vs[0] == Variant("a") and vs[0].denoise is True
    assert vs[1].denoise is False and vs[1].whisper_model == "small" and vs[1].min_speakers == 2


@pytest.mark.parametrize(
    "data, message",
    [
        ([], "non-empty"),
        ({"name": "a"}, "non-empty JSON list"),
        ([{}], "'name' is required"),
        ([{"name": "a"}, {"name": "a"}], "Duplicate"),
        ([{"name": "a", "denoize": False}], "unknown option"),
        ([{"name": "a", "denoise": "no"}], "true or false"),
        ([{"name": "a", "min_speakers": 0}], "1 to 10"),
        ([{"name": "a", "min_speakers": 4, "max_speakers": 2}], "cannot exceed"),
    ],
)
def test_bad_variants_are_explained(data, message):
    with pytest.raises(VariantError, match=message):
        parse_variants(data)


def test_example_variants_file_in_the_repo_is_valid():
    path = Path(__file__).resolve().parents[2] / "evaluation" / "variants.example.json"
    names = [v.name for v in load_variants(path)]
    assert names[0] == "baseline" and "no-denoise" in names and len(names) == len(set(names))


def test_default_variants_compare_denoise_on_and_off():
    assert [(v.name, v.denoise) for v in DEFAULT_VARIANTS] == [("baseline", True), ("no-denoise", False)]


# ====================================================================== runner (fake stages)

REF_TEXT = "Hello doctor. Hi, come"   # the fake pipeline says: "Hello doctor." / "Hi, um come"
REF_SEGS = [{"speaker": "A", "start": 0.1, "end": 1.0, "text": "Hello doctor."},
            {"speaker": "B", "start": 1.2, "end": 1.9, "text": "Hi, come"}]


def recording(name="rec1", text=REF_TEXT, segments=REF_SEGS):
    return Recording(name, Path(f"{name}.wav"), Reference(text=text, segments=segments))


def fake_der(ref, hyp, collar):
    return DerResult(total=2.0, missed=0.0, false_alarm=0.0, confusion=0.5)


def make_factory(seen_cfgs, calls_by_cfg=None):
    def factory(cfg):
        stages, calls = fake_stages()
        seen_cfgs.append(cfg)
        if calls_by_cfg is not None:
            calls_by_cfg.append(calls)
        return stages
    return factory


def run_eval(tmp_path, recs, variants, **kw):
    seen = []
    runs = evaluate(recs, variants, make_factory(seen), kw.pop("cfg", PipelineConfig()), tmp_path / "tmp",
                    der_fn=kw.pop("der_fn", fake_der), **kw)
    return runs, seen


def test_evaluate_scores_each_variant_and_recording(tmp_path):
    runs, _ = run_eval(tmp_path, [recording("a"), recording("b")], DEFAULT_VARIANTS)
    assert [(r["variant"], r["recording"]) for r in runs] == [
        ("baseline", "a"), ("baseline", "b"), ("no-denoise", "a"), ("no-denoise", "b")]
    r = runs[0]
    assert r["ok"] and r["wer"] == {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 4}  # fillers ignored
    assert r["audio_sec"] == 2.0 and r["rtf"] is not None
    assert r["der"]["confusion"] == 0.5 and r["speakers"] == {"reference": 2, "hypothesis": 2}


def test_variant_settings_reach_the_pipeline(tmp_path):
    calls_list = []
    evaluate([recording()], [Variant("with"), Variant("without", denoise=False)], make_factory([], calls_list),
             PipelineConfig(), tmp_path / "tmp", der_fn=fake_der)
    calls = calls_list[0]                      # both variants share one model, so one set of stages
    assert calls.count("denoise") == 1         # only the denoising variant ran that stage
    assert calls.count("convert") == 2


def test_models_load_once_per_size_and_local_model_dir_only_applies_to_default(tmp_path):
    variants = [Variant("a"), Variant("b", whisper_model="small"), Variant("c"), Variant("d", whisper_model="small")]
    cfg = PipelineConfig(whisper_model="large-v3", whisper_model_dir="/models/large-v3")
    runs, seen = run_eval(tmp_path, [recording()], variants, cfg=cfg)
    assert [c.whisper_model for c in seen] == ["large-v3", "small"]        # two loads, not four
    assert seen[0].whisper_model_dir == "/models/large-v3" and seen[1].whisper_model_dir is None
    assert [r["variant"] for r in runs] == ["a", "c", "b", "d"]


def test_redaction_is_switched_off_during_evaluation(tmp_path):
    captured = []

    def factory(cfg):
        stages, _ = fake_stages()
        stages.redact = lambda t: "SHOULD NOT RUN"
        captured.append(stages)
        return stages

    evaluate([recording()], [Variant("a")], factory, PipelineConfig(), tmp_path / "tmp", der_fn=fake_der)
    assert captured[0].redact is None


def test_failed_run_is_recorded_and_others_continue(tmp_path):
    def factory(cfg):
        stages, _ = fake_stages()
        calls = {"n": 0}
        real = stages.transcribe

        def flaky(wav, prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("CUDA out of memory")
            return real(wav, prompt)

        stages.transcribe = flaky
        return stages

    runs = evaluate([recording("a"), recording("b")], [Variant("v")], factory, PipelineConfig(), tmp_path / "tmp",
                    der_fn=fake_der)
    assert [r["ok"] for r in runs] == [False, True]
    assert "CUDA out of memory" in runs[0]["error"]


def test_missing_pyannote_metrics_is_recorded_not_fatal(tmp_path):
    def no_der(ref, hyp, collar):
        raise MetricsUnavailable("pyannote.metrics is not installed")

    runs, _ = run_eval(tmp_path, [recording()], [Variant("v")], der_fn=no_der)
    assert runs[0]["ok"] and "der" not in runs[0] and "pyannote.metrics" in runs[0]["der_error"]


def test_text_only_reference_skips_der(tmp_path):
    runs, _ = run_eval(tmp_path, [recording(segments=None)], [Variant("v")])
    assert "der" not in runs[0] and "speakers" not in runs[0] and "der_error" not in runs[0]


def test_term_recall_is_scored_when_terms_given(tmp_path):
    runs, _ = run_eval(tmp_path, [recording()], [Variant("v")], terms=["doctor", "hernia"])
    assert runs[0]["terms"] == {"found": 1, "total": 1, "missed": {}}


def test_the_initial_prompt_changes_what_whisper_hears(tmp_path):
    def factory(cfg):
        stages, _ = fake_stages()
        words = lambda text: [{"word": w, "start": 0.1 + i * 0.1, "end": 0.15 + i * 0.1, "prob": 0.9}
                              for i, w in enumerate(text.split())]
        stages.transcribe = lambda wav, prompt: words("cholecystectomy done" if prompt else "colic is tech to me done")
        stages.diarize = lambda wav, lo, hi: [{"start": 0, "end": 2, "speaker": "S"}]
        return stages

    rec = recording(text="cholecystectomy done", segments=None)
    runs = evaluate([rec], [Variant("plain"), Variant("prompted", initial_prompt="surgery terms")], factory,
                    PipelineConfig(), tmp_path / "tmp", terms=["cholecystectomy"], der_fn=fake_der)
    plain, prompted = runs
    assert plain["terms"]["found"] == 0 and prompted["terms"]["found"] == 1
    assert prompted["wer"]["hits"] == 2 and plain["wer"]["hits"] == 1


def test_transcripts_are_only_saved_when_asked(tmp_path):
    run_eval(tmp_path, [recording()], [Variant("v")])
    assert not (tmp_path / "saved").exists()
    run_eval(tmp_path, [recording()], [Variant("v")], save_dir=tmp_path / "saved")
    saved = list((tmp_path / "saved").glob("*.json"))
    assert [p.name for p in saved] == ["v__rec1.json"]
    assert "Hello doctor." in saved[0].read_text(encoding="utf-8")


# ====================================================================== report

def wer_run(variant, rec, errors, words, **extra):
    return {"variant": variant, "recording": rec, "ok": True, "audio_sec": 60.0, "elapsed_sec": 30.0, "rtf": 0.5,
            "wer": {"substitutions": errors, "deletions": 0, "insertions": 0, "hits": words - errors}, **extra}


def sample_runs():
    return [
        wer_run("base", "a", 10, 100, der={"total": 60, "missed": 3, "false_alarm": 0, "confusion": 3},
                speakers={"reference": 2, "hypothesis": 2}, terms={"found": 1, "total": 2, "missed": {"hernia": 1}}),
        wer_run("base", "b", 0, 900, der={"total": 540, "missed": 0, "false_alarm": 0, "confusion": 0},
                speakers={"reference": 2, "hypothesis": 3}, terms={"found": 2, "total": 2, "missed": {}}),
        wer_run("alt", "a", 5, 100, der={"total": 60, "missed": 0, "false_alarm": 0, "confusion": 6},
                speakers={"reference": 2, "hypothesis": 2}, terms={"found": 2, "total": 2, "missed": {}}),
        wer_run("alt", "b", 9, 900, der={"total": 540, "missed": 0, "false_alarm": 0, "confusion": 0},
                speakers={"reference": 2, "hypothesis": 2}, terms={"found": 2, "total": 2, "missed": {}}),
    ]


META = {"recordings": 2, "audio_sec": 120.0, "skipped": [], "device": "cuda", "collar": 0.25, "strip_fillers": True}


def test_aggregate_uses_totals_across_recordings():
    agg = report.aggregate(sample_runs(), ["base", "alt"])
    assert agg["base"]["wer"] == pytest.approx(10 / 1000)          # not the 5% average of 10% and 0%
    assert agg["alt"]["wer"] == pytest.approx(14 / 1000)
    assert agg["base"]["der"] == pytest.approx(6 / 600)
    assert agg["base"]["term_recall"] == pytest.approx(3 / 4)
    assert agg["base"]["speaker_count_accuracy"] == pytest.approx(0.5)
    assert agg["alt"]["speaker_count_accuracy"] == 1.0
    assert agg["base"]["rtf_median"] == 0.5 and agg["base"]["audio_sec"] == 120.0


def test_paired_comparison_counts_wins_per_recording():
    c = report.paired_comparison(sample_runs(), "base", "alt", report._wer_metric)
    assert c["n"] == 2 and c["better"] == 1 and c["worse"] == 1 and c["same"] == 0
    assert c["mean_delta"] == pytest.approx(((0.05 - 0.10) + (0.01 - 0.0)) / 2)


def test_paired_comparison_ignores_failed_and_unmatched_runs():
    runs = sample_runs()
    runs[2] = {"variant": "alt", "recording": "a", "ok": False, "error": "boom"}
    assert report.paired_comparison(runs, "base", "alt", report._wer_metric)["n"] == 1
    assert report.paired_comparison([], "base", "alt", report._wer_metric) is None


def test_markdown_report_contents():
    results = report.build_results(sample_runs(), ["base", "alt"], META)
    md = report.render_markdown(results)
    assert "base (baseline)" in md and "alt" in md
    assert "1.0%" in md and "1.4%" in md                  # WER of each variant
    assert "lowest WER" not in md.split("| base (baseline)")[1].split("\n")[0]   # the baseline row is not the best
    assert "Small sample" in md                           # 2 recordings, 2 minutes
    assert "Paired comparison" in md and "WER by recording" in md
    assert "hernia (1x)" in md                            # most-missed terms
    assert "No-denoise" not in md                         # nothing invented about variants that were not run


def test_summary_change_matches_the_pooled_wer_column():
    # alt is worse when pooled (1.4% vs 1.0%) even though it wins on the short recording: the summary must not
    # contradict its own WER column.
    md = report.render_markdown(report.build_results(sample_runs(), ["base", "alt"], META))
    row = next(line for line in md.splitlines() if line.startswith("| alt "))
    assert "+0.4 pts" in row and "-2.0" not in row


def test_lowest_wer_variant_is_marked():
    runs = [wer_run("base", "a", 10, 100), wer_run("alt", "a", 2, 100)]
    md = report.render_markdown(report.build_results(runs, ["base", "alt"], META))
    assert "alt (lowest WER)" in md and "-8.0 pts" in md


def test_report_flags_the_lowest_wer_and_notes_problems():
    runs = sample_runs()
    runs[3] = {"variant": "alt", "recording": "b", "ok": False, "error": "boom"}
    meta = {**META, "skipped": ["c.wav"]}
    md = report.render_markdown(report.build_results(runs, ["base", "alt"], meta))
    assert "Skipped (no reference file): c.wav" in md
    assert "Variant 'alt': 1 run(s) failed" in md
    assert "failed" in md.split("## WER by recording")[1]


def test_small_sample_warning_disappears_with_enough_data():
    runs = [wer_run("base", f"r{i}", 1, 100) for i in range(6)]
    meta = {**META, "recordings": 6, "audio_sec": 1200.0}
    md = report.render_markdown(report.build_results(runs, ["base"], meta))
    assert "Small sample" not in md


def test_csv_has_one_row_per_run():
    runs = sample_runs() + [{"variant": "alt", "recording": "c", "ok": False, "error": "boom"}]
    rows = report.render_csv(runs).strip().splitlines()
    assert rows[0].startswith("variant,recording,ok,error") and len(rows) == 6
    assert "boom" in rows[-1]


# ====================================================================== CLI

def write_dataset(folder: Path):
    folder.mkdir()
    (folder / "rec1.wav").write_bytes(b"x")
    write_ref(folder / "rec1.ref.json", REF_SEGS)
    (folder / "terms.txt").write_text("doctor\n", encoding="utf-8")


@pytest.fixture
def cli_settings(monkeypatch, settings):
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    return settings


def test_cli_writes_all_outputs(tmp_path, cli_settings):
    data, out = tmp_path / "data", tmp_path / "out"
    write_dataset(data)
    code = eval_main(["--data", str(data), "--out", str(out), "--terms", str(data / "terms.txt")],
                     stages_factory=make_factory([]), der_fn=fake_der)
    assert code == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["baseline"] == "baseline" and set(results["variants"]) == {"baseline", "no-denoise"}
    assert results["variants"]["baseline"]["wer"] == 0.0 and results["variants"]["baseline"]["term_recall"] == 1.0
    assert "Evaluation report" in (out / "report.md").read_text(encoding="utf-8")
    assert (out / "runs.csv").read_text(encoding="utf-8").count("\n") >= 3
    assert not (out / "transcripts").exists()                    # no patient speech unless requested


def test_cli_reports_bad_input_with_exit_code_2(tmp_path, cli_settings, capsys):
    assert eval_main(["--data", str(tmp_path / "missing")], stages_factory=make_factory([])) == 2
    assert "not found" in capsys.readouterr().err
    data = tmp_path / "data"
    write_dataset(data)
    bad = tmp_path / "v.json"
    bad.write_text('[{"name": "a", "bogus": 1}]', encoding="utf-8")
    assert eval_main(["--data", str(data), "--variants", str(bad)], stages_factory=make_factory([])) == 2


def test_cli_exit_code_1_when_a_run_failed_but_report_is_still_written(tmp_path, cli_settings):
    data, out = tmp_path / "data", tmp_path / "out"
    write_dataset(data)

    def factory(cfg):
        stages, _ = fake_stages(fail_in="transcribe")
        return stages

    assert eval_main(["--data", str(data), "--out", str(out)], stages_factory=factory, der_fn=fake_der) == 1
    assert "failed" in (out / "report.md").read_text(encoding="utf-8")


# ====================================================================== production prompt hook

def test_initial_prompt_loading(tmp_path):
    from worker.config import load_initial_prompt

    assert load_initial_prompt({}) is None
    assert load_initial_prompt({"INITIAL_PROMPT": "  hernia, sutures  "}) == "hernia, sutures"
    assert load_initial_prompt({"INITIAL_PROMPT": "   "}) is None
    f = tmp_path / "prompt.txt"
    f.write_text("from file\n", encoding="utf-8")
    assert load_initial_prompt({"INITIAL_PROMPT_FILE": str(f), "INITIAL_PROMPT": "ignored"}) == "from file"
    f.write_text("\n", encoding="utf-8")
    assert load_initial_prompt({"INITIAL_PROMPT_FILE": str(f)}) is None


def test_worker_passes_the_configured_prompt_to_the_pipeline(monkeypatch, settings):
    import worker.config as wconfig
    import worker.pipeline.run as run_mod
    from worker.run_worker import make_default_runner

    seen = {}

    def fake_process(audio, options, stages, cfg, work_root, on_stage=None):
        seen["prompt"] = options.initial_prompt
        return {}

    monkeypatch.setattr(run_mod, "process_audio", fake_process)
    monkeypatch.setattr(wconfig, "INITIAL_PROMPT", "surgery terms")
    runner = make_default_runner(settings)
    runner(Path("x.wav"), {}, lambda stage: None)
    assert seen["prompt"] == "surgery terms"
