import json

from tests.conftest import failed_redaction_result, ok_runner, redacted_result
from tests.test_api_gate import actions, upload
from tests.test_pipeline_run import fake_stages
from worker.pipeline.redact import Redactor, rule_entities
from worker.pipeline.run import Options, PipelineConfig, process_audio
from worker.redact_backfill import backfill
from worker.run_worker import run_once

PHI = "My name is Maria and my number is 555-014-2237."


def phi_stages(**overrides):
    """Fake stages whose transcript contains a phone number."""
    stages, calls = fake_stages()
    words = [{"word": f" {t}", "start": 0.1 + i * 0.1, "end": 0.18 + i * 0.1, "prob": 0.9}
             for i, t in enumerate(PHI.split())]
    stages.transcribe = lambda wav, prompt: words
    stages.diarize = lambda wav, lo, hi: [{"start": 0, "end": 2, "speaker": "SPEAKER_00"}]
    for k, v in overrides.items():
        setattr(stages, k, v)
    return stages, calls


def run(tmp_path, stages, on_stage=None):
    return process_audio(tmp_path / "in.wav", Options(denoise=False), stages, PipelineConfig(), tmp_path / "tmp",
                         on_stage=on_stage)


# ---------- pipeline ----------

def test_pipeline_redacts_each_segment_and_builds_full_text(tmp_path):
    stages, _ = phi_stages(redact=Redactor([rule_entities]).redact, redact_engine="rules")
    stages_seen = []
    out = run(tmp_path, stages, stages_seen.append)
    seg = out["segments"][0]
    assert seg["text"] == PHI and "555" not in seg["text_redacted"]
    assert seg["text_redacted"].endswith("[PHONE_NUMBER].")
    assert out["redaction"] == {"status": "ok", "engine": "rules"}
    assert out["full_text_redacted"] == f"Speaker 1: {seg['text_redacted']}"
    assert "redact" in out["pipeline"]["timings_sec"]
    assert stages_seen[-1] == "redact"


def test_redaction_failure_keeps_transcript_but_leaks_no_redacted_text(tmp_path):
    def boom(text):
        raise OSError("spaCy model not found")

    stages, _ = phi_stages(redact=boom, redact_engine="presidio+rules")
    out = run(tmp_path, stages)  # must not raise
    assert out["segments"][0]["text"] == PHI  # admins still get the transcript
    assert out["segments"][0]["text_redacted"] is None
    assert out["full_text_redacted"] is None
    assert out["redaction"]["status"] == "failed" and "spaCy model not found" in out["redaction"]["error"]


def test_partial_redaction_failure_is_all_or_nothing(tmp_path):
    stages, _ = fake_stages()  # two segments
    calls = []

    def flaky(text):
        calls.append(text)
        if len(calls) == 2:
            raise RuntimeError("second segment failed")
        return text.upper()

    stages.redact = flaky
    out = run(tmp_path, stages)
    assert out["redaction"]["status"] == "failed"
    assert all(s["text_redacted"] is None for s in out["segments"])


def test_no_redactor_configured_means_not_run_and_no_redacted_text(tmp_path):
    stages, _ = phi_stages()  # redact left as None
    out = run(tmp_path, stages)
    assert out["redaction"]["status"] == "not_run"
    assert out["segments"][0]["text_redacted"] is None and out["full_text_redacted"] is None


# ---------- worker stores the redaction result ----------

def test_worker_stores_redacted_text_and_status(client, auth, db, settings):
    upload(client, auth("uploader"))
    run_once(db, settings, lambda a, o, s: redacted_result())
    t = db.transcripts.find_one({})
    assert t["segments"][0]["text_redacted"] == "My name is [PERSON]."
    assert t["full_text_redacted"] == "Speaker 1: My name is [PERSON]."
    assert t["redaction"] == {"status": "ok", "engine": "test"}


def test_worker_defaults_to_not_run_when_runner_reports_nothing(client, auth, db, settings):
    upload(client, auth("uploader"))
    run_once(db, settings, ok_runner)
    assert db.transcripts.find_one({})["redaction"]["status"] == "not_run"


# ---------- what each role sees ----------

def completed_with(client, auth, db, settings, result):
    rid = upload(client, auth("uploader")).json()["id"]
    run_once(db, settings, lambda a, o, s: result)
    return rid


def test_non_admins_get_redacted_text_and_never_the_original(client, auth, db, settings):
    rid = completed_with(client, auth, db, settings, redacted_result())
    for role in ("viewer", "uploader"):
        r = client.get(f"/recordings/{rid}/transcript", headers=auth(role))
        body = r.json()
        assert r.status_code == 200 and body["redaction"] == "available"
        assert body["segments"][0]["text_redacted"] == "My name is [PERSON]."
        assert body["full_text_redacted"] == "Speaker 1: My name is [PERSON]."
        assert "John" not in json.dumps(body) and "Smith" not in json.dumps(body)
        assert "words" not in body["segments"][0] and "text" not in body["segments"][0]


def test_export_for_non_admins_is_redacted(client, auth, db, settings):
    rid = completed_with(client, auth, db, settings, redacted_result())
    r = client.get(f"/recordings/{rid}/export.json", headers=auth("viewer"))
    assert "[PERSON]" in r.text and "John" not in r.text


def test_admin_can_read_both_versions(client, auth, db, settings):
    rid = completed_with(client, auth, db, settings, redacted_result())
    h = auth("admin")
    original = client.get(f"/recordings/{rid}/transcript?original=true", headers=h).json()
    assert original["segments"][0]["text"] == "My name is John Smith."
    assert original["segments"][0]["text_redacted"] == "My name is [PERSON]."
    assert original["redaction"]["status"] == "ok"
    redacted = client.get(f"/recordings/{rid}/transcript", headers=h).json()
    assert redacted["segments"][0]["text_redacted"] == "My name is [PERSON]."
    assert "John" not in json.dumps(redacted)


def test_failed_redaction_stays_hidden_from_non_admins(client, auth, db, settings):
    rid = completed_with(client, auth, db, settings, failed_redaction_result())
    body = client.get(f"/recordings/{rid}/transcript", headers=auth("viewer")).json()
    assert body["redaction"] == "failed"
    assert body["segments"][0]["text_redacted"] is None and body["full_text_redacted"] is None
    assert "John" not in json.dumps(body)
    assert "model missing" not in json.dumps(body)  # internal error details stay admin-only
    admin = client.get(f"/recordings/{rid}/transcript?original=true", headers=auth("admin")).json()
    assert admin["redaction"]["error"].startswith("OSError")


def test_not_run_is_reported_as_pending(client, auth, db, settings):
    result = {**redacted_result(), "redaction": {"status": "not_run", "engine": None}, "full_text_redacted": None}
    rid = completed_with(client, auth, db, settings, result)
    assert client.get(f"/recordings/{rid}/transcript", headers=auth("viewer")).json()["redaction"] == "pending"


# ---------- backfill ----------

def make_transcript(db, rec_id, redaction=None, text="Call 555-014-2237 now."):
    doc = {"recording_id": rec_id, "language": "en", "speaker_count": 1,
           "segments": [{"speaker": "Speaker 1", "start": 0, "end": 1, "text": text, "text_redacted": None,
                         "confidence": 0.9, "words": []}],
           "full_text": f"Speaker 1: {text}", "full_text_redacted": None, "pipeline": {}}
    if redaction:
        doc["redaction"] = redaction
    db.transcripts.insert_one(doc)


def test_backfill_redacts_missing_and_failed_but_skips_done(db):
    make_transcript(db, "rec_a")  # created before redaction existed: no redaction field
    make_transcript(db, "rec_b", {"status": "failed", "engine": "x", "error": "e"})
    make_transcript(db, "rec_c", {"status": "ok", "engine": "x"}, text="keep this exact text")
    result = backfill(db, Redactor([rule_entities]).redact, "rules")
    assert result == {"redacted": 2, "failed": 0}
    a = db.transcripts.find_one({"recording_id": "rec_a"})
    assert a["segments"][0]["text_redacted"] == "Call [PHONE_NUMBER] now."
    assert a["full_text_redacted"] == "Speaker 1: Call [PHONE_NUMBER] now."
    assert a["redaction"] == {"status": "ok", "engine": "rules"}
    assert db.transcripts.find_one({"recording_id": "rec_c"})["segments"][0]["text_redacted"] is None  # untouched
    assert actions(db).count("redact") == 2
    assert backfill(db, Redactor([rule_entities]).redact, "rules") == {"redacted": 0, "failed": 0}  # idempotent


def test_backfill_failure_is_recorded_and_leaks_nothing(db):
    make_transcript(db, "rec_a")

    def boom(text):
        raise RuntimeError("no model")

    assert backfill(db, boom, "x") == {"redacted": 0, "failed": 1}
    t = db.transcripts.find_one({})
    assert t["redaction"]["status"] == "failed" and t["segments"][0]["text_redacted"] is None
    assert t["full_text_redacted"] is None
