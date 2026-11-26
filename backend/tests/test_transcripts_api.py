import json

from tests.conftest import make_wav, ok_runner
from tests.test_api_gate import actions, tmp_files, upload
from worker.run_worker import run_once


def completed(client, auth, db, settings, who="uploader"):
    rid = upload(client, auth(who)).json()["id"]
    run_once(db, settings, ok_runner)
    return rid


# ---------- transcript views ----------

def test_transcript_requires_auth(client, auth, db, settings):
    rid = completed(client, auth, db, settings)
    assert client.get(f"/recordings/{rid}/transcript").status_code == 401


def test_redacted_view_never_contains_original_text(client, auth, db, settings):
    rid = completed(client, auth, db, settings)
    for role in ("viewer", "uploader", "admin"):
        r = client.get(f"/recordings/{rid}/transcript", headers=auth(role))
        assert r.status_code == 200
        body = json.dumps(r.json())
        assert "John" not in body and "Smith" not in body, role  # original text and words are withheld
        assert r.json()["redaction"] == "pending"  # PHI redaction is Phase 5
        assert r.json()["segments"][0]["text_redacted"] is None


def test_original_text_is_admin_only_and_audited(client, auth, db, settings):
    rid = completed(client, auth, db, settings)
    for role in ("viewer", "uploader"):
        assert client.get(f"/recordings/{rid}/transcript?original=true", headers=auth(role)).status_code == 403
    assert "access_denied" in actions(db, rid)
    r = client.get(f"/recordings/{rid}/transcript?original=true", headers=auth("admin"))
    assert r.status_code == 200
    seg = r.json()["segments"][0]
    assert seg["text"] == "My name is John Smith." and seg["words"][0]["word"] == "My"
    assert "_id" not in r.json()
    assert "view_original_text" in actions(db, rid)


def test_transcript_not_ready_or_missing(client, auth, db):
    h = auth("admin")
    rid = upload(client, h).json()["id"]  # queued, not processed
    assert client.get(f"/recordings/{rid}/transcript", headers=h).status_code == 409
    assert client.get("/recordings/rec_nope/transcript", headers=h).status_code == 404


def test_view_is_audited(client, auth, db, settings):
    rid = completed(client, auth, db, settings)
    client.get(f"/recordings/{rid}/transcript", headers=auth("viewer"))
    assert "view_transcript" in actions(db, rid)


# ---------- export ----------

def test_export_json_redacted_and_original(client, auth, db, settings):
    rid = completed(client, auth, db, settings)
    r = client.get(f"/recordings/{rid}/export.json", headers=auth("viewer"))
    assert r.status_code == 200
    assert f"{rid}.json" in r.headers["content-disposition"]
    assert "John" not in r.text
    assert client.get(f"/recordings/{rid}/export.json?original=true", headers=auth("viewer")).status_code == 403
    r = client.get(f"/recordings/{rid}/export.json?original=true", headers=auth("admin"))
    assert json.loads(r.text)["segments"][0]["text"] == "My name is John Smith."
    assert actions(db, rid).count("export_json") == 2


# ---------- audio ----------

def test_audio_admin_only_returns_original_bytes_and_wipes_temp(client, auth, db, settings):
    wav = make_wav(1.0)
    rid = client.post("/recordings", headers=auth("uploader"),
                      files={"file": ("visit.wav", wav, "audio/wav")}).json()["id"]
    for role in ("viewer", "uploader"):
        assert client.get(f"/recordings/{rid}/audio", headers=auth(role)).status_code == 403
    r = client.get(f"/recordings/{rid}/audio", headers=auth("admin"))
    assert r.status_code == 200 and r.content == wav and r.headers["content-type"] == "audio/wav"
    assert tmp_files(settings) == []
    assert "download_audio" in actions(db, rid)


def test_audio_supports_range_requests(client, auth, settings):
    wav = make_wav(1.0)
    rid = client.post("/recordings", headers=auth("admin"),
                      files={"file": ("visit.wav", wav, "audio/wav")}).json()["id"]
    r = client.get(f"/recordings/{rid}/audio", headers={**auth("admin"), "Range": "bytes=0-99"})
    assert r.status_code == 206 and r.content == wav[:100]
    assert tmp_files(settings) == []


def test_audio_tampered_file_is_refused_and_leaves_no_temp(client, auth, db, settings):
    rid = upload(client, auth("admin")).json()["id"]
    enc = settings.audio_dir / f"{rid}.enc"
    data = bytearray(enc.read_bytes())
    data[30] ^= 1
    enc.write_bytes(bytes(data))
    assert client.get(f"/recordings/{rid}/audio", headers=auth("admin")).status_code == 500
    assert tmp_files(settings) == []


def test_audio_of_gate_failed_recording_is_404(client, auth, db):
    upload(client, auth("admin"), data=b"not audio" * 30)
    rid = db.recordings.find_one()["_id"]
    assert client.get(f"/recordings/{rid}/audio", headers=auth("admin")).status_code == 404


# ---------- retry ----------

def test_retry_flow(client, auth, db, settings):
    def boom(audio, options, on_stage):
        raise RuntimeError("gpu out of memory")

    rid = upload(client, auth("uploader")).json()["id"]
    run_once(db, settings, boom)
    assert db.recordings.find_one({"_id": rid})["status"] == "failed"

    assert client.post(f"/recordings/{rid}/retry", headers=auth("uploader")).status_code == 403
    r = client.post(f"/recordings/{rid}/retry", headers=auth("admin"))
    assert r.status_code == 200 and r.json()["status"] == "queued" and r.json()["attempts"] == 1
    assert r.json()["error"] is None and "data_key" not in r.json()
    assert "retry" in actions(db, rid)

    assert client.post(f"/recordings/{rid}/retry", headers=auth("admin")).status_code == 409  # already queued
    run_once(db, settings, ok_runner)
    assert db.recordings.find_one({"_id": rid})["status"] == "completed"
    assert db.transcripts.count_documents({"recording_id": rid}) == 1


def test_retry_rejected_for_completed_and_gate_failed(client, auth, db, settings):
    h = auth("admin")
    rid = completed(client, auth, db, settings)
    assert client.post(f"/recordings/{rid}/retry", headers=h).status_code == 409
    upload(client, h, data=b"not audio" * 30)
    bad = db.recordings.find_one({"status": "failed"})["_id"]
    assert client.post(f"/recordings/{bad}/retry", headers=h).status_code == 409
    assert client.post("/recordings/rec_nope/retry", headers=h).status_code == 404


def test_retry_overwrites_previous_transcript_not_duplicates(client, auth, db, settings):
    rid = completed(client, auth, db, settings)
    db.recordings.update_one({"_id": rid}, {"$set": {"status": "failed"}})
    client.post(f"/recordings/{rid}/retry", headers=auth("admin"))
    run_once(db, settings, ok_runner)
    assert db.transcripts.count_documents({"recording_id": rid}) == 1


# ---------- upload options ----------

def test_upload_options_stored_and_validated(client, auth, db):
    h = auth("uploader")
    wav = make_wav(1.0)

    def post(**data):
        return client.post("/recordings", headers=h, data=data, files={"file": ("a.wav", wav, "audio/wav")})

    r = post(min_speakers="2", max_speakers="3", denoise="false")
    assert r.status_code == 201
    assert r.json()["options"] == {"denoise": False, "filler_removal": False, "min_speakers": 2, "max_speakers": 3}
    assert post(min_speakers="4", max_speakers="2").status_code == 400
    assert post(min_speakers="0").status_code == 422
    assert post(max_speakers="99").status_code == 422
    assert db.recordings.count_documents({}) == 1  # invalid options never reach the gate
