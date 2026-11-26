from app.gate.crypto import encode_key, generate_data_key
from tests.conftest import fake_result, make_wav, ok_runner
from tests.test_api_gate import actions, tmp_files, upload
from worker.run_worker import recover_after_crash, run_once


def test_worker_processes_stores_transcript_and_cleans_up(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    assert run_once(db, settings, ok_runner) is True
    rec = db.recordings.find_one({"_id": rid})
    assert rec["status"] == "completed" and rec["completed_at"] and rec["stage"] is None
    t = db.transcripts.find_one({"recording_id": rid})
    assert t["segments"][0]["text"] == "My name is John Smith."
    assert t["segments"][0]["text_redacted"] is None and t["full_text_redacted"] is None
    assert actions(db, rid)[-2:] == ["process_start", "process_done"]
    assert tmp_files(settings) == []
    assert run_once(db, settings, ok_runner) is False  # queue drained


def test_runner_receives_recording_options(client, auth, db, settings):
    seen = {}

    def runner(audio, options, on_stage):
        seen.update(options)
        return fake_result()

    r = client.post(
        "/recordings", headers=auth("uploader"),
        data={"min_speakers": "2", "max_speakers": "4", "denoise": "false", "filler_removal": "true"},
        files={"file": ("a.wav", make_wav(1.0), "audio/wav")},
    )
    assert r.status_code == 201, r.text
    run_once(db, settings, runner)
    assert seen == {"denoise": False, "filler_removal": True, "min_speakers": 2, "max_speakers": 4}


def test_stage_progress_is_recorded(client, auth, db, settings):
    stages = []

    def runner(audio, options, on_stage):
        on_stage("transcribe")
        stages.append(db.recordings.find_one({})["stage"])
        return fake_result()

    upload(client, auth("uploader"))
    run_once(db, settings, runner)
    assert stages == ["transcribe"]


def test_pipeline_failure_marks_failed_stores_no_transcript_and_cleans_up(client, auth, db, settings):
    def boom(audio, options, on_stage):
        raise RuntimeError("model crashed")

    rid = upload(client, auth("uploader")).json()["id"]
    run_once(db, settings, boom)
    rec = db.recordings.find_one({"_id": rid})
    assert rec["status"] == "failed" and "model crashed" in rec["error"]
    assert db.transcripts.count_documents({}) == 0
    assert actions(db, rid)[-1] == "process_fail"
    assert tmp_files(settings) == []


def test_tampered_audio_fails_and_cleans_up(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    enc = settings.audio_dir / f"{rid}.enc"
    data = bytearray(enc.read_bytes())
    data[30] ^= 1
    enc.write_bytes(bytes(data))
    run_once(db, settings, ok_runner)
    rec = db.recordings.find_one({"_id": rid})
    assert rec["status"] == "failed" and "Authentication failed" in rec["error"]
    assert tmp_files(settings) == []


def test_wrong_data_key_fails_and_cleans_up(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    db.recordings.update_one({"_id": rid}, {"$set": {"data_key": encode_key(generate_data_key())}})
    run_once(db, settings, ok_runner)
    assert db.recordings.find_one({"_id": rid})["status"] == "failed"
    assert tmp_files(settings) == []


def test_hash_mismatch_fails_and_cleans_up(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    db.recordings.update_one({"_id": rid}, {"$set": {"sha256": "0" * 64}})
    run_once(db, settings, ok_runner)
    rec = db.recordings.find_one({"_id": rid})
    assert rec["status"] == "failed" and "Integrity" in rec["error"]
    assert tmp_files(settings) == []


def test_failed_gate_recordings_are_never_picked_up(client, auth, db, settings):
    upload(client, auth("uploader"), data=b"not audio" * 30)
    assert run_once(db, settings, ok_runner) is False


def test_recover_after_crash_requeues_and_wipes_stale_temp(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    db.recordings.update_one({"_id": rid}, {"$set": {"status": "processing"}})
    stale = settings.tmp_dir / "job_dead"
    stale.mkdir(parents=True)
    (stale / "audio.bin").write_bytes(b"leftover plaintext")
    assert recover_after_crash(db, settings) == 1
    assert db.recordings.find_one({"_id": rid})["status"] == "queued"
    assert not stale.exists()
    assert run_once(db, settings, ok_runner) is True  # and it can then be processed normally
