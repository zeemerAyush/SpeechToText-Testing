from app.gate.crypto import encode_key, generate_data_key
from tests.test_api_gate import actions, tmp_files, upload
from worker.run_worker import run_once


def test_worker_processes_and_cleans_up(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    assert run_once(db, settings) is True
    rec = db.recordings.find_one({"_id": rid})
    assert rec["status"] == "completed" and rec["completed_at"]
    assert actions(db, rid)[-2:] == ["process_start", "process_done"]
    assert tmp_files(settings) == []
    assert run_once(db, settings) is False  # queue drained


def test_tampered_audio_fails_and_cleans_up(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    enc = settings.audio_dir / f"{rid}.enc"
    data = bytearray(enc.read_bytes())
    data[30] ^= 1
    enc.write_bytes(bytes(data))
    run_once(db, settings)
    rec = db.recordings.find_one({"_id": rid})
    assert rec["status"] == "failed" and "Authentication failed" in rec["error"]
    assert actions(db, rid)[-1] == "process_fail"
    assert tmp_files(settings) == []


def test_wrong_data_key_fails_and_cleans_up(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    db.recordings.update_one({"_id": rid}, {"$set": {"data_key": encode_key(generate_data_key())}})
    run_once(db, settings)
    assert db.recordings.find_one({"_id": rid})["status"] == "failed"
    assert tmp_files(settings) == []


def test_hash_mismatch_fails_and_cleans_up(client, auth, db, settings):
    rid = upload(client, auth("uploader")).json()["id"]
    db.recordings.update_one({"_id": rid}, {"$set": {"sha256": "0" * 64}})
    run_once(db, settings)
    rec = db.recordings.find_one({"_id": rid})
    assert rec["status"] == "failed" and "Integrity" in rec["error"]
    assert tmp_files(settings) == []


def test_failed_gate_recordings_are_never_picked_up(client, auth, db, settings):
    upload(client, auth("uploader"), data=b"not audio" * 30)
    assert run_once(db, settings) is False
