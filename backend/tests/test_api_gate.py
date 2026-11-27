from tests.conftest import make_wav


def upload(client, headers, data=None, name="visit.wav"):
    data = make_wav(1.0) if data is None else data
    return client.post("/recordings", files={"file": (name, data, "audio/wav")}, headers=headers)


def actions(db, rec_id=None):
    q = {"recording_id": rec_id} if rec_id else {}
    return [a["action"] for a in db.audit_log.find(q).sort([("at", 1), ("_id", 1)])]


def tmp_files(settings):
    return [p for p in settings.tmp_dir.rglob("*") if p.is_file()] if settings.tmp_dir.exists() else []


def test_unauthenticated_rejected(client):
    assert upload(client, {}).status_code == 401
    assert client.get("/recordings").status_code == 401
    assert upload(client, {"Authorization": "Bearer not-a-token"}).status_code == 401


def test_bad_login_is_audited(client, db):
    r = client.post("/auth/login", json={"username": "ghost", "password": "x"})
    assert r.status_code == 401
    assert "login_failed" in actions(db)


def test_viewer_cannot_upload_but_can_list(client, auth, db):
    h = auth("viewer")
    assert upload(client, h).status_code == 403
    assert "access_denied" in actions(db)
    assert client.get("/recordings", headers=h).status_code == 200


def test_successful_upload_flow(client, auth, db, settings):
    r = upload(client, auth("uploader"))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["duration_sec"] and len(body["sha256"]) == 64
    assert "data_key" not in body and "audio_path" not in body
    assert actions(db, body["id"]) == ["upload", "gate_pass", "queued"]

    # Only the encrypted file exists permanently; nothing plaintext remains anywhere.
    enc_files = list(settings.audio_dir.glob("*.enc"))
    assert [f.name for f in enc_files] == [body["id"] + ".enc"]
    assert tmp_files(settings) == []
    assert b"RIFF" not in enc_files[0].read_bytes()[:64]


def test_fake_audio_rejected_and_nothing_stored(client, auth, db, settings):
    r = upload(client, auth("uploader"), data=b"MZ not audio at all" * 20)
    assert r.status_code == 400
    rec = db.recordings.find_one()
    assert rec["status"] == "failed" and rec["error"]
    assert "gate_fail" in actions(db, rec["_id"])
    assert not (settings.audio_dir.exists() and list(settings.audio_dir.glob("*")))
    assert tmp_files(settings) == []


def test_oversize_upload_rejected_413(client, auth, settings):
    r = upload(client, auth("admin"), data=make_wav(1.0) + b"\0" * (2 * 1024 * 1024))
    assert r.status_code == 413
    assert tmp_files(settings) == []


def test_extension_mismatch_rejected(client, auth):
    assert upload(client, auth("admin"), name="visit.mp3").status_code == 400


def test_path_traversal_filename_sanitised(client, auth):
    r = upload(client, auth("admin"), name="../../etc/evil.wav")
    assert r.status_code == 201
    assert r.json()["original_filename"] == "evil.wav"


def test_get_recording_hides_secrets_and_404(client, auth):
    h = auth("admin")
    rid = upload(client, h).json()["id"]
    got = client.get(f"/recordings/{rid}", headers=h).json()
    assert got["id"] == rid and "data_key" not in got
    assert client.get("/recordings/rec_missing", headers=h).status_code == 404
