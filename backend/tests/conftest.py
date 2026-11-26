from __future__ import annotations

import io
import math
import struct
import uuid
import wave
from datetime import datetime, timezone

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.auth.security import hash_password
from app.config import Settings, get_settings
from app.db.mongo import ensure_indexes, get_db, get_db_or_none
from app.main import app


def make_wav(seconds: float = 1.0, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(int(seconds * rate))
        )
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        jwt_secret="test-secret-test-secret-test-secret",
        storage_dir=tmp_path / "storage",
        max_upload_mb=1,
        max_duration_sec=30,
        require_ffprobe=False,
    )


@pytest.fixture
def db():
    database = mongomock.MongoClient(tz_aware=True)["test"]
    ensure_indexes(database)
    return database


@pytest.fixture
def client(db, settings):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_db_or_none] = lambda: db
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def _add_user(db, username: str, role: str) -> None:
    db.users.insert_one(
        {"_id": uuid.uuid4().hex, "username": username, "password_hash": hash_password("password-123456"),
         "role": role, "created_at": datetime.now(timezone.utc)}
    )


@pytest.fixture
def auth(client, db):
    """Return a function: auth('admin') -> Authorization headers."""
    def _auth(role: str) -> dict:
        if not db.users.find_one({"username": role}):
            _add_user(db, role, role)
        r = client.post("/auth/login", json={"username": role, "password": "password-123456"})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    return _auth


def fake_result(text: str = "My name is John Smith.") -> dict:
    words = [{"word": t, "start": i * 0.5, "end": i * 0.5 + 0.4, "prob": 0.9} for i, t in enumerate(text.split())]
    seg = {"speaker": "Speaker 1", "start": 0.0, "end": 2.0, "text": text, "confidence": 0.9, "words": words}
    return {
        "language": "en", "duration_sec": 1.0, "speaker_count": 1, "segments": [seg],
        "full_text": f"Speaker 1: {text}",
        "pipeline": {"whisper_model": "fake", "denoise": True, "timings_sec": {}},
    }


def redacted_result() -> dict:
    """A transcript as the pipeline returns it when redaction succeeded."""
    result = fake_result("My name is John Smith.")
    result["segments"][0]["text_redacted"] = "My name is [PERSON]."
    result["full_text_redacted"] = "Speaker 1: My name is [PERSON]."
    result["redaction"] = {"status": "ok", "engine": "test"}
    return result


def failed_redaction_result() -> dict:
    result = fake_result("My name is John Smith.")
    result["segments"][0]["text_redacted"] = None
    result["full_text_redacted"] = None
    result["redaction"] = {"status": "failed", "engine": "test", "error": "OSError: model missing"}
    return result


def ok_runner(audio, options, on_stage):
    """Stand-in for the ML pipeline: reads the decrypted audio, reports stages, returns a transcript."""
    assert audio.exists() and audio.stat().st_size > 0
    for stage in ("convert", "transcribe"):
        on_stage(stage)
    return fake_result()
