"""DEVELOPMENT ONLY: run the whole app in memory, with no MongoDB, no ML models and no GPU.

    python scripts/dev_server.py [--port N]  # then open http://127.0.0.1:8000  (after: cd frontend && npm run build)
    # or, with hot reload:  cd frontend && npm run dev   (proxies the API to this server)

Logins (all use the password below):  admin / uploader / viewer
Uploads go through the real gate (validation + encryption). A fake pipeline then "transcribes" them with
a canned two-speaker conversation, one stage at a time, so progress states are visible in the UI.

Safety: binds to loopback only, uses throw-away keys, and stores everything under storage/dev (git-ignored).
Never use this for real data or on a server: the accounts are public and the transcript is fake.
"""
from __future__ import annotations

import math
import re
import secrets
import shutil
import struct
import sys
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

import mongomock  # noqa: E402
import uvicorn  # noqa: E402

from app.auth.security import hash_password  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.db.mongo import ensure_indexes, get_db, get_db_or_none  # noqa: E402
from app.gate.service import run_gate  # noqa: E402
from app.main import app  # noqa: E402
from worker.pipeline.redact import Entity, Redactor, rule_entities  # noqa: E402
from worker.run_worker import run_once  # noqa: E402

PASSWORD = "devpassword123"
HOST = "127.0.0.1"

SCRIPT = [
    (1, "Good morning, I am Dr. Patel. Thanks for coming in today. How have you been feeling since the procedure?"),
    (2, "Hello. My name is Maria Lopez and you can reach me on 555 014 2237 or maria.lopez@example.com."),
    (2, "Honestly, a bit sore around the incision, but the pain is much better than last week."),
    (1, "That's expected. Any fever, redness or discharge from the wound?"),
    (2, "No fever. There was a little redness at first, but it has gone down."),
    (1, "Good. Let me take a look. The healing looks right on track. We'll remove the stitches next week."),
    (2, "Okay. Is it alright to start walking more? I would like to get back to my normal routine."),
    (1, "Short walks are fine. Avoid lifting anything heavy for another two weeks."),
]


def dev_names(text: str) -> list[Entity]:
    """Stand-in for NER (Presidio + spaCy are not installed in development)."""
    return [Entity(m.start(), m.end(), "PERSON", 0.85) for m in re.finditer(r"Maria Lopez|Patel", text)]


REDACTOR = Redactor([rule_entities, dev_names])


def make_tone_wav(path: Path, seconds: float = 40.0, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / rate))) for i in range(int(seconds * rate))
        ))


def canned_transcript() -> dict:
    segments, t = [], 0.5
    for who, text in SCRIPT:
        words, start = [], t
        for tok in text.split():
            dur = 0.18 + 0.03 * len(tok)
            words.append({"word": tok, "start": round(t, 2), "end": round(t + dur, 2), "prob": 0.95})
            t += dur + 0.05
        segments.append({"speaker": f"Speaker {who}", "start": round(start, 2), "end": round(t, 2),
                         "text": text, "text_redacted": REDACTOR.redact(text), "confidence": 0.94,
                         "words": words})
        t += 0.8
    return {
        "language": "en", "duration_sec": round(t, 2), "speaker_count": 2, "segments": segments,
        "full_text": "\n".join(f"{s['speaker']}: {s['text']}" for s in segments),
        "full_text_redacted": "\n".join(f"{s['speaker']}: {s['text_redacted']}" for s in segments),
        "redaction": {"status": "ok", "engine": "dev rules"},
        "pipeline": {"whisper_model": "dev-fake", "diarization_model": "dev-fake", "device": "none",
                     "denoise": True, "timings_sec": {}},
    }


def fake_runner(audio: Path, options: dict, on_stage) -> dict:
    assert audio.stat().st_size > 0
    for stage in ("convert", "denoise", "vad", "transcribe", "diarize", "redact"):
        on_stage(stage)
        time.sleep(1.2)
    return canned_transcript()


def main() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    storage = ROOT / "storage" / "dev"
    shutil.rmtree(storage, ignore_errors=True)
    settings = Settings(_env_file=None, jwt_secret=secrets.token_urlsafe(32), storage_dir=storage,
                        enable_docs=True, max_upload_mb=200)
    db = mongomock.MongoClient(tz_aware=True)["dev"]
    ensure_indexes(db)
    for role in ("admin", "uploader", "viewer"):
        db.users.insert_one({"_id": uuid.uuid4().hex, "username": role, "password_hash": hash_password(PASSWORD),
                             "role": role, "created_at": datetime.now(timezone.utc)})
    admin = db.users.find_one({"username": "admin"})["_id"]

    storage.mkdir(parents=True, exist_ok=True)
    demo = storage / "demo.wav"
    make_tone_wav(demo)
    with open(demo, "rb") as fh:
        run_gate(db, settings, stream=fh, filename="demo-visit.wav", user_id=admin, source="upload",
                 reference="demo recording", ip=None, options={})
    demo.unlink()

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_db_or_none] = lambda: db
    app.dependency_overrides[get_settings] = lambda: settings

    def worker_loop() -> None:
        while True:
            try:
                if not run_once(db, settings, fake_runner):
                    time.sleep(0.5)
            except Exception as exc:  # keep the dev worker alive
                print("worker error:", exc)

    threading.Thread(target=worker_loop, daemon=True).start()
    print(f"DEV SERVER on http://{HOST}:{port}  (admin|uploader|viewer / {PASSWORD})  - fake pipeline, in-memory data")
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
