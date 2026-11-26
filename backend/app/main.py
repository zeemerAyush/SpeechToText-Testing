from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.auth.router import router as auth_router
from app.config import ROOT_DIR, get_settings
from app.db.mongo import get_db_or_none
from app.recordings.router import router as recordings_router

FRONTEND_DIR = ROOT_DIR / "frontend" / "dist"

_settings = get_settings()
app = FastAPI(
    title="Surgeon-Patient Transcription",
    version="0.1.0",
    docs_url="/docs" if _settings.enable_docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _settings.enable_docs else None,
)
app.include_router(auth_router)
app.include_router(recordings_router)

# The built UI is same-origin with the API, so no CORS is needed. These headers lock the page down.
CSP = (
    "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if _settings.enable_docs and request.url.path.startswith(("/docs", "/openapi")):
        return response  # Swagger UI needs inline scripts and a CDN
    response.headers.setdefault("Content-Security-Policy", CSP)
    if not request.url.path.startswith("/assets/"):
        response.headers.setdefault("Cache-Control", "no-store")  # transcripts and tokens: never cached
    return response


@app.get("/health")
def health(db: Database | None = Depends(get_db_or_none)) -> dict:
    """Unauthenticated liveness check, also used to confirm the server's device and model settings."""
    try:
        if db is None:
            raise PyMongoError("no connection")
        db.command("ping")
        mongo = "ok"
    except PyMongoError:
        mongo = "unavailable"
    device = os.environ.get("DEVICE", "cpu")
    return {
        "status": "ok" if mongo == "ok" else "degraded",
        "mongo": mongo,
        "device": device,  # configured device for the worker (cuda | cpu)
        "whisper_model": os.environ.get("WHISPER_MODEL", "small"),
        "ffprobe": shutil.which("ffprobe") is not None,
        "frontend": FRONTEND_DIR.is_dir(),
    }


def mount_frontend(application: FastAPI, directory: Path = FRONTEND_DIR) -> bool:
    """Serve the built React app at '/'. Mounted last so every API route keeps priority."""
    if not (directory / "index.html").is_file():
        return False
    application.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
    return True


mount_frontend(app)
