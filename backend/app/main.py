from __future__ import annotations

from fastapi import FastAPI

from app.auth.router import router as auth_router
from app.recordings.router import router as recordings_router

app = FastAPI(title="Surgeon-Patient Transcription", version="0.1.0")
app.include_router(auth_router)
app.include_router(recordings_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
