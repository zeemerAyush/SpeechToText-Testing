"""Recording endpoints. Uploads always pass through the guardrail gate."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pymongo.database import Database
from starlette.background import BackgroundTask

from app.auth.deps import CurrentUser, client_ip, require_permission
from app.auth.roles import can
from app.config import Settings, get_settings
from app.db.mongo import get_db
from app.gate import audit
from app.gate.crypto import DecryptionError
from app.gate.service import run_gate
from app.gate.storage import wipe_dir
from app.gate.validation import ValidationError
from app.gate.workspace import IntegrityError, decrypt_to_temp
from app.recordings.views import original_view, redacted_view

router = APIRouter(prefix="/recordings", tags=["recordings"])

# Fields that must never leave the server.
_HIDDEN = {"audio_path", "data_key"}

AUDIO_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".webm": "audio/webm",
}


def public(rec: dict) -> dict:
    return {("id" if k == "_id" else k): v for k, v in rec.items() if k not in _HIDDEN}


def _get_recording(db: Database, recording_id: str) -> dict:
    rec = db.recordings.find_one({"_id": recording_id})
    if rec is None:
        raise HTTPException(404, "Recording not found.")
    return rec


def _get_transcript(db: Database, rec: dict) -> dict:
    if rec["status"] != "completed":
        raise HTTPException(409, f"Transcript not ready (status: {rec['status']}).")
    transcript = db.transcripts.find_one({"recording_id": rec["_id"]})
    if transcript is None:
        raise HTTPException(404, "Transcript not found.")
    return transcript


def _transcript_for(
    db: Database, rec: dict, user: CurrentUser, original: bool, request: Request, action: str
) -> dict:
    """Return the transcript view this user may see, and audit the access."""
    if original and not can(user.role, "view_original_text"):
        audit.record(db, "access_denied", user_id=user.id, recording_id=rec["_id"],
                     detail="original text requested", ip=client_ip(request))
        raise HTTPException(403, "You do not have permission to view the original text.")
    transcript = _get_transcript(db, rec)
    logged, detail = action, ""
    if original:
        if action == "view_transcript":
            logged = "view_original_text"
        else:
            detail = "original text"
    audit.record(db, logged, user_id=user.id, recording_id=rec["_id"], detail=detail, ip=client_ip(request))
    return original_view(transcript) if original else redacted_view(transcript)


@router.post("", status_code=201)
def upload_recording(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("upload"),
    reference: str | None = Form(None),
    min_speakers: int | None = Form(None, ge=1, le=10),
    max_speakers: int | None = Form(None, ge=1, le=10),
    denoise: bool = Form(True),
    filler_removal: bool = Form(False),
    user: CurrentUser = Depends(require_permission("upload")),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    if source not in {"upload", "browser_recording"}:
        raise HTTPException(400, "source must be 'upload' or 'browser_recording'.")
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise HTTPException(400, "min_speakers cannot exceed max_speakers.")
    options = {"denoise": denoise, "filler_removal": filler_removal,
               "min_speakers": min_speakers, "max_speakers": max_speakers}
    try:
        rec = run_gate(
            db, settings,
            stream=file.file, filename=file.filename or "unnamed",
            user_id=user.id, source=source, reference=(reference or None), ip=client_ip(request),
            options=options,
        )
    except ValidationError as exc:
        raise HTTPException(exc.status_code, str(exc))
    return public(rec)


@router.get("")
def list_recordings(
    user: CurrentUser = Depends(require_permission("view_transcript")),
    db: Database = Depends(get_db),
) -> list[dict]:
    recs = db.recordings.find().sort("created_at", -1).limit(200)
    return [public(r) for r in recs]


@router.get("/{recording_id}")
def get_recording(
    recording_id: str,
    user: CurrentUser = Depends(require_permission("view_transcript")),
    db: Database = Depends(get_db),
) -> dict:
    return public(_get_recording(db, recording_id))


@router.get("/{recording_id}/transcript")
def get_transcript(
    recording_id: str,
    request: Request,
    original: bool = False,
    user: CurrentUser = Depends(require_permission("view_transcript")),
    db: Database = Depends(get_db),
) -> dict:
    rec = _get_recording(db, recording_id)
    return _transcript_for(db, rec, user, original, request, "view_transcript")


@router.get("/{recording_id}/export.json")
def export_transcript(
    recording_id: str,
    request: Request,
    original: bool = False,
    user: CurrentUser = Depends(require_permission("view_transcript")),
    db: Database = Depends(get_db),
) -> Response:
    rec = _get_recording(db, recording_id)
    view = _transcript_for(db, rec, user, original, request, "export_json")
    return Response(
        content=json.dumps(view, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{recording_id}.json"'},
    )


@router.get("/{recording_id}/audio")
def get_audio(
    recording_id: str,
    request: Request,
    user: CurrentUser = Depends(require_permission("download_audio")),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    rec = _get_recording(db, recording_id)
    if not rec.get("audio_path"):
        raise HTTPException(404, "No stored audio for this recording.")
    try:
        path, job_dir = decrypt_to_temp(settings, rec)
    except (DecryptionError, IntegrityError) as exc:
        audit.record(db, "process_fail", user_id=user.id, recording_id=recording_id,
                     detail=f"audio access failed: {type(exc).__name__}", ip=client_ip(request))
        raise HTTPException(500, "Stored audio could not be verified or decrypted.")
    audit.record(db, "download_audio", user_id=user.id, recording_id=recording_id, ip=client_ip(request))
    media_type = AUDIO_TYPES.get(Path(rec["original_filename"]).suffix.lower(), "application/octet-stream")
    # The decrypted copy is deleted as soon as the response has been sent.
    return FileResponse(path, media_type=media_type, background=BackgroundTask(wipe_dir, job_dir))


@router.post("/{recording_id}/retry")
def retry_recording(
    recording_id: str,
    request: Request,
    user: CurrentUser = Depends(require_permission("retry")),
    db: Database = Depends(get_db),
) -> dict:
    rec = _get_recording(db, recording_id)
    if not rec.get("audio_path"):
        raise HTTPException(409, "This recording failed the gate and has no stored audio to process.")
    # Atomic: only a failed recording can be re-queued (prevents double-queueing).
    updated = db.recordings.find_one_and_update(
        {"_id": recording_id, "status": "failed"},
        {"$set": {"status": "queued", "error": None, "stage": None}, "$inc": {"attempts": 1}},
        return_document=True,
    )
    if updated is None:
        raise HTTPException(409, f"Only failed recordings can be retried (status: {rec['status']}).")
    audit.record(db, "retry", user_id=user.id, recording_id=recording_id, ip=client_ip(request))
    return public(updated)
