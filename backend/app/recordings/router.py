"""Recording endpoints. Uploads always pass through the guardrail gate."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pymongo.database import Database

from app.auth.deps import CurrentUser, client_ip, require_permission
from app.config import Settings, get_settings
from app.db.mongo import get_db
from app.gate import audit
from app.gate.service import run_gate
from app.gate.validation import ValidationError

router = APIRouter(prefix="/recordings", tags=["recordings"])

# Fields that must never leave the server.
_HIDDEN = {"audio_path", "data_key"}


def public(rec: dict) -> dict:
    out = {("id" if k == "_id" else k): v for k, v in rec.items() if k not in _HIDDEN}
    return out


@router.post("", status_code=201)
def upload_recording(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("upload"),
    reference: str | None = Form(None),
    user: CurrentUser = Depends(require_permission("upload")),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    if source not in {"upload", "browser_recording"}:
        raise HTTPException(400, "source must be 'upload' or 'browser_recording'.")
    try:
        rec = run_gate(
            db, settings,
            stream=file.file, filename=file.filename or "unnamed",
            user_id=user.id, source=source, reference=(reference or None), ip=client_ip(request),
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
    rec = db.recordings.find_one({"_id": recording_id})
    if rec is None:
        raise HTTPException(404, "Recording not found.")
    return public(rec)
