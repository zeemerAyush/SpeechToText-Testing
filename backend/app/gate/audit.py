"""Audit log: every gate decision and data access is recorded."""
from __future__ import annotations

from datetime import datetime, timezone

from pymongo.database import Database

ACTIONS = {
    "login", "login_failed", "upload", "gate_pass", "gate_fail", "queued",
    "process_start", "process_done", "process_fail",
    "view_transcript", "view_original_text", "download_audio", "export_json",
    "access_denied",
}


def record(
    db: Database,
    action: str,
    *,
    user_id: str | None = None,
    recording_id: str | None = None,
    detail: str = "",
    ip: str | None = None,
) -> None:
    if action not in ACTIONS:
        raise ValueError(f"Unknown audit action: {action}")
    db.audit_log.insert_one(
        {
            "user_id": user_id,
            "action": action,
            "recording_id": recording_id,
            "detail": detail[:500],
            "ip": ip,
            "at": datetime.now(timezone.utc),
        }
    )
