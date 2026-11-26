"""Role-based access control (DESIGN.md section 10)."""
from __future__ import annotations

ROLES = {"admin", "uploader", "viewer"}

PERMISSIONS: dict[str, set[str]] = {
    "upload": {"admin", "uploader"},
    "view_transcript": {"admin", "uploader", "viewer"},
    "view_original_text": {"admin"},
    "download_audio": {"admin"},
    "retry": {"admin"},
}


def can(role: str, permission: str) -> bool:
    return role in PERMISSIONS.get(permission, set())
