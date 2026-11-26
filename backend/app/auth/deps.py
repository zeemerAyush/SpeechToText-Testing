"""FastAPI dependencies: authentication and authorization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo.database import Database

from app.auth.roles import can
from app.auth.security import decode_token
from app.config import Settings, get_settings
from app.db.mongo import get_db
from app.gate import audit

_bearer = HTTPBearer(auto_error=False)


@dataclass
class CurrentUser:
    id: str
    username: str
    role: str


def client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED, "Authentication required.", headers={"WWW-Authenticate": "Bearer"}
    )
    if creds is None:
        raise unauthorized
    try:
        claims = decode_token(settings, creds.credentials)
    except jwt.PyJWTError:
        raise unauthorized
    # Re-read the user so role changes / removals take effect immediately.
    user = db.users.find_one({"_id": claims["sub"]})
    if user is None:
        raise unauthorized
    return CurrentUser(id=str(user["_id"]), username=user["username"], role=user["role"])


def require_permission(permission: str) -> Callable[..., CurrentUser]:
    def dependency(
        request: Request,
        user: CurrentUser = Depends(get_current_user),
        db: Database = Depends(get_db),
    ) -> CurrentUser:
        if not can(user.role, permission):
            audit.record(db, "access_denied", user_id=user.id, detail=f"{permission} ({request.url.path})",
                         ip=client_ip(request))
            raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have permission for this action.")
        return user

    return dependency
