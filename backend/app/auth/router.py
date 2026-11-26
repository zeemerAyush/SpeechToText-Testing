from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from pymongo.database import Database

from app.auth.deps import client_ip
from app.auth.security import create_token, verify_password
from app.config import Settings, get_settings
from app.db.mongo import get_db
from app.gate import audit

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    request: Request,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    user = db.users.find_one({"username": body.username})
    ok = verify_password(body.password, user["password_hash"] if user else None)
    if not user or not ok:
        audit.record(db, "login_failed", detail=f"username={body.username[:64]}", ip=client_ip(request))
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password.")
    audit.record(db, "login", user_id=str(user["_id"]), ip=client_ip(request))
    return TokenResponse(access_token=create_token(settings, str(user["_id"]), user["role"]), role=user["role"])
