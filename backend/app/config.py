"""Application settings, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    mongo_uri: str = "mongodb://127.0.0.1:27017"
    mongo_db: str = "speech_to_text"

    jwt_secret: str = Field(default="", repr=False)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    storage_dir: Path = ROOT_DIR / "storage"
    max_upload_mb: int = 500
    max_duration_sec: int = 7200
    require_ffprobe: bool = False
    enable_docs: bool = False  # /docs lists every endpoint; keep off outside development

    @property
    def audio_dir(self) -> Path:
        return self.storage_dir / "audio"

    @property
    def tmp_dir(self) -> Path:
        return self.storage_dir / "tmp"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def require_jwt_secret(self) -> str:
        if len(self.jwt_secret) < 16 or self.jwt_secret == "change-me":
            raise RuntimeError("JWT_SECRET is missing or too weak; set a long random value.")
        return self.jwt_secret


@lru_cache
def get_settings() -> Settings:
    return Settings()
