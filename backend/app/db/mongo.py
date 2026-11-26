"""MongoDB access. One shared client per process."""
from __future__ import annotations

from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.config import get_settings


@lru_cache
def get_db() -> Database:
    s = get_settings()
    client = MongoClient(s.mongo_uri, tz_aware=True, serverSelectionTimeoutMS=5000)
    db = client[s.mongo_db]
    ensure_indexes(db)
    return db


def get_db_or_none() -> Database | None:
    """For the health check: report MongoDB as unavailable instead of failing the request."""
    try:
        return get_db()
    except PyMongoError:
        return None


def ensure_indexes(db: Database) -> None:
    db.users.create_index("username", unique=True)
    db.recordings.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
    db.transcripts.create_index("recording_id", unique=True)
    db.audit_log.create_index([("recording_id", ASCENDING), ("at", ASCENDING)])
