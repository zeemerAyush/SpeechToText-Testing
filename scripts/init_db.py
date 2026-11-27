"""Create the MongoDB indexes and verify the connection. Safe to run repeatedly.

    python scripts/init_db.py

Reads MONGO_URI / MONGO_DB from .env. Collections are created on first use; this script makes sure the
unique and lookup indexes exist (usernames, transcripts per recording, job queue, audit trail) and reports
what it found, so a wrong URI or missing credentials show up now instead of at the first upload.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pymongo import MongoClient  # noqa: E402
from pymongo.errors import PyMongoError  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.mongo import ensure_indexes  # noqa: E402

COLLECTIONS = ["users", "recordings", "transcripts", "audit_log"]


def main() -> int:
    settings = get_settings()
    client = MongoClient(settings.mongo_uri, tz_aware=True, serverSelectionTimeoutMS=5000)
    try:
        version = client.server_info()["version"]
        db = client[settings.mongo_db]
        ensure_indexes(db)
        host, port = client.address or ("?", "?")
        print(f"Connected to MongoDB {version} at {host}:{port}, database '{settings.mongo_db}'.")
        for name in COLLECTIONS:
            indexes = sorted(db[name].index_information())
            print(f"  {name:<12} {db[name].estimated_document_count():>6} docs   indexes: {', '.join(indexes)}")
        if db.users.count_documents({}) == 0:
            print("No users yet. Create one:  python scripts/create_user.py <username> admin")
    except PyMongoError as exc:
        # The message names the host/problem; it does not include the password.
        print(f"error: MongoDB check failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        print("Check that mongod is running and MONGO_URI in .env is correct "
              "(credentials must be URL-encoded).", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
