"""Create a user. Usage: python scripts/create_user.py <username> <admin|uploader|viewer>
The password is read from the prompt (never passed on the command line)."""
import getpass
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.auth.roles import ROLES  # noqa: E402
from app.auth.security import hash_password  # noqa: E402
from app.db.mongo import get_db  # noqa: E402

if len(sys.argv) != 3 or sys.argv[2] not in ROLES:
    sys.exit(__doc__)
username, role = sys.argv[1], sys.argv[2]
password = getpass.getpass("Password: ")
if len(password) < 10:
    sys.exit("Password must be at least 10 characters.")
if password != getpass.getpass("Repeat password: "):
    sys.exit("Passwords do not match.")
get_db().users.insert_one(
    {
        "_id": uuid.uuid4().hex,
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now(timezone.utc),
    }
)
print(f"Created {role} user '{username}'.")
