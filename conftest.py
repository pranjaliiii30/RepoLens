"""Shared fixtures: a small multi-language sample repository and an offline-embedder index."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402

SAMPLE_FILES = {
    "app/__init__.py": "",
    "app/main.py": '''"""Demo web app."""
from flask import Flask
from .auth import verify_token, login_user
from . import db

app = Flask(__name__)


@app.route("/login", methods=["POST"])
def login():
    """Authenticate a user and return a session token."""
    return login_user(request.json)


@app.get("/users")
def list_users():
    """Return all users from the database."""
    return db.query_users()


if __name__ == "__main__":
    app.run(debug=True)
''',
    "app/auth.py": '''import hashlib, jwt

SECRET_KEY = "supersecretvalue123"


def hash_password(pw):
    """Hash a password using md5."""
    return hashlib.md5(pw.encode()).hexdigest()


def verify_token(token):
    """Validate a JWT bearer token and return its claims."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except:
        return None


def login_user(data):
    """Check credentials, then issue a signed JWT session token."""
    if hash_password(data["pw"]) == lookup(data["user"]):
        return jwt.encode({"u": data["user"]}, SECRET_KEY)
    return None


def lookup(user):
    return "abc"


def unused_helper():
    return 1
''',
    "app/db.py": '''import sqlite3


def get_conn():
    """Open a database connection."""
    return sqlite3.connect("x.db")


def query_users():
    """Run the SQL query that lists every user row."""
    conn = get_conn()
    return conn.execute("SELECT * FROM users").fetchall()


def find_user(name):
    conn = get_conn()
    return conn.execute("select * from users where name='%s'" % name).fetchall()
''',
    "tests/test_auth.py": '''from app.auth import verify_token


def test_verify():
    assert verify_token("x") is None
''',
    "web/server.js": '''const express = require('express');
const { check } = require('./util');
const app = express();
app.get('/health', (req, res) => res.send('ok'));
app.post('/api/items', createItem);
function createItem(req, res) { check(req); res.json({}); }
''',
    "web/util.js": "function check(r) { return !!r; }\nmodule.exports = { check };\n",
    "requirements.txt": "flask\njwt\npytest\n",
    "README.md": "# Demo\n\nA tiny demo app.\n\n## Run\n\n```bash\npython -m app.main\n```\n",
}


def write_sample(root: Path) -> Path:
    for rel, text in SAMPLE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def sample_dir(tmp_path):
    return write_sample(tmp_path / "sample")


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def sample_index(sample_dir, data_dir):
    """Indexed sample repo (offline embedder, so no model download)."""
    from core.indexer import IndexParams, build_index
    from core.ingest import SourceSpec, acquire

    config.ALLOW_LOCAL_PATHS = True
    with acquire(SourceSpec(kind="local", value=str(sample_dir))) as snap:
        return build_index(snap, IndexParams(embedder=config.OFFLINE_EMBEDDER), data_dir=data_dir)


@pytest.fixture
def sample_stack(sample_index, data_dir):
    from core.indexer import open_stack

    return open_stack(sample_index, data_dir)
