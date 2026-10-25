"""Central configuration. Everything tunable lives here or in environment variables."""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv missing or broken .env
    pass

APP_NAME = "RepoLens"
TAGLINE = "Chat with any codebase. Every answer cites the exact file and line."

# --- storage -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("REPOLENS_DATA_DIR", ".repolens_data")).expanduser().resolve()

# --- models ------------------------------------------------------------------
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_CHOICES = [
    DEFAULT_GEMINI_MODEL,
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]
OFFLINE_EMBEDDER = "offline-hashing"
DEFAULT_EMBEDDER = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_CHOICES = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "all-mpnet-base-v2",
    OFFLINE_EMBEDDER,
]
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- ingestion limits ----------------------------------------------------------
MAX_REPO_MB = 200
MAX_FILES = 5_000
HARD_MAX_FILES = 20_000
MAX_FILE_BYTES = 500 * 1024
MAX_ZIP_UNCOMPRESSED_MB = 400
CLONE_TIMEOUT_S = 240
ALLOWED_GIT_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")
# Reading arbitrary server paths is only safe on a private, local install.
ALLOW_LOCAL_PATHS = os.getenv("REPOLENS_ALLOW_LOCAL", "0").lower() in {"1", "true", "yes"}

# --- chunking ----------------------------------------------------------------
WINDOW_LINES = 60          # sliding-window fallback
WINDOW_OVERLAP = 10
LONG_SYMBOL_LINES = 150    # symbols longer than this are split into parts
SPLIT_WINDOW = 100
SPLIT_OVERLAP = 10
MAX_EMBED_CHARS = 1800
MAX_CONTEXT_CHARS_PER_CHUNK = 9000

# --- retrieval ---------------------------------------------------------------
RRF_K = 60
DEFAULT_TOP_K = 8
MAX_PER_FILE = 3

# --- misc --------------------------------------------------------------------
HISTORY_TURNS = 3
EXAMPLE_REPOS = [
    ("psf/requests", "https://github.com/psf/requests"),
    ("pallets/click", "https://github.com/pallets/click"),
    ("expressjs/express", "https://github.com/expressjs/express"),
    ("gin-gonic/gin", "https://github.com/gin-gonic/gin"),
]
