"""Small shared helpers: tokenisation, path utilities, formatting."""
from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath

STOPWORDS = frozenset(
    """a an the and or of to in on for with is are was were be been it its this that these those
    do does did how what where when which who why can could should would will i we you they he she
    me my our your their from by as at if then else not no yes into out up down over under about
    there here than so such via per use used using show find list give tell explain please
    self cls def return import pass none true false""".split()
)

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(identifier: str) -> list[str]:
    """``verifyToken`` / ``verify_token`` -> ``['verify', 'token']`` (lower-cased)."""
    parts: list[str] = []
    for chunk in re.split(r"[_\W]+", identifier):
        if not chunk:
            continue
        parts.extend(p.lower() for p in _CAMEL_SPLIT.split(chunk) if p)
    return parts


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Code-aware tokeniser used by both BM25 and the offline embedder.

    Identifiers contribute their joined form *and* their sub-words, so a query for
    "verify token" matches ``verify_token`` and ``verifyToken``.
    """
    tokens: list[str] = []
    for raw in _IDENT_RE.findall(text):
        parts = split_identifier(raw)
        if not parts:
            continue
        if len(parts) > 1:
            tokens.append(raw.lower())
        for p in parts:
            if len(p) < 2:
                continue
            if drop_stopwords and p in STOPWORDS:
                continue
            tokens.append(p)
    return tokens


def posix(path: str) -> str:
    return str(PurePosixPath(path.replace("\\", "/")))


TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "e2e", "testing"}


def is_test_path(path: str) -> bool:
    p = PurePosixPath(posix(path))
    name = p.name.lower()
    if any(part.lower() in TEST_DIRS for part in p.parts[:-1]):
        return True
    if name.startswith("test_") or name.startswith("tests."):
        return True
    stem = p.stem.lower()
    if stem.endswith(("_test", ".test", ".spec", "_spec")):
        return True
    return bool(re.search(r"(Tests?|Spec)\.(java|cs|kt|scala|rb)$", p.name))


def matches_any(path: str, patterns: list[str]) -> bool:
    path = posix(path)
    base = path.rsplit("/", 1)[-1]
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(base, pat):
            return True
        if pat.endswith("/") and (path.startswith(pat) or f"/{pat}" in f"/{path}"):
            return True
    return False


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"  # pragma: no cover


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    return text if len(text) <= limit else text[: max(0, limit - len(suffix))] + suffix


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._").lower()
    return slug[:max_len] or "repo"


def short_path(path: str, parts: int = 3) -> str:
    segs = posix(path).split("/")
    return "/".join(segs[-parts:]) if len(segs) > parts else "/".join(segs)
