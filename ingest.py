"""Repository acquisition: URL parsing, safe cloning, safe ZIP extraction, file filtering."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import config
from .models import FileInfo
from .utils import matches_any, posix, slugify

log = logging.getLogger("repolens.ingest")
ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------- errors
class RepoError(Exception):
    """Base class: message is safe to show to the user."""


class InvalidRepoURL(RepoError):
    pass


class RepoNotFound(RepoError):
    pass


class RepoAuthRequired(RepoError):
    pass


class RepoNetworkError(RepoError):
    pass


class RepoTooLarge(RepoError):
    """Raised when a repo exceeds the soft limits; the UI can offer 'continue anyway'."""

    def __init__(self, message: str, *, size_mb: float | None = None, n_files: int | None = None):
        super().__init__(message)
        self.size_mb = size_mb
        self.n_files = n_files


class UnsafeArchive(RepoError):
    pass


# --------------------------------------------------------------------------- URL parsing
@dataclass(frozen=True)
class RepoRef:
    host: str
    owner: str
    repo: str
    branch: str | None = None

    @property
    def clone_url(self) -> str:
        return f"https://{self.host}/{self.owner}/{self.repo}.git"

    @property
    def display(self) -> str:
        return f"{self.owner}/{self.repo}"


_SHORTHAND = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")
_SSH = re.compile(r"^git@([\w.-]+):([\w.-]+)/([\w.-]+?)(?:\.git)?$")


def parse_repo_url(text: str) -> RepoRef:
    """Accept https URLs, ``git@host:o/r.git``, ``host/o/r`` and ``owner/repo``."""
    raw = (text or "").strip()
    if not raw:
        raise InvalidRepoURL("Paste a GitHub URL such as https://github.com/owner/repo")

    m = _SSH.match(raw)
    if m:
        host, owner, repo = m.group(1).lower(), m.group(2), m.group(3)
        return _validated(host, owner, repo, None)

    m = _SHORTHAND.match(raw)
    if m and "." not in m.group(1):
        return _validated("github.com", m.group(1), re.sub(r"\.git$", "", m.group(2)), None)

    candidate = raw if "://" in raw else f"https://{raw}"
    mm = re.match(r"^https?://(?:[^@/]+@)?([^/\s:]+)(?::\d+)?/(.+)$", candidate)
    if not mm:
        raise InvalidRepoURL("That doesn't look like a repository URL. Try https://github.com/owner/repo")
    host = mm.group(1).lower().removeprefix("www.")
    segs = [s for s in mm.group(2).split("/") if s]
    if len(segs) < 2:
        raise InvalidRepoURL("The URL needs an owner and a repository name, e.g. github.com/owner/repo")
    owner, repo = segs[0], re.sub(r"\.git$", "", segs[1])
    branch = None
    if len(segs) >= 4 and segs[2] in ("tree", "blob", "-"):
        branch = segs[3] if segs[2] != "-" else (segs[4] if len(segs) > 4 else None)
    return _validated(host, owner, repo, branch)


def _validated(host: str, owner: str, repo: str, branch: str | None) -> RepoRef:
    if host not in config.ALLOWED_GIT_HOSTS:
        raise InvalidRepoURL(
            f"Host '{host}' is not supported. Supported hosts: {', '.join(config.ALLOWED_GIT_HOSTS)}."
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise InvalidRepoURL("Owner/repository names may only contain letters, digits, '.', '_' and '-'.")
    if owner in (".", "..") or repo in (".", "..") or owner.startswith("-") or repo.startswith("-"):
        raise InvalidRepoURL("That owner/repository name isn't valid.")
    return RepoRef(host, owner, repo, branch)


# --------------------------------------------------------------------------- source spec
@dataclass
class SourceSpec:
    kind: str                       # "url" | "zip" | "local"
    value: str = ""                 # URL / local path / zip filename
    branch: str | None = None
    token: str | None = None
    zip_bytes: bytes | None = None


@dataclass
class SourceSnapshot:
    name: str
    source: str
    root: Path
    branch: str
    commit: str
    kind: str
    key_branch: str = ""            # stable branch label used for cache keys ("HEAD" when unspecified)


def _redact(text: str, secrets: list[str | None]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"          # never hang waiting for credentials
    env["GIT_ASKPASS"] = "echo"
    env["GCM_INTERACTIVE"] = "never"
    return env


def _authed_url(ref: RepoRef, token: str | None) -> str:
    if token:
        return f"https://x-access-token:{token}@{ref.host}/{ref.owner}/{ref.repo}.git"
    return ref.clone_url


def _classify_git_failure(stderr: str, ref: RepoRef, token: str | None) -> RepoError:
    low = stderr.lower()
    if any(s in low for s in ("could not resolve host", "unable to access", "network is unreachable",
                              "connection timed out", "connection refused", "ssl")) and "not found" not in low:
        return RepoNetworkError("Couldn't reach the git host. Check your internet connection and try again.")
    if "remote branch" in low and "not found" in low:
        return RepoNotFound("That branch doesn't exist in the repository. Leave the branch field empty for the default.")
    if "not found" in low or "does not exist" in low:
        if token:
            return RepoNotFound(f"Repository {ref.display} was not found, or your token can't see it.")
        return RepoAuthRequired(
            f"Repository {ref.display} was not found. If it is private, add a GitHub token in the sidebar."
        )
    if any(s in low for s in ("authentication failed", "could not read username", "terminal prompts disabled",
                              "permission denied", "403")):
        return RepoAuthRequired(
            "Access denied. This repository looks private — add a GitHub token with read access."
            if not token else "Access denied. Check that your token has read access to this repository."
        )
    return RepoNetworkError("Cloning failed. " + _redact(stderr.strip().splitlines()[-1] if stderr.strip() else "", [token]))


def peek_remote_commit(ref: RepoRef, branch: str | None, token: str | None, timeout: int = 25) -> str | None:
    """Resolve the current commit with ``git ls-remote`` (no clone). Returns None on any failure."""
    try:
        target = branch or "HEAD"
        proc = subprocess.run(
            ["git", "ls-remote", _authed_url(ref, token), target],
            capture_output=True, text=True, timeout=timeout, env=_git_env(),
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        return proc.stdout.split()[0]
    except Exception as exc:  # network, git missing, timeout
        log.debug("ls-remote failed: %s", _redact(str(exc), [token]))
        return None


def github_size_mb(ref: RepoRef, token: str | None, timeout: int = 6) -> float | None:
    """Best-effort repo size from the GitHub API (None if unavailable)."""
    if ref.host != "github.com":
        return None
    req = urllib.request.Request(f"https://api.github.com/repos/{ref.owner}/{ref.repo}",
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": "repolens"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            data = json.loads(resp.read().decode("utf-8"))
        return float(data.get("size", 0)) / 1024.0
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def clone_repo(ref: RepoRef, dest: Path, branch: str | None, token: str | None,
               timeout: int = config.CLONE_TIMEOUT_S) -> None:
    """Shallow clone (depth=1). Never prompts for credentials; token is redacted from errors."""
    cmd = ["git", "clone", "--depth=1", "--single-branch", "--no-tags"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [_authed_url(ref, token), str(dest)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_git_env())
    except subprocess.TimeoutExpired as exc:
        raise RepoNetworkError(f"Cloning took longer than {timeout}s and was cancelled.") from exc
    except FileNotFoundError as exc:
        raise RepoError("git is not installed on this machine. Install git or upload a ZIP instead.") from exc
    if proc.returncode != 0:
        raise _classify_git_failure(_redact(proc.stderr, [token]), ref, token)


def git_head_commit(path: Path) -> str:
    try:
        from git import Repo  # GitPython

        return Repo(str(path)).head.commit.hexsha
    except Exception:
        try:
            out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""


def git_current_branch(path: Path) -> str:
    try:
        from git import Repo

        return Repo(str(path)).active_branch.name
    except Exception:
        try:
            out = subprocess.run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""


# --------------------------------------------------------------------------- ZIP
def extract_zip(data: bytes | Path, dest: Path,
                max_uncompressed_mb: int = config.MAX_ZIP_UNCOMPRESSED_MB) -> Path:
    """Extract defensively (zip-slip, symlinks, zip-bombs). Returns the project root inside ``dest``."""
    import io

    handle = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else open(data, "rb")
    dest = dest.resolve()
    try:
        try:
            zf = zipfile.ZipFile(handle)
        except zipfile.BadZipFile as exc:
            raise UnsafeArchive("That file isn't a valid ZIP archive.") from exc
        with zf:
            infos = zf.infolist()
            total = sum(i.file_size for i in infos)
            if total > max_uncompressed_mb * 1024 * 1024:
                raise RepoTooLarge(f"ZIP expands to {total / 1048576:.0f} MB (limit {max_uncompressed_mb} MB).",
                                   size_mb=total / 1048576)
            if len(infos) > config.HARD_MAX_FILES * 2:
                raise UnsafeArchive("ZIP contains too many entries.")
            for info in infos:
                name = info.filename.replace("\\", "/")
                if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in name.split("/"):
                    raise UnsafeArchive(f"ZIP rejected: unsafe path '{name}'.")
                mode = (info.external_attr >> 16) & 0xFFFF
                if mode and (mode & 0o170000) == 0o120000:      # symlink entry
                    continue
                target = (dest / name).resolve()
                if not str(target).startswith(str(dest)):
                    raise UnsafeArchive(f"ZIP rejected: path escapes target directory ('{name}').")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 256)
    finally:
        handle.close()

    entries = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


# --------------------------------------------------------------------------- file filtering
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__", "dist", "build",
    ".next", ".nuxt", "target", ".idea", ".vscode", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "coverage", "htmlcov", "site-packages", "bower_components", ".gradle", ".terraform", "__MACOSX",
    ".cache", ".parcel-cache", ".turbo", ".svelte-kit", "Pods", ".eggs", "*.egg-info", ".repolens_data",
})
LOCK_FILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock", "Cargo.lock",
    "go.sum", "composer.lock", "Gemfile.lock", "bun.lockb", "uv.lock", "gradle.lockfile", ".DS_Store",
})
BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svgz", ".tiff", ".psd",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".class", ".jar", ".war",
    ".pyc", ".pyo", ".whl", ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".flac", ".ogg", ".webm",
    ".sqlite", ".sqlite3", ".db", ".parquet", ".pkl", ".pickle", ".npy", ".npz", ".h5", ".onnx", ".pt",
    ".pth", ".ckpt", ".safetensors", ".wasm", ".dat", ".iso", ".dmg", ".apk", ".ipa", ".lock",
})

LANGUAGE_BY_EXT = {
    ".py": "python", ".pyw": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".cs": "csharp",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".kts": "kotlin", ".swift": "swift", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell", ".bat": "shell",
    ".sql": "sql", ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss", ".sass": "scss",
    ".vue": "vue", ".svelte": "svelte",
    ".md": "markdown", ".markdown": "markdown", ".rst": "markdown", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    ".xml": "xml", ".gradle": "groovy", ".proto": "protobuf", ".tf": "hcl", ".r": "r", ".lua": "lua",
    ".dart": "dart", ".ex": "elixir", ".exs": "elixir", ".hs": "haskell", ".ipynb": "json",
}
SPECIAL_FILENAMES = {
    "dockerfile": "dockerfile", "makefile": "makefile", "cmakelists.txt": "cmake", "jenkinsfile": "groovy",
    "gemfile": "ruby", "rakefile": "ruby", "procfile": "text", ".env.example": "ini", "requirements.txt": "text",
}
CODE_LANGS = {"python", "javascript", "typescript", "java", "go", "rust", "csharp", "c", "cpp", "ruby", "php",
              "kotlin", "swift", "scala", "shell", "dart", "lua", "r", "elixir", "haskell", "powershell"}
DOC_LANGS = {"markdown", "text"}
CONFIG_LANGS = {"json", "yaml", "toml", "ini", "xml", "hcl", "protobuf", "dockerfile", "makefile", "cmake", "groovy"}


def detect_language(path: str) -> str:
    p = Path(path)
    special = SPECIAL_FILENAMES.get(p.name.lower())
    if special:
        return special
    return LANGUAGE_BY_EXT.get(p.suffix.lower(), "")


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(8192)
    except OSError:
        return True


@dataclass
class ScanResult:
    files: list[FileInfo] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def scan_files(root: Path, extra_ignores: list[str] | None = None,
               max_file_bytes: int = config.MAX_FILE_BYTES,
               hard_max_files: int = config.HARD_MAX_FILES) -> ScanResult:
    """Walk ``root`` and return the text files worth indexing (symlinks are never followed)."""
    extra = list(extra_ignores or [])
    root = root.resolve()
    result = ScanResult()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = posix(os.path.relpath(dirpath, root))
        rel_dir = "" if rel_dir == "." else rel_dir
        keep = []
        for d in sorted(dirnames):
            full = Path(dirpath) / d
            if d in IGNORED_DIRS or d.endswith(".egg-info"):
                result.skip("ignored directory")
                continue
            if full.is_symlink():
                result.skip("symlink")
                continue
            if extra and matches_any(posix(f"{rel_dir}/{d}/" if rel_dir else f"{d}/"), extra):
                result.skip("user ignore pattern")
                continue
            keep.append(d)
        dirnames[:] = keep

        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = posix(f"{rel_dir}/{name}" if rel_dir else name)
            if full.is_symlink():
                result.skip("symlink")
                continue
            low = name.lower()
            ext = full.suffix.lower()
            if name in LOCK_FILES or low.endswith((".min.js", ".min.css", ".map", ".bundle.js")):
                result.skip("lock/minified file")
                continue
            if ext in BINARY_EXTS:
                result.skip("binary file")
                continue
            if extra and matches_any(rel, extra):
                result.skip("user ignore pattern")
                continue
            try:
                size = full.stat().st_size
            except OSError:
                result.skip("unreadable")
                continue
            if size == 0:
                result.skip("empty file")
                continue
            if size > max_file_bytes:
                result.skip(f"larger than {max_file_bytes // 1024} KB")
                continue
            language = detect_language(rel)
            if not language:
                if _is_binary(full):
                    result.skip("binary file")
                    continue
                language = "text"
            elif _is_binary(full):
                result.skip("binary file")
                continue
            try:
                data = full.read_bytes()
            except OSError:
                result.skip("unreadable")
                continue
            text = data.decode("utf-8", errors="replace")
            if text.count("�") > max(8, len(text) // 10):
                result.skip("not valid text")
                continue
            n_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
            result.files.append(FileInfo(rel, language, size, n_lines))
            result.total_bytes += size
            if len(result.files) >= hard_max_files:
                result.skip("hard file limit reached")
                return result
    return result


def materialize(root: Path, files: list[FileInfo], dest: Path) -> None:
    """Copy the kept files into ``dest`` (so .git, node_modules etc. never hit disk long-term)."""
    root = root.resolve()
    for info in files:
        src = (root / info.path).resolve()
        if not str(src).startswith(str(root)):
            continue
        target = dest / info.path
        target.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_bytes().decode("utf-8", errors="replace").lstrip("﻿")
        target.write_text(text, encoding="utf-8", newline="\n")


def fingerprint_dir(root: Path, files: list[FileInfo] | None = None) -> str:
    h = hashlib.sha1()
    items = files if files is not None else scan_files(root).files
    for f in sorted(items, key=lambda i: i.path):
        try:
            mtime = int((root / f.path).stat().st_mtime)
        except OSError:
            mtime = 0
        h.update(f"{f.path}:{f.size}:{mtime}\n".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- orchestration
def describe_source(spec: SourceSpec) -> tuple[str, str, str | None]:
    """Return ``(name, source_label, commit_or_None)`` cheaply, for cache lookups."""
    if spec.kind == "url":
        ref = parse_repo_url(spec.value)
        branch = spec.branch or ref.branch
        commit = peek_remote_commit(ref, branch, spec.token)
        return ref.display, f"{ref.host}/{ref.display}", commit
    if spec.kind == "zip":
        if not spec.zip_bytes:
            raise RepoError("Upload a ZIP file first.")
        name = Path(spec.value or "upload.zip").stem
        return name, f"zip:{spec.value}", hashlib.sha1(spec.zip_bytes).hexdigest()
    if spec.kind == "local":
        if not config.ALLOW_LOCAL_PATHS:
            raise RepoError("Local paths are disabled. Set REPOLENS_ALLOW_LOCAL=1 to enable them on a private install.")
        p = Path(spec.value).expanduser()
        if not p.is_dir():
            raise RepoError(f"'{spec.value}' is not a directory.")
        commit = git_head_commit(p) or fingerprint_dir(p)
        return p.name or "local", f"local:{p.resolve()}", commit
    raise RepoError(f"Unknown source kind '{spec.kind}'.")


@dataclass
class ResolvedSource:
    name: str
    source: str
    key_branch: str
    commit: str            # "" when it can't be determined cheaply (cache is then skipped)


def resolve_source(spec: SourceSpec) -> ResolvedSource:
    """Cheap lookup of everything that identifies a snapshot (no clone), used for cache hits."""
    if spec.kind == "url":
        ref = parse_repo_url(spec.value)
        branch = spec.branch or ref.branch
        return ResolvedSource(ref.display, f"{ref.host}/{ref.display}", branch or "HEAD",
                              peek_remote_commit(ref, branch, spec.token) or "")
    if spec.kind == "zip":
        if not spec.zip_bytes:
            raise RepoError("Upload a ZIP file first.")
        return ResolvedSource(Path(spec.value or "upload.zip").stem, f"zip:{spec.value}", "upload",
                              hashlib.sha1(spec.zip_bytes).hexdigest())
    name, label, commit = describe_source(spec)
    p = Path(spec.value).expanduser()
    return ResolvedSource(name, label, git_current_branch(p) or "local", commit or "")


@contextmanager
def acquire(spec: SourceSpec, *, allow_large: bool = False,
            progress: ProgressFn | None = None) -> Iterator[SourceSnapshot]:
    """Fetch the source into a temp dir (or point at a local dir); always cleans up."""
    progress = progress or (lambda f, m: None)
    tmp = Path(tempfile.mkdtemp(prefix="repolens_"))
    try:
        if spec.kind == "url":
            ref = parse_repo_url(spec.value)
            branch = spec.branch or ref.branch
            if not allow_large:
                size = github_size_mb(ref, spec.token)
                if size is not None and size > config.MAX_REPO_MB:
                    raise RepoTooLarge(f"{ref.display} is about {size:.0f} MB (soft limit {config.MAX_REPO_MB} MB).",
                                       size_mb=size)
            progress(0.02, f"Cloning {ref.display} …")
            dest = tmp / "repo"
            clone_repo(ref, dest, branch, spec.token)
            commit = git_head_commit(dest)
            snap = SourceSnapshot(ref.display, f"{ref.host}/{ref.display}", dest,
                                  branch or git_current_branch(dest) or "default", commit, "url",
                                  key_branch=branch or "HEAD")
        elif spec.kind == "zip":
            progress(0.02, "Extracting ZIP …")
            root = extract_zip(spec.zip_bytes or b"", tmp / "zip")
            name = Path(spec.value or "upload.zip").stem
            snap = SourceSnapshot(name, f"zip:{spec.value}", root, "upload",
                                  hashlib.sha1(spec.zip_bytes or b"").hexdigest(), "zip", key_branch="upload")
        elif spec.kind == "local":
            name, label, commit = describe_source(spec)
            _b = git_current_branch(Path(spec.value).expanduser()) or "local"
            snap = SourceSnapshot(name, label, Path(spec.value).expanduser().resolve(), _b, commit or "",
                                  "local", key_branch=_b)
        else:
            raise RepoError(f"Unknown source kind '{spec.kind}'.")
        yield snap
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_repo_id(name: str, source: str, branch: str, commit: str, embedder: str) -> str:
    digest = hashlib.sha1(f"{source}|{branch}|{commit}|{embedder}".encode()).hexdigest()[:8]
    return f"{slugify(name, 32)}-{digest}"
