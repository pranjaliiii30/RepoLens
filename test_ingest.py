"""Ingestion: URL parsing, safe ZIP extraction, file filtering, git error classification."""
from __future__ import annotations

import io
import zipfile

import pytest

from core import config
from core.ingest import (InvalidRepoURL, RepoAuthRequired, RepoNetworkError, RepoNotFound, RepoRef, RepoTooLarge,
                         UnsafeArchive, _authed_url, _classify_git_failure, _redact, detect_language, extract_zip,
                         make_repo_id, parse_repo_url, scan_files)


# ----------------------------------------------------------------------------- URLs
def test_parse_common_url_forms():
    for text in ("https://github.com/psf/requests", "https://github.com/psf/requests.git",
                 "github.com/psf/requests", "psf/requests", "git@github.com:psf/requests.git",
                 "https://www.github.com/psf/requests/", "  https://github.com/psf/requests  "):
        ref = parse_repo_url(text)
        assert (ref.host, ref.owner, ref.repo) == ("github.com", "psf", "requests"), text


def test_parse_branch_from_tree_url():
    ref = parse_repo_url("https://github.com/pallets/click/tree/8.1.x/src")
    assert ref.branch == "8.1.x" and ref.clone_url == "https://github.com/pallets/click.git"


def test_parse_rejects_bad_input():
    for bad in ("", "   ", "not a url", "https://github.com/onlyowner", "https://evil.example.com/a/b",
                "https://github.com/a b/c", "file:///etc/passwd", "https://github.com/../etc"):
        with pytest.raises(InvalidRepoURL):
            parse_repo_url(bad)


def test_token_is_only_in_clone_url_and_gets_redacted():
    ref = RepoRef("github.com", "o", "r")
    assert "TOKEN123" in _authed_url(ref, "TOKEN123") and "TOKEN123" not in _authed_url(ref, None)
    assert "TOKEN123" not in _redact("fatal: https://x-access-token:TOKEN123@github.com/o/r", ["TOKEN123"])


def test_git_failures_map_to_friendly_errors():
    ref = RepoRef("github.com", "o", "r")
    assert isinstance(_classify_git_failure("fatal: repository 'x' not found", ref, None), RepoAuthRequired)
    assert isinstance(_classify_git_failure("fatal: repository 'x' not found", ref, "tok"), RepoNotFound)
    assert isinstance(_classify_git_failure("fatal: Could not resolve host: github.com", ref, None), RepoNetworkError)
    assert isinstance(_classify_git_failure("fatal: Remote branch nope not found in upstream origin", ref, None), RepoNotFound)
    assert isinstance(_classify_git_failure("remote: Permission denied", ref, None), RepoAuthRequired)


def test_detect_language():
    assert detect_language("a/b.py") == "python"
    assert detect_language("x.tsx") == "typescript"
    assert detect_language("Dockerfile") == "dockerfile"
    assert detect_language("weird.xyz") == ""


def test_repo_id_depends_on_commit_and_embedder():
    a = make_repo_id("n", "s", "main", "c1", "e")
    assert a == make_repo_id("n", "s", "main", "c1", "e")
    assert a != make_repo_id("n", "s", "main", "c2", "e") and a != make_repo_id("n", "s", "main", "c1", "e2")


# ----------------------------------------------------------------------------- ZIP
def _zip(entries: dict[str, bytes | str], symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (0o120777 << 16)
            zf.writestr(info, "/etc/passwd")
    return buf.getvalue()


def test_zip_extracts_and_unwraps_single_top_folder(tmp_path):
    root = extract_zip(_zip({"proj/a.py": "x=1\n", "proj/sub/b.py": "y=2\n"}), tmp_path / "out")
    assert root.name == "proj" and (root / "sub" / "b.py").exists()


def test_zip_slip_is_rejected(tmp_path):
    for evil in ("../evil.py", "a/../../evil.py", "/abs.py", "C:/win.py"):
        with pytest.raises(UnsafeArchive):
            extract_zip(_zip({evil: "boom"}), tmp_path / "o")
    assert not (tmp_path / "evil.py").exists()


def test_zip_symlink_entries_are_skipped_not_followed(tmp_path):
    root = extract_zip(_zip({"p/a.py": "x=1\n"}, symlink="p/link"), tmp_path / "out")
    assert not (root / "link").exists()


def test_zip_bomb_and_garbage_are_rejected(tmp_path):
    with pytest.raises(RepoTooLarge):
        extract_zip(_zip({"p/big.txt": "0" * 3_000_000}), tmp_path / "o", max_uncompressed_mb=1)
    with pytest.raises(UnsafeArchive):
        extract_zip(b"this is not a zip", tmp_path / "o2")


# ----------------------------------------------------------------------------- scanning
def _tree(root, files: dict[str, bytes | str]):
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())


def test_scan_skips_noise_and_keeps_source(tmp_path):
    _tree(tmp_path, {
        "src/app.py": "print(1)\n", "README.md": "# hi\n",
        "node_modules/x/index.js": "x", ".git/config": "x", "venv/lib/a.py": "x", "dist/out.js": "x",
        "package-lock.json": "{}", "bundle.min.js": "x" * 50, "logo.png": b"\x89PNG\r\n",
        "data.bin": b"\x00\x01\x02", "empty.py": "", "big.py": "x = 1\n" * 200_000,
        "notes.xyz": "plain text file\n", "blob.xyz": b"\x00" * 100,
    })
    r = scan_files(tmp_path)
    kept = {f.path for f in r.files}
    assert kept == {"src/app.py", "README.md", "notes.xyz"}
    for reason in ("ignored directory", "lock/minified file", "binary file", "empty file"):
        assert r.skipped.get(reason, 0) >= 1, reason
    assert any("larger than" in k for k in r.skipped)


def test_scan_user_ignore_patterns_for_files_and_folders(tmp_path):
    _tree(tmp_path, {"a.py": "x=1\n", "docs/a.md": "# d\n", "gen/x.generated.py": "x=1\n", "src/y.py": "y=1\n"})
    kept = {f.path for f in scan_files(tmp_path, ["docs/", "*.generated.*"]).files}
    assert kept == {"a.py", "src/y.py"}


def test_scan_never_follows_symlinks(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("TOKEN='x'\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "ok.py").write_text("x=1\n")
    try:
        (proj / "link.py").symlink_to(outside / "secret.py")
        (proj / "linkdir").symlink_to(outside, target_is_directory=True)
    except OSError:
        return  # symlinks unavailable on this platform
    kept = {f.path for f in scan_files(proj).files}
    assert kept == {"ok.py"}


def test_scan_reports_line_counts_and_language(tmp_path):
    _tree(tmp_path, {"m.py": "a=1\nb=2\nc=3"})
    f = scan_files(tmp_path).files[0]
    assert (f.language, f.lines) == ("python", 3)


def test_local_paths_are_disabled_unless_opted_in(tmp_path):
    from core.ingest import RepoError, SourceSpec, describe_source

    old = config.ALLOW_LOCAL_PATHS
    config.ALLOW_LOCAL_PATHS = False
    try:
        with pytest.raises(RepoError):
            describe_source(SourceSpec(kind="local", value=str(tmp_path)))
    finally:
        config.ALLOW_LOCAL_PATHS = old
