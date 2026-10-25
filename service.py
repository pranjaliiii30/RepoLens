"""High-level orchestration used by the UI: cache-aware indexing and the repo overview."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .indexer import IndexParams, build_index, find_cached, get_embedder, load_index
from .ingest import SourceSpec, acquire, make_repo_id, resolve_source
from .insights import (detect_dependencies, detect_tech_stack, find_entry_points, find_routes, folder_summary,
                       repo_stats)
from .llm import GeminiClient, LLMError
from .models import RepoIndex

log = logging.getLogger("repolens.service")
ProgressFn = Callable[[float, str], None]


def index_or_load(spec: SourceSpec, params: IndexParams, *, force: bool = False,
                  progress: ProgressFn | None = None, data_dir: Path | None = None) -> tuple[RepoIndex, bool]:
    """Return ``(index, from_cache)``. A cache hit needs the same repo, branch, commit and embedder."""
    progress = progress or (lambda f, m: None)
    resolved = resolve_source(spec)
    emb, _ = get_embedder(params.embedder)
    repo_id = make_repo_id(resolved.name, resolved.source, resolved.key_branch, resolved.commit, emb.name)
    if not force and resolved.commit and find_cached(repo_id, data_dir):
        progress(0.5, "Found a cached index — loading …")
        try:
            idx = load_index(repo_id, data_dir)
            progress(1.0, "Loaded from cache")
            return idx, True
        except Exception as exc:  # corrupt cache -> rebuild
            log.warning("cache load failed (%s); rebuilding", exc)
    with acquire(spec, allow_large=params.allow_large, progress=progress) as snap:
        return build_index(snap, params, progress, data_dir), False


# ----------------------------------------------------------------------------- overview
def _readme_excerpt(index: RepoIndex, limit: int = 3500) -> str:
    for f in index.files:
        if f.path.lower().rsplit("/", 1)[-1] in ("readme.md", "readme.rst", "readme.txt", "readme") and f.path.count("/") <= 1:
            try:
                return index.read_file(f.path)[:limit]
            except Exception:
                return ""
    return ""


def offline_overview(index: RepoIndex) -> str:
    st = repo_stats(index)
    stack = detect_tech_stack(index)
    entries = find_entry_points(index)[:6]
    routes = find_routes(index)[:8]
    langs = ", ".join(f"{k} ({v:,} lines)" for k, v in list(st["languages"].items())[:5])
    lines = [f"### {index.name}", "",
             f"**{st['files']:,} files · {st['loc']:,} lines · {st['chunks']:,} indexed chunks** · {langs}", ""]
    if stack:
        lines.append("**Tech stack:** " + ", ".join(stack))
    readme = _readme_excerpt(index, 500).strip()
    if readme:
        lines += ["", "**From the README:**", "> " + " ".join(readme.split())[:420] + " …"]
    if entries:
        lines += ["", "**Entry points**"] + [f"- `{e['file']}` — {e['kind']}" + (f" ({e['note']})" if e["note"] else "") for e in entries]
    if routes:
        lines += ["", "**HTTP routes**"] + [f"- `{r['method']} {r['path']}` — `{r['file']}:{r['line']}`" for r in routes]
    lines += ["", "_Add a Gemini API key for an AI-written overview with architecture and run instructions._"]
    return "\n".join(lines)


def generate_overview(client: GeminiClient, index: RepoIndex) -> str:
    """AI-written overview grounded in README, manifests, folder layout and detected entry points."""
    if not client.available:
        return offline_overview(index)
    deps = detect_dependencies(index)
    entries = find_entry_points(index)[:10]
    routes = find_routes(index)[:12]
    prompt = (
        "Write a concise project overview for a developer new to this repository, using ONLY the facts below. "
        "Use these markdown sections: **Purpose**, **Tech stack**, **Folder structure** (one line per important "
        "folder), **Main entry points**, **How to run it** (only if the facts support it; otherwise say it isn't "
        "documented), **Key things to know**. Be specific and short. Do not invent anything.\n\n"
        f"REPO: {index.name} ({len(index.files)} files, {index.loc} lines)\n"
        f"LANGUAGES: {dict(list(index.languages.items())[:6])}\n"
        f"TECH STACK (detected): {', '.join(detect_tech_stack(index))}\n"
        f"README (excerpt):\n{_readme_excerpt(index)}\n\n"
        f"FOLDERS:\n{folder_summary(index)}\n\n"
        f"DEPENDENCY MANIFESTS: { {k: v[:25] for k, v in deps.items()} }\n"
        f"ENTRY POINTS: {entries}\nROUTES: {routes}\n"
    )
    try:
        return client.generate(prompt, temperature=0.2, max_tokens=1400)
    except LLMError as exc:
        return offline_overview(index) + f"\n\n> ⚠️ AI overview unavailable: {exc}"
