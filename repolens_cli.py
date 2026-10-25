"""Terminal interface: index a repo, ask questions, run the retrieval benchmark.

  python repolens_cli.py index https://github.com/psf/requests
  python repolens_cli.py ask   psf-requests "where are redirects handled?"
  python repolens_cli.py bench psf-requests
  python repolens_cli.py list
"""
from __future__ import annotations

import argparse
import sys

from core import config
from core.evaluation import generate_cases, run_benchmark
from core.indexer import IndexParams, list_indexes, load_index, open_stack
from core.ingest import RepoError, SourceSpec
from core.llm import SYSTEM_PROMPT, GeminiClient, LLMError, build_prompt, extractive_answer
from core.render import benchmark_summary
from core.retriever import HybridRetriever, heuristic_rewrite
from core.service import index_or_load


def _resolve(prefix: str) -> str:
    rows = [r for r in list_indexes() if r["repo_id"].startswith(prefix) or r["name"].replace("/", "-") == prefix]
    if not rows:
        sys.exit(f"No index matches '{prefix}'. Run `list` to see indexed repos.")
    return rows[0]["repo_id"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="repolens")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="index a GitHub URL, ZIP file or (with REPOLENS_ALLOW_LOCAL=1) a folder")
    p.add_argument("source")
    p.add_argument("--branch")
    p.add_argument("--embedder", default=config.DEFAULT_EMBEDDER)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("repo")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    p = sub.add_parser("bench", help="retrieval benchmark")
    p.add_argument("repo")
    p.add_argument("-n", type=int, default=50)
    sub.add_parser("list", help="list indexed repos")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "list":
            for r in list_indexes():
                print(f"{r['repo_id']:34} {r['n_files']:>5} files {r['n_chunks']:>6} chunks  {r['embedder']}")
            return 0
        if a.cmd == "index":
            src = a.source
            if src.endswith(".zip"):
                spec = SourceSpec(kind="zip", value=src, zip_bytes=open(src, "rb").read())
            elif "/" in src and not src.startswith(("http", "git@")) and __import__("os").path.isdir(src):
                spec = SourceSpec(kind="local", value=src)
            else:
                spec = SourceSpec(kind="url", value=src, branch=a.branch)
            idx, cached = index_or_load(spec, IndexParams(embedder=a.embedder), force=a.force,
                                        progress=lambda f, m: print(f"\r{f:4.0%} {m:60}", end="", file=sys.stderr))
            print(f"\n{'Loaded from cache' if cached else 'Indexed'}: {idx.repo_id} "
                  f"({len(idx.files)} files, {len(idx.chunks)} chunks)")
            return 0
        index = load_index(_resolve(a.repo))
        retr = HybridRetriever(open_stack(index))
        if a.cmd == "ask":
            res = retr.search(a.question, k=a.k, plan=heuristic_rewrite(a.question))
            client = GeminiClient()
            if client.available:
                try:
                    print(client.generate(build_prompt(a.question, res.hits, None, index.name), system=SYSTEM_PROMPT))
                    return 0
                except LLMError as exc:
                    print(f"[{exc}]\n", file=sys.stderr)
            print(extractive_answer(a.question, res.hits))
        elif a.cmd == "bench":
            result = run_benchmark(retr, generate_cases(index, n=a.n))
            for name, m in result["metrics"].items():
                print(f"{name:18} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
            print("\n" + benchmark_summary(result))
    except (RepoError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
