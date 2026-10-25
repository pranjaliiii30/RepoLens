"""Retrieval benchmark: auto-generates questions from the repo's own docstrings and measures how well
vector, BM25 and hybrid(+rerank) retrieval find the right symbol. Gives CV-ready numbers (Recall@k, MRR)."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .models import Chunk, RepoIndex
from .retriever import HybridRetriever, heuristic_rewrite
from .utils import is_test_path, split_identifier

STRATEGIES = {
    "Vector only": dict(mode="vector", rerank=False),
    "BM25 only": dict(mode="bm25", rerank=False),
    "Hybrid (RRF)": dict(mode="hybrid", rerank=False),
    "Hybrid + rerank": dict(mode="hybrid", rerank=True),
}


@dataclass
class EvalCase:
    question: str
    target_id: str
    target_name: str
    target_file: str
    kind: str          # "docstring" | "name"


def _first_sentence(doc: str) -> str:
    doc = re.sub(r"\s+", " ", doc.strip())
    m = re.match(r"(.{20,220}?[.!?])(?:\s|$)", doc)
    return (m.group(1) if m else doc[:200]).strip()


def _strip_identifiers(text: str, name: str) -> str:
    """Remove the symbol's own name so the question can't be answered by trivial string matching."""
    out = re.sub(re.escape(name), "it", text, flags=re.I)
    for part in split_identifier(name):
        if len(part) > 4:
            out = re.sub(rf"\b{re.escape(part)}\w*", "it", out, flags=re.I)
    return out


def generate_cases(index: RepoIndex, n: int = 40, seed: int = 7, include_name_cases: bool = True) -> list[EvalCase]:
    """Questions from docstrings (target = the documented symbol). Falls back to name-derived questions
    for repos without docstrings."""
    rng = random.Random(seed)
    cands: list[Chunk] = [c for c in index.chunks
                          if c.symbol_type in ("function", "method", "class") and c.part == 0
                          and not is_test_path(c.file_path) and c.n_lines >= 4
                          and not c.symbol_name.startswith("__")]
    documented = [c for c in cands if len(c.docstring.strip()) >= 25]
    rng.shuffle(documented)
    cases: list[EvalCase] = []
    for c in documented:
        sent = _strip_identifiers(_first_sentence(c.docstring), c.symbol_name)
        if len(sent.split()) < 4:
            continue
        cases.append(EvalCase(f"Which code {sent[0].lower() + sent[1:]}", c.id, c.qualname, c.file_path, "docstring"))
        if len(cases) >= n:
            return cases
    if include_name_cases and len(cases) < n:
        rest = [c for c in cands if c not in documented and len(split_identifier(c.symbol_name)) >= 2]
        rng.shuffle(rest)
        for c in rest:
            words = " ".join(split_identifier(c.symbol_name))
            cases.append(EvalCase(f"where is the logic to {words}", c.id, c.qualname, c.file_path, "name"))
            if len(cases) >= n:
                break
    return cases


def _rank_of(target_id: str, target_file: str, hits) -> int | None:
    for r, h in enumerate(hits, 1):
        if h.chunk.id == target_id:
            return r
    return None


def run_benchmark(retriever: HybridRetriever, cases: list[EvalCase], ks: tuple[int, ...] = (1, 3, 5, 10),
                  strategies: dict[str, dict] | None = None, progress=None) -> dict:
    strategies = strategies or STRATEGIES
    kmax = max(ks)
    results: dict[str, dict] = {}
    detail: list[dict] = []
    total_steps = len(strategies) * len(cases)
    step = 0
    for name, opts in strategies.items():
        ranks: list[int | None] = []
        times: list[float] = []
        for case in cases:
            plan = heuristic_rewrite(case.question)
            res = retriever.search(case.question, k=kmax, plan=plan, max_per_file=kmax, **opts)
            rank = _rank_of(case.target_id, case.target_file, res.hits)
            ranks.append(rank)
            times.append(res.timings_ms.get("total", 0.0))
            detail.append({"strategy": name, "question": case.question, "target": case.target_name,
                           "file": case.target_file, "rank": rank})
            step += 1
            if progress:
                progress(step / max(1, total_steps), f"{name}: {step}/{total_steps}")
        n = max(1, len(cases))
        metrics = {f"Recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
        metrics["MRR"] = sum((1.0 / r) for r in ranks if r) / n
        metrics["Avg latency (ms)"] = sum(times) / n
        results[name] = metrics
    return {"n_cases": len(cases), "metrics": results, "detail": detail}
