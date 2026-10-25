"""Hybrid retrieval: vector + BM25, Reciprocal Rank Fusion, heuristic and optional cross-encoder reranking."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from . import config
from .indexer import SearchStack
from .models import Chunk, SearchHit
from .utils import is_test_path, split_identifier, tokenize

log = logging.getLogger("repolens.retriever")


# ----------------------------------------------------------------------------- filters
@dataclass(frozen=True)
class Filters:
    languages: tuple[str, ...] = ()
    folder: str = ""
    file: str = ""
    types: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.languages or self.folder or self.file or self.types)

    def allows(self, c: Chunk) -> bool:
        if self.languages and c.language not in self.languages:
            return False
        if self.file and c.file_path != self.file:
            return False
        if self.folder:
            folder = self.folder.strip("/")
            if not (c.file_path == folder or c.file_path.startswith(folder + "/")):
                return False
        if self.types and c.symbol_type not in self.types:
            return False
        return True


# ----------------------------------------------------------------------------- fusion
def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


BOOST_SCALE = 0.05  # tuned on the built-in benchmark: larger values pushed correct chunks down


def heuristic_boost(chunk: Chunk, q_tokens: set[str], q_lower: str) -> float:
    """Tie-breaker adjustments on top of RRF. RRF scores for adjacent ranks differ by ~0.0005, so the
    boosts are deliberately tiny: they reorder near-ties, they never override the fused ranking.
    (An earlier, 20x larger version measurably *hurt* MRR on the Retrieval Lab benchmark.)"""
    boost = 0.0
    name_parts = set(split_identifier(chunk.symbol_name))
    if name_parts and chunk.symbol_type in ("function", "method", "class"):
        overlap = len(name_parts & q_tokens) / len(name_parts)
        boost += 0.02 * overlap
        if len(chunk.symbol_name) > 3 and chunk.symbol_name.lower() in q_lower:
            boost += 0.01
    path_parts: set[str] = set()
    for seg in chunk.file_path.split("/"):
        path_parts.update(split_identifier(seg.rsplit(".", 1)[0]))
    boost += 0.004 * min(2, len(path_parts & q_tokens))
    if is_test_path(chunk.file_path) and "test" not in q_tokens and "tests" not in q_tokens:
        boost -= 0.006
    return boost * BOOST_SCALE


# ----------------------------------------------------------------------------- query planning
SYNONYMS = {
    "auth": "login authenticate authentication token jwt session password credentials oauth permission user",
    "authentication": "login authenticate token jwt session password credentials oauth",
    "authorization": "permission role access policy acl guard",
    "login": "authenticate session token password credentials",
    "database": "db sql query connection session model orm migrate cursor transaction",
    "db": "database sql query connection session orm cursor",
    "error": "exception raise try except catch handler logging failure",
    "errors": "exception raise try except catch handler logging failure",
    "exception": "error raise try except catch handler",
    "handling": "handler handle catch except",
    "test": "tests assert pytest unittest mock fixture spec",
    "tests": "test assert pytest unittest mock fixture spec",
    "route": "router endpoint handler request response path url",
    "routes": "router endpoint handler request response path url",
    "endpoint": "route router handler request response api path",
    "api": "route endpoint handler request response rest http",
    "config": "configuration settings env environ options setup",
    "configuration": "config settings env environ options",
    "settings": "config configuration env options",
    "cache": "caching redis memoize ttl lru",
    "log": "logger logging logs",
    "logging": "logger log handler",
    "async": "await asyncio coroutine task",
    "parse": "parser parsing tokenize lexer ast",
    "render": "template view html draw display",
    "request": "http client session response url fetch",
    "http": "request response client session url fetch",
    "retry": "backoff attempts timeout",
    "security": "auth token secret encrypt hash permission",
    "build": "compile bundle pipeline make setup",
    "main": "entry run start app cli",
    "entry": "main run start app cli",
    "start": "run main launch serve",
    "middleware": "request response handler chain next",
    "cli": "argparse click command argv main",
    "model": "schema entity orm table class",
    "validate": "validation validator schema check verify",
}
_PRONOUNS = re.compile(r"\b(it|its|that|this|those|these|them|there|above|same)\b", re.I)


@dataclass
class QueryPlan:
    original: str
    keywords: str
    vector_query: str
    bm25_query: str
    source: str = "heuristic"      # "heuristic" | "llm"


def heuristic_rewrite(question: str, previous_symbols: list[str] | None = None) -> QueryPlan:
    toks = tokenize(question)
    expanded: list[str] = []
    for t in toks:
        expanded.append(t)
        syn = SYNONYMS.get(t)
        if syn:
            expanded.extend(syn.split())
    if previous_symbols and (len(question.split()) <= 9 and _PRONOUNS.search(question)):
        expanded.extend(previous_symbols[:3])
    seen: set[str] = set()
    uniq = [t for t in expanded if not (t in seen or seen.add(t))]
    keywords = " ".join(uniq[:40])
    return QueryPlan(question, keywords, f"{question}\n{keywords}", f"{keywords} {question}", "heuristic")


def llm_rewrite(question: str, history_text: str, generate: Callable[[str], str],
                previous_symbols: list[str] | None = None) -> QueryPlan:
    """Ask the LLM for identifier-rich search keywords; falls back to the heuristic on any problem."""
    prompt = (
        "Rewrite the developer question below as a short keyword query for searching a code index.\n"
        "Include likely function/class/file names and synonyms (snake_case and camelCase variants are fine).\n"
        "Resolve pronouns using the conversation. Output ONLY the keywords, one line, max 25 words.\n\n"
        f"Conversation so far:\n{history_text or '(none)'}\n\nQuestion: {question}\nKeywords:"
    )
    try:
        raw = generate(prompt).strip().splitlines()[0].strip().strip("`'\"")
        raw = re.sub(r"^keywords?:\s*", "", raw, flags=re.I)
        if not raw or len(raw) > 300:
            raise ValueError("unusable rewrite")
        base = heuristic_rewrite(question, previous_symbols)
        return QueryPlan(question, raw, f"{question}\n{raw}", f"{raw} {base.keywords}", "llm")
    except Exception as exc:
        log.info("LLM rewrite failed (%s); using heuristic", exc)
        return heuristic_rewrite(question, previous_symbols)


# ----------------------------------------------------------------------------- retriever
@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    plan: QueryPlan | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, stack: SearchStack):
        self.stack = stack
        self.index = stack.index
        self._cross = None
        self._cross_failed = False

    # -- cross encoder ---------------------------------------------------------------------
    def _cross_encoder(self):
        if self._cross is None and not self._cross_failed:
            try:
                from sentence_transformers import CrossEncoder

                self._cross = CrossEncoder(config.CROSS_ENCODER_MODEL)
            except Exception as exc:
                log.warning("cross-encoder unavailable: %s", exc)
                self._cross_failed = True
        return self._cross

    # -- main entry --------------------------------------------------------------------------
    def search(self, query: str, *, k: int = config.DEFAULT_TOP_K, mode: str = "hybrid",
               filters: Filters | None = None, rerank: bool = True, cross_encoder: bool = False,
               max_per_file: int = config.MAX_PER_FILE, plan: QueryPlan | None = None) -> RetrievalResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        notes: list[str] = []
        stack = self.stack
        chunks = self.index.chunks
        vector_query = plan.vector_query if plan else query
        bm25_query = plan.bm25_query if plan else query

        allowed_ids: set[str] | None = None
        allowed_idx: set[int] | None = None
        if filters and filters.active:
            allowed_idx = {i for i, c in enumerate(chunks) if filters.allows(c)}
            allowed_ids = {chunks[i].id for i in allowed_idx}
            if not allowed_idx:
                return RetrievalResult([], plan, {"total": 0.0}, ["No chunks match the active filters."])

        pool = max(k * 4, 40)
        vec_list: list[tuple[str, float]] = []
        bm_list: list[tuple[str, float]] = []
        if mode in ("hybrid", "vector"):
            t = time.perf_counter()
            qv = stack.embedder.encode_query(vector_query)
            vec_list = stack.store.query(qv, pool, allowed_ids)
            timings["vector"] = (time.perf_counter() - t) * 1000
        if mode in ("hybrid", "bm25"):
            t = time.perf_counter()
            raw = stack.bm25.search(tokenize(bm25_query), pool, allowed_idx)
            bm_list = [(chunks[i].id, s) for i, s in raw]
            timings["bm25"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        vec_rank = {cid: r for r, (cid, _) in enumerate(vec_list, 1)}
        bm_rank = {cid: r for r, (cid, _) in enumerate(bm_list, 1)}
        vec_score = dict(vec_list)
        bm_score = dict(bm_list)
        fused = rrf_fuse([[cid for cid, _ in vec_list], [cid for cid, _ in bm_list]])
        by_id = self.index.by_id

        q_tokens = set(tokenize(query))
        q_lower = query.lower()
        scored: list[tuple[str, float]] = []
        for cid, s in fused.items():
            if cid not in by_id:
                continue
            if rerank:
                s += heuristic_boost(by_id[cid], q_tokens, q_lower)
            scored.append((cid, s))
        scored.sort(key=lambda kv: -kv[1])
        timings["fuse"] = (time.perf_counter() - t) * 1000

        cross_scores: dict[str, float] = {}
        if cross_encoder and scored:
            ce = self._cross_encoder()
            if ce is None:
                notes.append("Cross-encoder reranking is unavailable (sentence-transformers not installed).")
            else:
                t = time.perf_counter()
                top = scored[:30]
                pairs = [(query, by_id[cid].embed_text()[:1200]) for cid, _ in top]
                raw_scores = [float(x) for x in ce.predict(pairs)]
                lo, hi = min(raw_scores), max(raw_scores)
                span = (hi - lo) or 1.0
                s_hi = max(s for _, s in top)
                s_lo = min(s for _, s in top)
                s_span = (s_hi - s_lo) or 1.0
                blended = []
                for (cid, s), cs in zip(top, raw_scores):
                    cross_scores[cid] = cs
                    blended.append((cid, 0.5 * (s - s_lo) / s_span + 0.5 * (cs - lo) / span))
                blended.sort(key=lambda kv: -kv[1])
                scored = blended + scored[30:]
                timings["cross"] = (time.perf_counter() - t) * 1000

        hits: list[SearchHit] = []
        per_file: dict[str, int] = {}
        for cid, s in scored:
            c = by_id[cid]
            if per_file.get(c.file_path, 0) >= max_per_file:
                continue
            per_file[c.file_path] = per_file.get(c.file_path, 0) + 1
            hits.append(SearchHit(c, s, vec_score.get(cid), bm_score.get(cid), vec_rank.get(cid),
                                  bm_rank.get(cid), cross_scores.get(cid)))
            if len(hits) >= k:
                break
        timings["total"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(hits, plan, timings, notes)
