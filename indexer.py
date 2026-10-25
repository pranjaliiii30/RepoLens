"""Embeddings, vector stores, BM25 and the index build / load / cache lifecycle."""
from __future__ import annotations

import heapq
import json
import logging
import math
import shutil
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from . import config
from .chunker import chunk_file
from .ingest import (RepoError, RepoTooLarge, SourceSnapshot, make_repo_id, materialize, scan_files)
from .models import Chunk, FileInfo, RepoIndex
from .utils import tokenize

log = logging.getLogger("repolens.indexer")
ProgressFn = Callable[[float, str], None]


# ============================================================================= embedders
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class EmbedderUnavailable(RepoError):
    pass


class HashingEmbedder:
    """Deterministic, dependency-free feature-hashing embedder (signed hashing trick over identifier-aware
    unigrams + bigrams). Not semantic, but fast, offline and good for lexical/identifier overlap.
    Used as a fallback when sentence-transformers or the model download is unavailable."""

    name = config.OFFLINE_EMBEDDER

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)[:700]
        counts = Counter(tokens)
        for tok, c in counts.items():
            h = zlib.crc32(tok.encode())
            sign = 1.0 if zlib.crc32(tok.encode() + b"|s") & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(c))
        for a, b in zip(tokens, tokens[1:]):
            bg = f"{a} {b}".encode()
            h = zlib.crc32(bg)
            sign = 1.0 if zlib.crc32(bg + b"|s") & 1 else -1.0
            vec[h % self.dim] += 0.5 * sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class STEmbedder:
    """sentence-transformers wrapper (local, free)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, kept lazy

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self._query_prefix = ("Represent this sentence for searching relevant passages: "
                              if model_name.startswith("BAAI/bge") else "")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([self._query_prefix + text])[0]


_EMBEDDER_CACHE: dict[str, Embedder] = {}


def get_embedder(name: str, *, strict: bool = False) -> tuple[Embedder, str]:
    """Return ``(embedder, note)``. With ``strict`` there is no silent fallback (needed when a stored index
    was built with a specific model, because vector dimensions must match)."""
    if name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[name], ""
    if name == config.OFFLINE_EMBEDDER:
        emb: Embedder = HashingEmbedder()
        _EMBEDDER_CACHE[name] = emb
        return emb, ""
    try:
        emb = STEmbedder(name)
        _EMBEDDER_CACHE[name] = emb
        return emb, ""
    except Exception as exc:
        reason = f"{exc.__class__.__name__}: {str(exc)[:160]}"
        if strict:
            raise EmbedderUnavailable(
                f"This index was built with '{name}', which can't be loaded here ({reason}). "
                "Install sentence-transformers / check your connection, or re-index with the offline embedder."
            ) from exc
        log.warning("embedding model %s unavailable (%s); using offline hashing embedder", name, reason)
        return get_embedder(config.OFFLINE_EMBEDDER)[0], (
            f"'{name}' could not be loaded ({reason}). Using the built-in offline embedder instead — "
            "retrieval relies more on keyword matching. Install sentence-transformers for semantic search."
        )


# ============================================================================= vector stores
class NumpyStore:
    """Exact cosine search over a numpy matrix. Used when chromadb isn't installed."""

    kind = "numpy"

    def __init__(self, directory: Path):
        self.dir = directory
        self.ids: list[str] = []
        self.mat = np.zeros((0, 1), dtype=np.float32)
        self._pos: dict[str, int] = {}

    def add(self, ids: list[str], vectors: np.ndarray, metas: list[dict] | None = None) -> None:
        self.ids = list(ids)
        self.mat = np.asarray(vectors, dtype=np.float32)
        self._pos = {cid: i for i, cid in enumerate(self.ids)}

    def count(self) -> int:
        return len(self.ids)

    def query(self, vec: np.ndarray, n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.mat @ np.asarray(vec, dtype=np.float32)
        if allowed is not None:
            mask = np.full(len(self.ids), -np.inf, dtype=np.float32)
            idx = [self._pos[a] for a in allowed if a in self._pos]
            if not idx:
                return []
            mask[idx] = sims[idx]
            sims = mask
        n = min(n, len(sims))
        top = np.argpartition(-sims, n - 1)[:n] if n < len(sims) else np.arange(len(sims))
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top if np.isfinite(sims[i])]

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", self.mat)
        (self.dir / "vector_ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    def load(self) -> None:
        self.mat = np.load(self.dir / "vectors.npy")
        self.ids = json.loads((self.dir / "vector_ids.json").read_text(encoding="utf-8"))
        self._pos = {cid: i for i, cid in enumerate(self.ids)}

    def delete(self) -> None:
        for name in ("vectors.npy", "vector_ids.json"):
            (self.dir / name).unlink(missing_ok=True)


class ChromaStore:
    """ChromaDB persistent collection (cosine space). One collection per indexed repo."""

    kind = "chroma"

    def __init__(self, repo_id: str, chroma_dir: Path):
        import chromadb
        from chromadb.config import Settings

        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.name = f"rl-{repo_id}"[:63]
        self.client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def add(self, ids: list[str], vectors: np.ndarray, metas: list[dict] | None = None) -> None:
        step = 1000
        for i in range(0, len(ids), step):
            batch_meta = None
            if metas:
                batch_meta = [{k: v for k, v in m.items() if v != ""} or {"file_path": "?"} for m in metas[i:i + step]]
            self.col.add(ids=ids[i:i + step], embeddings=vectors[i:i + step].astype(float).tolist(),
                         metadatas=batch_meta)

    def count(self) -> int:
        return int(self.col.count())

    def query(self, vec: np.ndarray, n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        total = self.count()
        if total == 0:
            return []
        fetch = min(total, n if allowed is None else max(n * 8, 200))
        while True:
            res = self.col.query(query_embeddings=[np.asarray(vec, dtype=float).tolist()], n_results=fetch,
                                 include=["distances"])
            pairs = [(cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0])]
            if allowed is not None:
                pairs = [p for p in pairs if p[0] in allowed]
            if len(pairs) >= n or fetch >= total:
                return pairs[:n]
            fetch = min(total, fetch * 4)

    def save(self) -> None:  # persistence is automatic
        return None

    def load(self) -> None:
        return None

    def delete(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass


def chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401

        return True
    except Exception:
        return False


def open_store(repo_id: str, repo_dir: Path, data_dir: Path, backend: str):
    if backend == "chroma":
        if not chroma_available():
            raise EmbedderUnavailable("This index uses ChromaDB, which isn't installed. `pip install chromadb`.")
        return ChromaStore(repo_id, data_dir / "chroma")
    return NumpyStore(repo_dir)


# ============================================================================= BM25
class BM25Index:
    """Okapi BM25 with an inverted index (pure Python; no extra dependency)."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        postings: dict[str, list[tuple[int, int]]] = {}
        for di, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((di, tf))
        self.postings = postings
        self.idf = {t: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}

    def search(self, query_tokens: list[str], n: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in set(query_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for di, tf in plist:
                if allowed is not None and di not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * float(self.doc_len[di]) / (self.avgdl or 1.0))
                scores[di] = scores.get(di, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return heapq.nlargest(n, scores.items(), key=lambda kv: kv[1])


def build_bm25(chunks: list[Chunk]) -> BM25Index:
    return BM25Index([tokenize(c.bm25_text()) for c in chunks])


# ============================================================================= index lifecycle
@dataclass
class IndexParams:
    embedder: str = config.DEFAULT_EMBEDDER
    extra_ignores: list[str] = field(default_factory=list)
    max_file_bytes: int = config.MAX_FILE_BYTES
    use_tree_sitter: bool = True
    backend: str = "chroma"          # preferred vector backend ("chroma" falls back to numpy)
    allow_large: bool = False
    max_files: int = config.MAX_FILES


@dataclass
class SearchStack:
    index: RepoIndex
    embedder: Embedder
    store: object
    bm25: BM25Index
    pos: dict[str, int]


def repos_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or config.DATA_DIR) / "repos"


def find_cached(repo_id: str, data_dir: Path | None = None) -> bool:
    return (repos_dir(data_dir) / repo_id / "manifest.json").exists()


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def build_index(snap: SourceSnapshot, params: IndexParams, progress: ProgressFn | None = None,
                data_dir: Path | None = None) -> RepoIndex:
    progress = progress or (lambda f, m: None)
    data_dir = (data_dir or config.DATA_DIR).resolve()

    progress(0.10, "Scanning files …")
    scan = scan_files(snap.root, params.extra_ignores, params.max_file_bytes)
    if len(scan.files) > params.max_files and not params.allow_large:
        raise RepoTooLarge(f"This repository has {len(scan.files):,} indexable files "
                           f"(soft limit {params.max_files:,}).", n_files=len(scan.files))
    if not scan.files:
        raise RepoError("No indexable text files were found in this repository.")

    embedder, note = get_embedder(params.embedder)
    repo_id = make_repo_id(snap.name, snap.source, snap.key_branch or snap.branch, snap.commit, embedder.name)
    base = repos_dir(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    final_dir, tmp_dir = base / repo_id, base / f"{repo_id}.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    backend = "chroma" if params.backend == "chroma" and chroma_available() else "numpy"
    store = None
    try:
        # ---- parse ------------------------------------------------------------------------------
        warnings: list[str] = []
        parsers: Counter = Counter()
        chunks: list[Chunk] = []
        total = len(scan.files)
        for i, info in enumerate(scan.files):
            text = (snap.root / info.path).read_bytes().decode("utf-8", errors="replace")
            meta: dict = {}
            chunks.extend(chunk_file(info.path, text, info.language, warnings=warnings, info=meta,
                                     use_tree_sitter=params.use_tree_sitter))
            parsers[meta.get("parser", "?")] += 1
            if i % 10 == 0 or i == total - 1:
                progress(0.15 + 0.25 * (i + 1) / total, f"Parsing {i + 1}/{total} files …")
        if not chunks:
            raise RepoError("Files were found but nothing could be chunked.")

        # ---- embed ------------------------------------------------------------------------------
        texts = [c.embed_text() for c in chunks]
        vectors: list[np.ndarray] = []
        batch = 64
        for s in range(0, len(texts), batch):
            vectors.append(embedder.encode(texts[s:s + batch]))
            progress(0.40 + 0.50 * min(1.0, (s + batch) / len(texts)), f"Embedding {min(s + batch, len(texts))}/{len(texts)} chunks …")
        matrix = np.vstack(vectors)

        # ---- store ------------------------------------------------------------------------------
        progress(0.92, "Writing vector store …")
        store = open_store(repo_id, tmp_dir, data_dir, backend)
        if backend == "chroma":
            store.reset()
        store.add([c.id for c in chunks], matrix, [c.store_metadata() for c in chunks])
        store.save()

        progress(0.96, "Saving source snapshot …")
        materialize(snap.root, scan.files, tmp_dir / "source")
        _write_chunks(tmp_dir / "chunks.jsonl", chunks)
        index = RepoIndex(
            repo_id=repo_id, name=snap.name, source=snap.source, branch=snap.branch, commit=snap.commit,
            embedder=embedder.name, vector_backend=backend,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            root=final_dir / "source", files=scan.files, chunks=chunks,
            warnings=warnings, skipped=scan.skipped, embedder_note=note,
        )
        manifest = index.manifest()
        manifest["parsers"] = dict(parsers)
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        shutil.rmtree(final_dir, ignore_errors=True)
        tmp_dir.rename(final_dir)
        if backend == "numpy":
            store.dir = final_dir
        progress(1.0, "Done")
        return index
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if store is not None and backend == "chroma":
            store.delete()
        raise


def load_index(repo_id: str, data_dir: Path | None = None) -> RepoIndex:
    data_dir = (data_dir or config.DATA_DIR).resolve()
    repo_dir = repos_dir(data_dir) / repo_id
    manifest_path = repo_dir / "manifest.json"
    if not manifest_path.exists():
        raise RepoError(f"Index '{repo_id}' was not found. Re-index the repository.")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(json.loads(line)) for line in
              (repo_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return RepoIndex(
        repo_id=m["repo_id"], name=m["name"], source=m["source"], branch=m.get("branch", ""),
        commit=m.get("commit", ""), embedder=m["embedder"], vector_backend=m.get("vector_backend", "numpy"),
        created_at=m.get("created_at", ""), root=repo_dir / "source",
        files=[FileInfo(**f) for f in m.get("files", [])], chunks=chunks,
        warnings=m.get("warnings", []), skipped=m.get("skipped", {}), embedder_note=m.get("embedder_note", ""),
    )


def open_stack(index: RepoIndex, data_dir: Path | None = None) -> SearchStack:
    data_dir = (data_dir or config.DATA_DIR).resolve()
    embedder, _ = get_embedder(index.embedder, strict=True)
    store = open_store(index.repo_id, repos_dir(data_dir) / index.repo_id, data_dir, index.vector_backend)
    store.load()
    return SearchStack(index=index, embedder=embedder, store=store, bm25=build_bm25(index.chunks),
                       pos={c.id: i for i, c in enumerate(index.chunks)})


def list_indexes(data_dir: Path | None = None) -> list[dict]:
    base = repos_dir(data_dir)
    out: list[dict] = []
    if not base.exists():
        return out
    for d in base.iterdir():
        mf = d / "manifest.json"
        if d.suffix == ".tmp" or not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            out.append({"repo_id": m["repo_id"], "name": m["name"], "branch": m.get("branch", ""),
                        "embedder": m.get("embedder", ""), "n_files": len(m.get("files", [])),
                        "n_chunks": m.get("n_chunks", 0), "created_at": m.get("created_at", ""),
                        "commit": (m.get("commit") or "")[:7]})
        except Exception:
            continue
    return sorted(out, key=lambda r: r["created_at"], reverse=True)


def delete_index(repo_id: str, data_dir: Path | None = None) -> None:
    data_dir = (data_dir or config.DATA_DIR).resolve()
    repo_dir = repos_dir(data_dir) / repo_id
    try:
        m = json.loads((repo_dir / "manifest.json").read_text(encoding="utf-8"))
        if m.get("vector_backend") == "chroma" and chroma_available():
            ChromaStore(repo_id, data_dir / "chroma").delete()
    except Exception:
        pass
    shutil.rmtree(repo_dir, ignore_errors=True)
