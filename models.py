"""Dataclasses shared across the engine."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import MAX_EMBED_CHARS

SYMBOL_TYPES = ("function", "method", "class", "module", "doc", "config", "block")
CODE_SYMBOL_TYPES = ("function", "method", "class")


def make_chunk_id(path: str, start: int, end: int, name: str, part: int) -> str:
    raw = f"{path}:{start}:{end}:{name}:{part}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    id: str
    file_path: str
    language: str
    symbol_name: str
    symbol_type: str            # one of SYMBOL_TYPES
    start_line: int             # 1-based, inclusive
    end_line: int               # 1-based, inclusive
    code: str
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    parent: str = ""            # enclosing class, dotted for nested classes
    signature: str = ""
    part: int = 0               # sub-chunk index for very long symbols
    parts: int = 1
    complexity: int = 0         # cyclomatic complexity estimate (functions/methods)
    calls: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)  # method names, for class summaries

    # -- presentation ---------------------------------------------------------
    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.symbol_name}" if self.parent else self.symbol_name

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1

    def header(self) -> str:
        """Short context line prepended to the text that gets embedded."""
        bits = [f"File: {self.file_path}"]
        if self.symbol_type in ("function", "method"):
            if self.parent:
                bits.append(f"Class: {self.parent}")
            bits.append(f"{'Method' if self.symbol_type == 'method' else 'Function'}: {self.symbol_name}")
        elif self.symbol_type == "class":
            bits.append(f"Class: {self.qualname}")
        elif self.symbol_type == "module":
            bits.append("Module-level code")
        elif self.symbol_type == "doc":
            bits.append(f"Section: {self.symbol_name}")
        if self.parts > 1:
            bits.append(f"Part {self.part + 1}/{self.parts}")
        return " | ".join(bits)

    def embed_text(self) -> str:
        extra = ""
        if self.members:
            extra = "\nMembers: " + ", ".join(self.members[:40])
        return f"{self.header()}{extra}\n{self.code[:MAX_EMBED_CHARS]}"

    def bm25_text(self) -> str:
        extra = " ".join(self.members[:60])
        return f"{self.header()} {self.signature} {extra} {self.code}"

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def store_metadata(self) -> dict[str, Any]:
        """Scalar-only metadata (Chroma requires str/int/float/bool values)."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "parent": self.parent,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class SearchHit:
    chunk: Chunk
    score: float                       # final fused score used for ordering
    vec_score: float | None = None     # cosine similarity
    bm25_score: float | None = None
    vec_rank: int | None = None        # 1-based rank in the vector list
    bm25_rank: int | None = None
    cross_score: float | None = None

    def to_ref(self) -> dict[str, Any]:
        return {
            "id": self.chunk.id,
            "score": self.score,
            "vec_score": self.vec_score,
            "bm25_score": self.bm25_score,
            "vec_rank": self.vec_rank,
            "bm25_rank": self.bm25_rank,
            "cross_score": self.cross_score,
        }


@dataclass
class FileInfo:
    path: str
    language: str
    size: int
    lines: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepoIndex:
    repo_id: str
    name: str
    source: str
    branch: str
    commit: str
    embedder: str
    vector_backend: str
    created_at: str
    root: Path                          # directory holding the filtered source copy
    files: list[FileInfo] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    embedder_note: str = ""

    def __post_init__(self) -> None:
        self._by_id: dict[str, Chunk] | None = None

    @property
    def by_id(self) -> dict[str, Chunk]:
        if self._by_id is None or len(self._by_id) != len(self.chunks):
            self._by_id = {c.id: c for c in self.chunks}
        return self._by_id

    @property
    def loc(self) -> int:
        return sum(f.lines for f in self.files)

    @property
    def languages(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.files:
            out[f.language] = out.get(f.language, 0) + f.lines
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def read_file(self, rel_path: str) -> str:
        target = (self.root / rel_path).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            raise ValueError("path escapes repository root")
        return target.read_text(encoding="utf-8", errors="replace")

    def chunks_for(self, rel_path: str) -> list[Chunk]:
        return sorted((c for c in self.chunks if c.file_path == rel_path), key=lambda c: c.start_line)

    def manifest(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "name": self.name,
            "source": self.source,
            "branch": self.branch,
            "commit": self.commit,
            "embedder": self.embedder,
            "embedder_note": self.embedder_note,
            "vector_backend": self.vector_backend,
            "created_at": self.created_at,
            "files": [f.to_dict() for f in self.files],
            "warnings": self.warnings[:200],
            "skipped": self.skipped,
            "n_chunks": len(self.chunks),
        }
