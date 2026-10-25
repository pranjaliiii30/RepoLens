"""Indexing lifecycle, BM25, vector stores, RRF fusion, filters and hybrid search quality."""
from __future__ import annotations

import numpy as np
import pytest

from core import config
from core.indexer import (BM25Index, HashingEmbedder, IndexParams, NumpyStore, build_bm25, delete_index,
                          find_cached, list_indexes, load_index, open_stack)
from core.ingest import RepoError, SourceSpec
from core.retriever import (Filters, HybridRetriever, heuristic_boost, heuristic_rewrite, llm_rewrite, rrf_fuse)
from core.service import index_or_load
from core.utils import is_test_path, split_identifier, tokenize


# ----------------------------------------------------------------------------- text utils
def test_split_identifier_and_tokenize_handle_snake_and_camel_case():
    assert split_identifier("verifyToken") == ["verify", "token"]
    assert split_identifier("verify_token") == ["verify", "token"]
    assert split_identifier("HTTPServerError") == ["http", "server", "error"]
    toks = tokenize("def verifyToken(self, the_user):")
    assert "verifytoken" in toks and "verify" in toks and "token" in toks and "user" in toks
    assert "def" not in toks and "self" not in toks                      # stopwords removed


def test_is_test_path():
    for p in ("tests/test_a.py", "src/__tests__/a.js", "a_test.go", "app/foo.spec.ts", "FooTest.java"):
        assert is_test_path(p), p
    for p in ("src/app.py", "latest/notes.py", "contest.py"):
        assert not is_test_path(p), p


# ----------------------------------------------------------------------------- fusion
def test_rrf_math_and_consensus_wins():
    scores = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert max(scores, key=scores.get) == "b"                           # appears high in both lists
    assert set(scores) == {"a", "b", "c", "d"}


def test_rrf_empty_lists():
    assert rrf_fuse([[], []]) == {}


def test_heuristic_boost_is_tiny_relative_to_rrf_rank_gaps():
    """Regression: oversized boosts once pushed correct answers down (MRR 0.92 -> 0.67)."""
    from core.models import Chunk

    c = Chunk("i", "app/auth.py", "python", "verify_token", "function", 1, 3, "x")
    boost = heuristic_boost(c, {"verify", "token", "auth"}, "verify token auth")
    assert 0 < boost < 0.005                                            # less than ~10 RRF ranks at the top


# ----------------------------------------------------------------------------- BM25 / stores
def test_bm25_ranks_exact_terms_and_respects_allowed_set():
    docs = [tokenize(t) for t in ("hash password md5 digest", "database connection sqlite", "jwt token verify decode")]
    bm = BM25Index(docs)
    assert bm.search(tokenize("verify jwt token"), 3)[0][0] == 2
    assert bm.search(tokenize("password"), 3)[0][0] == 0
    assert bm.search(tokenize("password"), 3, allowed={1, 2}) == []
    assert bm.search(["zzz"], 3) == []


def test_hashing_embedder_is_deterministic_normalised_and_similarity_aware():
    e = HashingEmbedder()
    a, b = e.encode(["hash the password with md5", "hash password md5"])
    c = e.encode(["render html template"])[0]
    assert np.allclose(a, e.encode(["hash the password with md5"])[0])
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert float(a @ b) > float(a @ c)


def test_numpy_store_query_filter_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(20, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"id{i}" for i in range(20)]
    s = NumpyStore(tmp_path)
    s.add(ids, vecs)
    s.save()
    top = s.query(vecs[3], 3)
    assert top[0][0] == "id3" and top[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [i for i, _ in s.query(vecs[3], 5, allowed={"id7", "id9"})] and \
        {i for i, _ in s.query(vecs[3], 5, allowed={"id7", "id9"})} <= {"id7", "id9"}
    assert s.query(vecs[3], 5, allowed={"nope"}) == []
    s2 = NumpyStore(tmp_path)
    s2.load()
    assert s2.query(vecs[3], 1)[0][0] == "id3"


# ----------------------------------------------------------------------------- lifecycle
def test_index_build_load_roundtrip(sample_index, data_dir):
    idx = sample_index
    assert idx.files and idx.chunks and idx.embedder == config.OFFLINE_EMBEDDER
    loaded = load_index(idx.repo_id, data_dir)
    assert [c.id for c in loaded.chunks] == [c.id for c in idx.chunks]
    assert loaded.read_file("app/auth.py").startswith("import hashlib")
    assert find_cached(idx.repo_id, data_dir) and list_indexes(data_dir)[0]["repo_id"] == idx.repo_id
    assert not list((data_dir / "repos").glob("*.tmp"))                 # atomic build leaves no temp dir


def test_read_file_cannot_escape_repo_root(sample_index):
    with pytest.raises(ValueError):
        sample_index.read_file("../../etc/passwd")


def test_cache_hit_on_same_commit_and_forced_rebuild(sample_dir, data_dir):
    config.ALLOW_LOCAL_PATHS = True
    spec, params = SourceSpec(kind="local", value=str(sample_dir)), IndexParams(embedder=config.OFFLINE_EMBEDDER)
    a, cached_a = index_or_load(spec, params, data_dir=data_dir)
    b, cached_b = index_or_load(spec, params, data_dir=data_dir)
    c, cached_c = index_or_load(spec, params, force=True, data_dir=data_dir)
    assert (cached_a, cached_b, cached_c) == (False, True, False)
    assert a.repo_id == b.repo_id == c.repo_id


def test_changed_files_produce_a_new_index_id(sample_dir, data_dir):
    config.ALLOW_LOCAL_PATHS = True
    spec, params = SourceSpec(kind="local", value=str(sample_dir)), IndexParams(embedder=config.OFFLINE_EMBEDDER)
    a, _ = index_or_load(spec, params, data_dir=data_dir)
    (sample_dir / "app" / "extra.py").write_text("def added():\n    return 1\n")
    b, cached = index_or_load(spec, params, data_dir=data_dir)
    assert not cached and a.repo_id != b.repo_id and any(c.symbol_name == "added" for c in b.chunks)


def test_delete_index_removes_everything(sample_index, data_dir):
    delete_index(sample_index.repo_id, data_dir)
    assert not find_cached(sample_index.repo_id, data_dir) and list_indexes(data_dir) == []


def test_empty_project_is_a_clean_error(tmp_path, data_dir):
    config.ALLOW_LOCAL_PATHS = True
    (tmp_path / "empty").mkdir()
    with pytest.raises(RepoError):
        index_or_load(SourceSpec(kind="local", value=str(tmp_path / "empty")),
                      IndexParams(embedder=config.OFFLINE_EMBEDDER), data_dir=data_dir)


# ----------------------------------------------------------------------------- search quality
def _top(retr, q, **kw):
    return [h.chunk.qualname for h in retr.search(q, k=kw.pop("k", 5), plan=heuristic_rewrite(q), **kw).hits]


def test_hybrid_search_finds_the_right_symbols(sample_stack):
    r = HybridRetriever(sample_stack)
    assert _top(r, "where is the password hashed")[0] == "hash_password"
    assert _top(r, "validate the jwt bearer token")[0] == "verify_token"
    assert "query_users" in _top(r, "run the SQL query that lists every user")[:2]
    assert "login" in _top(r, "authenticate a user and return a session token")[:3]


def test_search_modes_all_work_and_scores_are_populated(sample_stack):
    r = HybridRetriever(sample_stack)
    for mode in ("vector", "bm25", "hybrid"):
        res = r.search("hash password", k=3, mode=mode)
        assert res.hits and "total" in res.timings_ms
    h = r.search("hash password", k=3, mode="hybrid").hits[0]
    assert h.vec_rank is not None and h.bm25_rank is not None and h.vec_score is not None


def test_filters_restrict_results(sample_stack):
    r = HybridRetriever(sample_stack)
    js = r.search("check request", k=10, filters=Filters(languages=("javascript",))).hits
    assert js and all(h.chunk.language == "javascript" for h in js)
    folder = r.search("user", k=10, filters=Filters(folder="app")).hits
    assert folder and all(h.chunk.file_path.startswith("app/") for h in folder)
    one = r.search("token", k=10, filters=Filters(file="app/auth.py")).hits
    assert {h.chunk.file_path for h in one} == {"app/auth.py"}
    typed = r.search("password", k=10, filters=Filters(types=("function",))).hits
    assert typed and all(h.chunk.symbol_type == "function" for h in typed)
    empty = r.search("x", k=5, filters=Filters(folder="does/not/exist"))
    assert empty.hits == [] and empty.notes


def test_max_per_file_diversifies_results(sample_stack):
    r = HybridRetriever(sample_stack)
    hits = r.search("def", k=10, max_per_file=1).hits
    files = [h.chunk.file_path for h in hits]
    assert len(files) == len(set(files))


def test_cross_encoder_request_degrades_gracefully_without_the_model(sample_stack):
    r = HybridRetriever(sample_stack)
    r._cross_failed = True
    res = r.search("hash password", k=3, cross_encoder=True)
    assert res.hits and any("Cross-encoder" in n for n in res.notes)


def test_heuristic_rewrite_expands_synonyms_and_resolves_pronouns():
    p = heuristic_rewrite("where is authentication handled")
    assert {"login", "jwt", "token"} <= set(p.keywords.split())
    follow = heuristic_rewrite("what calls it?", previous_symbols=["verify_token"])
    assert "verify_token" in follow.keywords


def test_llm_rewrite_uses_model_output_and_falls_back_on_failure():
    good = llm_rewrite("where is login", "", lambda p: "login authenticate verify_token session\nextra", None)
    assert good.source == "llm" and "verify_token" in good.bm25_query

    def boom(_):
        raise RuntimeError("down")

    bad = llm_rewrite("where is login", "", boom, None)
    assert bad.source == "heuristic"
    assert llm_rewrite("q", "", lambda p: "", None).source == "heuristic"
