"""RepoLens — chat with any GitHub repository. Run with:  streamlit run app.py"""
from __future__ import annotations

import collections
import json
import logging
import os
import time

import pandas as pd
import streamlit as st

from core import config, insights
from core.evaluation import STRATEGIES, generate_cases, run_benchmark
from core.indexer import (EmbedderUnavailable, IndexParams, chroma_available, delete_index, list_indexes,
                          load_index, open_stack)
from core.ingest import RepoError, RepoTooLarge, SourceSpec
from core.llm import (QUICK_ACTIONS, SYSTEM_PROMPT, GeminiClient, LLMError, build_prompt, extractive_answer,
                      heuristic_followups, history_text, suggest_followups, validate_citations)
from core.models import SearchHit
from core.render import (benchmark_summary, chat_to_markdown, safe_markdown, security_report_markdown,
                         tree_lines)
from core.retriever import Filters, HybridRetriever, heuristic_rewrite, llm_rewrite
from core.service import generate_overview, index_or_load, offline_overview
from ui import components as ui

st.set_page_config(page_title="RepoLens · Chat with any codebase", page_icon="🔎", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

# ============================================================================ logging (shown in sidebar)
class _Ring(logging.Handler):
    def __init__(self, cap: int = 200):
        super().__init__(logging.INFO)
        self.lines: collections.deque[str] = collections.deque(maxlen=cap)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@st.cache_resource
def _log_ring() -> _Ring:
    ring = _Ring()
    root = logging.getLogger("repolens")
    root.setLevel(logging.INFO)
    root.addHandler(ring)
    return ring


RING = _log_ring()
log = logging.getLogger("repolens.app")


# ============================================================================ state
def _init_state() -> None:
    defaults = {"index": None, "stack": None, "retriever": None, "messages": [], "pending_large": None,
                "overview": {}, "bench": None, "queued_q": None, "flash": None, "repo_url": "",
                "scope_file": "", "feedback": {}}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


@st.cache_resource(show_spinner=False, max_entries=3)
def _stack_for(repo_id: str):
    index = load_index(repo_id)
    stack = open_stack(index)
    return index, stack, HybridRetriever(stack)


def _activate(repo_id: str) -> None:
    index, stack, retr = _stack_for(repo_id)
    st.session_state.update(index=index, stack=stack, retriever=retr, messages=[], overview={}, bench=None,
                            scope_file="")


@st.cache_resource(show_spinner=False, max_entries=4)
def _memo(repo_id: str) -> dict:
    """Per-index cache for static analyses. Streamlit re-runs every tab on each interaction, so anything
    that walks the whole repo (security scan, import graph, ...) must be computed once, not per message."""
    return {}


def cached(fn, *args):
    memo = _memo(st.session_state.index.repo_id)
    key = (fn.__name__, args)
    if key not in memo:
        memo[key] = fn(st.session_state.index, *args)
    return memo[key]


def _client() -> GeminiClient:
    return GeminiClient(st.session_state.get("api_key") or None,
                        st.session_state.get("model", config.DEFAULT_GEMINI_MODEL),
                        st.session_state.get("temperature", 0.2))


# ============================================================================ sidebar
def _indexing(spec: SourceSpec, params: IndexParams, force: bool) -> None:
    with st.status("Indexing repository …", expanded=True) as status:
        bar = st.progress(0.0)

        def cb(frac: float, msg: str) -> None:
            bar.progress(min(1.0, max(0.0, frac)))
            status.update(label=msg)

        try:
            index, cached = index_or_load(spec, params, force=force, progress=cb)
            st.session_state.pending_large = None
            _stack_for.clear()
            _activate(index.repo_id)
            status.update(label=("Loaded from cache ⚡" if cached else "Indexed ✔") +
                          f" — {len(index.files):,} files, {len(index.chunks):,} chunks", state="complete", expanded=False)
            if index.embedder_note:
                st.warning(index.embedder_note)
            for w in index.warnings[:3]:
                st.toast(w, icon="⚠️")
        except RepoTooLarge as exc:
            status.update(label="Repository is large", state="error")
            st.session_state.pending_large = {"spec": spec, "params": params, "msg": str(exc)}
        except (RepoError, EmbedderUnavailable) as exc:
            status.update(label="Couldn't index", state="error")
            st.error(str(exc))
        except Exception as exc:  # unexpected: log, show a safe message
            log.exception("indexing failed")
            status.update(label="Unexpected error", state="error")
            st.error(f"Something went wrong while indexing ({exc.__class__.__name__}). Check the debug log in the sidebar.")


def sidebar() -> dict:
    sb = st.sidebar
    sb.markdown("## 🔎 RepoLens")
    if st.session_state.flash:
        sb.success(st.session_state.flash)
        st.session_state.flash = None

    # ---- 1. repository ----------------------------------------------------------------------------
    with sb.expander("📥 Repository", expanded=st.session_state.index is None):
        kinds = ["GitHub URL", "ZIP upload"] + (["Local folder"] if config.ALLOW_LOCAL_PATHS else [])
        kind = st.radio("Source", kinds, horizontal=True, label_visibility="collapsed")
        spec = SourceSpec(kind="url")
        if kind == "GitHub URL":
            url = st.text_input("Repository URL", key="repo_url", placeholder="https://github.com/owner/repo")
            c1, c2 = st.columns(2)
            branch = c1.text_input("Branch", placeholder="default")
            token = c2.text_input("GitHub token", type="password", placeholder="private repos",
                                  help="Used only for this clone, never stored or logged.")
            spec = SourceSpec(kind="url", value=url, branch=branch.strip() or None, token=token.strip() or None)
        elif kind == "ZIP upload":
            up = st.file_uploader("Project ZIP", type=["zip"])
            spec = SourceSpec(kind="zip", value=up.name if up else "", zip_bytes=up.getvalue() if up else None)
        else:
            path = st.text_input("Folder path", placeholder="/path/to/project")
            spec = SourceSpec(kind="local", value=path)
        ignores = st.text_area("Extra ignore patterns", placeholder="docs/\n*.generated.*\nvendor/", height=70,
                               help="One glob per line. Built-in ignores already skip .git, node_modules, venv, "
                                    "dist, build, lock files, binaries, minified files and files over 500 KB.")
        emb_choices = config.EMBEDDING_CHOICES
        emb = st.selectbox("Embedding model", emb_choices, index=emb_choices.index(config.DEFAULT_EMBEDDER)
                           if config.DEFAULT_EMBEDDER in emb_choices else 0,
                           help="Local sentence-transformers model. 'offline-hashing' needs no download "
                                "(keyword-style matching).")
        ts = st.toggle("Use tree-sitter when available", value=True,
                       help="More precise parsing for JS/TS/Java/Go/C/C++/Rust. Falls back automatically.")
        params = IndexParams(embedder=emb, extra_ignores=[p.strip() for p in ignores.splitlines() if p.strip()],
                             use_tree_sitter=ts, backend="chroma")
        b1, b2 = st.columns([3, 2])
        go = b1.button("Index repository", type="primary")
        force = b2.button("Re-index", help="Ignore the cache and rebuild from scratch")
    if go or force:
        _indexing(spec, params, force)

    pending = st.session_state.pending_large
    if pending:
        sb.warning(pending["msg"] + "  \nIndexing may be slow and memory-hungry.")
        c1, c2 = sb.columns(2)
        if c1.button("Continue anyway", type="primary"):
            pending["params"].allow_large = True
            _indexing(pending["spec"], pending["params"], False)
        if c2.button("Cancel"):
            st.session_state.pending_large = None
            st.rerun()

    # ---- 2. model ----------------------------------------------------------------------------------
    with sb.expander("🤖 Model & retrieval"):
        env_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        st.text_input("Gemini API key", key="api_key", type="password",
                      placeholder="using key from .env" if env_key else "paste key (aistudio.google.com)",
                      help="Read from GEMINI_API_KEY in .env when left empty. Never hard-coded or stored.")
        st.selectbox("Gemini model", list(dict.fromkeys(config.GEMINI_MODEL_CHOICES)), key="model")
        st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="temperature")
        st.slider("Top-k chunks", 3, 20, config.DEFAULT_TOP_K, key="top_k")
        st.selectbox("Search mode", ["hybrid", "vector", "bm25"], key="mode",
                     help="hybrid = vector + BM25 fused with Reciprocal Rank Fusion")
        st.toggle("Heuristic rerank", value=True, key="rerank", help="Small tie-breaking boosts for symbol/path matches.")
        st.toggle("Cross-encoder rerank", value=False, key="cross",
                  help="Re-scores the top 30 with a cross-encoder (downloads a small model; slower, often more precise).")
        st.toggle("AI query rewriting", value=True, key="rewrite",
                  help="Uses Gemini to turn your question into identifier-rich search keywords.")
        st.toggle("Suggest follow-ups", value=True, key="followups")
        st.toggle("Show retrieval debug", value=False, key="debug_ui")

    # ---- 3. filters --------------------------------------------------------------------------------
    index = st.session_state.index
    filters = Filters()
    if index:
        with sb.expander("🎯 Filters"):
            langs = st.multiselect("Language", list(index.languages))
            folders = sorted({"/".join(f.path.split("/")[:d]) for f in index.files for d in (1, 2) if f.path.count("/") >= d})
            folder = st.selectbox("Folder", [""] + folders, format_func=lambda x: x or "(all)")
            files = [""] + sorted(f.path for f in index.files)
            scope = st.session_state.get("scope_file", "")
            fsel = st.selectbox("File", files, index=files.index(scope) if scope in files else 0,
                                format_func=lambda x: x or "(all)")
            kinds = st.multiselect("Symbol type", ["function", "method", "class", "module", "doc", "config", "block"])
            filters = Filters(tuple(langs), folder, fsel, tuple(kinds))

    # ---- 4. indexed repos --------------------------------------------------------------------------
    with sb.expander("🗂️ Indexed repositories"):
        rows = list_indexes()
        if not rows:
            st.caption("Nothing indexed yet.")
        for r in rows:
            cur = index and index.repo_id == r["repo_id"]
            st.markdown(f"**{r['name']}** {'🟢' if cur else ''}  \n"
                        f"<span class='rl-src-head'>{r['branch']} · {r['n_files']} files · {r['n_chunks']} chunks · {r['embedder']}</span>",
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            if a.button("Open", key=f"open_{r['repo_id']}", disabled=bool(cur)):
                try:
                    _activate(r["repo_id"])
                    st.rerun()
                except (RepoError, EmbedderUnavailable) as exc:
                    st.error(str(exc))
            if b.button("Delete", key=f"del_{r['repo_id']}"):
                delete_index(r["repo_id"])
                _stack_for.clear()
                if cur:
                    st.session_state.update(index=None, stack=None, retriever=None, messages=[])
                st.rerun()
        st.caption(f"Vector store: {'ChromaDB' if chroma_available() else 'numpy (install chromadb for persistence)'}")

    # ---- 5. chat controls & debug -------------------------------------------------------------------
    if index and st.session_state.messages:
        if sb.button("🧹 Clear chat"):
            st.session_state.messages = []
            st.rerun()
    with sb.expander("🪵 Debug log"):
        st.code("\n".join(list(RING.lines)[-40:]) or "(empty)", language="text")
    return {"filters": filters}


# ============================================================================ chat
def _answer(question: str, flt: Filters) -> dict:
    index, retr = st.session_state.index, st.session_state.retriever
    client = _client()
    history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]
    prev_symbols = [s for m in reversed(history) if m["role"] == "assistant" for s in m.get("symbols", [])][:3]
    t0 = time.perf_counter()
    if client.available and st.session_state.get("rewrite", True):
        plan = llm_rewrite(question, history_text(history), lambda p: client.generate(p, temperature=0.0, max_tokens=80),
                           prev_symbols)
    else:
        plan = heuristic_rewrite(question, prev_symbols)
    res = retr.search(question, k=st.session_state.get("top_k", config.DEFAULT_TOP_K),
                      mode=st.session_state.get("mode", "hybrid"), filters=flt,
                      rerank=st.session_state.get("rerank", True), cross_encoder=st.session_state.get("cross", False),
                      plan=plan)
    return {"hits": res.hits, "plan": plan, "timings": res.timings_ms, "notes": res.notes, "client": client,
            "history": history, "t0": t0}


def render_message(i: int, m: dict) -> None:
    index = st.session_state.index
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🔎"):
        if m["role"] == "user":
            st.markdown(m["content"])
            return
        verified = {tuple(v) for v in m.get("verified_set", [])}
        st.markdown(safe_markdown(m["content"], verified if m.get("mode") == "ai" else None), unsafe_allow_html=True)
        ok, bad = m.get("verified", (0, 0))
        if ok or bad:
            msg = f"✓ {ok} citation{'s' if ok != 1 else ''} verified against retrieved code"
            if bad:
                msg += f" · ⚠ {bad} not found in retrieved code (shown struck through)"
            st.markdown(f'<div class="rl-verify">{msg}</div>', unsafe_allow_html=True)
        refs = m.get("refs", [])
        if refs:
            with st.expander(f"📚 Sources ({len(refs)})"):
                for j, ref in enumerate(refs):
                    chunk = index.by_id.get(ref["id"])
                    if chunk:
                        ui.source_card(SearchHit(chunk, ref["score"], ref.get("vec_score"), ref.get("bm25_score"),
                                                 ref.get("vec_rank"), ref.get("bm25_rank"), ref.get("cross_score")),
                                       key=f"src_{i}_{j}")
        if st.session_state.get("debug_ui") and m.get("debug"):
            with st.expander("🔬 Retrieval debug"):
                d = m["debug"]
                st.markdown(f"**Query plan ({d['plan_source']}):** `{d['keywords']}`")
                st.json(d["timings"], expanded=False)
        if hasattr(st, "feedback"):
            with st.container():
                fb = st.feedback("thumbs", key=f"fb_{i}")
                if fb is not None and st.session_state.feedback.get(i) != fb:
                    st.session_state.feedback[i] = fb
                    _save_feedback(m, fb)
        if m.get("followups") and i == len(st.session_state.messages) - 1:
            cols = st.columns(len(m["followups"]))
            for col, q in zip(cols, m["followups"]):
                if col.button(q, key=f"fu_{i}_{hash(q) & 0xFFFF}"):
                    st.session_state.queued_q = q
                    st.rerun()


def _save_feedback(m: dict, value: int) -> None:
    try:
        path = config.DATA_DIR / "feedback.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "question": m.get("question", ""), "up": bool(value),
                                 "mode": m.get("mode")}) + "\n")
    except OSError:
        pass


def tab_chat(flt: Filters) -> None:
    index = st.session_state.index
    if not st.session_state.messages:
        st.markdown("##### Quick actions")
        keys = list(QUICK_ACTIONS)
        for row in (keys[:4], keys[4:]):
            cols = st.columns(len(row))
            for col, k in zip(cols, row):
                if col.button(k, key=f"qa_{k}"):
                    st.session_state.queued_q = QUICK_ACTIONS[k]
                    st.rerun()
        if flt.active:
            st.info("Search is restricted by the filters in the sidebar.")
        st.markdown("<div class='rl-empty'>Ask anything about "
                    f"<b>{index.name}</b> — e.g. <i>“where is authentication handled?”</i></div>", unsafe_allow_html=True)
    for i, m in enumerate(st.session_state.messages):
        render_message(i, m)

    question = st.chat_input(f"Ask about {index.name} …") or st.session_state.pop("queued_q", None)
    if st.session_state.messages:
        c1, _ = st.columns([2, 6])
        c1.download_button("⬇ Download chat (.md)", chat_to_markdown(st.session_state.messages, index.name),
                           file_name=f"repolens-{index.name}.md", mime="text/markdown")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🔎"):
        slot = st.empty()
        try:
            with st.spinner("Searching the codebase …"):
                ctx = _answer(question, flt)
        except EmbedderUnavailable as exc:
            slot.error(str(exc))
            st.session_state.messages.pop()
            return
        except Exception as exc:
            log.exception("retrieval failed")
            slot.error(f"Search failed ({exc.__class__.__name__}). See the debug log.")
            st.session_state.messages.pop()
            return
        for note in ctx["notes"]:
            st.caption(f"ℹ️ {note}")
        hits, client = ctx["hits"], ctx["client"]
        mode, text = "ai", ""
        if not hits:
            text, mode = "I couldn't find this in the indexed code. Try rephrasing, or loosen the sidebar filters.", "offline"
            slot.markdown(text)
        elif client.available:
            prompt = build_prompt(question, hits, ctx["history"][:-1],
                                  f"{index.name} ({', '.join(list(index.languages)[:4])})")
            try:
                for tok in client.stream(prompt, system=SYSTEM_PROMPT):
                    text += tok
                    slot.markdown(safe_markdown(text + " ▌", badges=False), unsafe_allow_html=True)
                if not text.strip():
                    raise LLMError("Gemini returned an empty response.")
            except LLMError as exc:
                st.warning(f"{exc} Showing the matching code locations instead.")
                text, mode = extractive_answer(question, hits, str(exc)), "offline"
        else:
            text, mode = extractive_answer(question, hits), "offline"
        ok, bad = validate_citations(text, hits) if mode == "ai" else ([], [])
        verified_set = [(c.path, c.start, c.end) for c in ok]
        slot.markdown(safe_markdown(text, {(h.chunk.file_path, h.chunk.start_line, h.chunk.end_line) for h in hits}
                                    if mode == "ai" else None), unsafe_allow_html=True)
        followups: list[str] = []
        if st.session_state.get("followups", True) and hits:
            followups = suggest_followups(client, question, text) if (client.available and mode == "ai") else []
            followups = followups or heuristic_followups(hits)
        st.session_state.messages.append({
            "role": "assistant", "content": text, "mode": mode, "question": question,
            "refs": [h.to_ref() for h in hits], "sources": [h.chunk.citation for h in hits],
            "verified": (len(ok), len(bad)), "verified_set": verified_set,
            "symbols": [h.chunk.symbol_name for h in hits[:3] if h.chunk.symbol_type in ("function", "method", "class")],
            "followups": followups,
            "debug": {"plan_source": ctx["plan"].source, "keywords": ctx["plan"].keywords, "timings": ctx["timings"]},
        })
    st.rerun()


# ============================================================================ explorer
def tab_explorer() -> None:
    index = st.session_state.index
    left, right = st.columns([2, 5])
    with left:
        q = st.text_input("Find file", placeholder="type to filter …", label_visibility="collapsed")
        paths = [f.path for f in index.files if q.lower() in f.path.lower()]
        with st.expander("🌳 File tree", expanded=False):
            st.code(tree_lines(index), language="text")
        choice = st.selectbox("File", paths[:2000], label_visibility="collapsed") if paths else None
        if not paths:
            st.caption("No files match.")
    if not choice:
        return
    info = next(f for f in index.files if f.path == choice)
    with right:
        st.markdown(f"### `{choice}`")
        st.caption(f"{info.language} · {info.lines:,} lines · {info.size / 1024:.1f} KB")
        a, b = st.columns([1, 1])
        if a.button("🎯 Scope chat to this file"):
            st.session_state.scope_file = choice
            st.toast(f"Chat is now scoped to {choice}", icon="🎯")
            st.rerun()
        try:
            text = index.read_file(choice)
        except Exception as exc:
            st.error(f"Couldn't read the file: {exc}")
            return
        syms = [c for c in index.chunks_for(choice) if c.symbol_type in ("function", "method", "class") and c.part == 0]
        t1, t2 = st.tabs([f"Code", f"Symbols ({len(syms)})"])
        with t1:
            if len(text) > 120_000:
                st.warning("Large file: showing the first 2,000 lines.")
                text = "\n".join(text.splitlines()[:2000])
            ui.code(text, info.language)
        with t2:
            if not syms:
                st.caption("No functions or classes were extracted from this file.")
            else:
                ui.df([{"symbol": c.qualname, "type": c.symbol_type, "lines": f"{c.start_line}-{c.end_line}",
                        "length": c.n_lines, "complexity": c.complexity or None,
                        "doc": (c.docstring.strip().splitlines() or [""])[0][:90]} for c in syms])
                pick = st.selectbox("Inspect symbol", [c.qualname for c in syms])
                if pick:
                    sym_name = pick.split(".")[-1]
                    cA, cB = st.columns(2)
                    callers = cached(insights.callers_of, sym_name)
                    callees = cached(insights.callees_of, sym_name)
                    cA.markdown(f"**Called by ({len(callers)})**")
                    for c in callers[:12]:
                        cA.markdown(f"- `{c.qualname}` — `{c.citation}`")
                    cB.markdown(f"**Calls ({len(callees)})**")
                    for c in callees[:12]:
                        cB.markdown(f"- `{c.qualname}` — `{c.citation}`")


# ============================================================================ insights
def tab_insights() -> None:
    index = st.session_state.index
    stats = cached(insights.repo_stats)
    cols = st.columns(5)
    cols[0].metric("Files", f"{stats['files']:,}")
    cols[1].metric("Lines of code", f"{stats['loc']:,}")
    cols[2].metric("Chunks", f"{stats['chunks']:,}")
    cols[3].metric("Languages", stats["n_languages"])
    cols[4].metric("Test files", stats["test_files"])

    st.markdown("#### 🧭 Repo overview")
    key = (index.repo_id, bool(_client().available))
    cache = st.session_state.overview
    if key not in cache:
        cache[key] = offline_overview(index)
    c1, c2 = st.columns([5, 2])
    with c1:
        st.markdown(cache[key])
    with c2:
        if _client().available:
            if st.button("✨ Generate AI overview", type="primary"):
                with st.spinner("Reading the repo …"):
                    cache[key] = generate_overview(_client(), index)
                st.rerun()
        else:
            st.caption("Add a Gemini key for an AI-written overview.")
        stack = cached(insights.detect_tech_stack)
        if stack:
            st.markdown("**Tech stack**  \n" + " ".join(ui.pill(s) for s in stack), unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Language breakdown")
        ui.plot(ui.language_donut(stats["languages"]), key="lang_donut")
    with b:
        st.markdown("#### Largest files")
        top = stats["largest_files"]
        ui.plot(ui.hbar([p.split("/")[-1] if len(p) > 36 else p for p, _ in top], [n for _, n in top]), key="largest")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Most functions per file")
        top = stats["most_functions"]
        if top:
            ui.plot(ui.hbar([p for p, _ in top], [n for _, n in top], color="#22d3ee"), key="most_fn")
    with b:
        st.markdown("#### Most imported modules")
        mi = cached(insights.most_imported)
        if mi:
            ui.plot(ui.hbar([p for p, _ in mi], [n for _, n in mi], color="#34d399"), key="most_imp")
        else:
            st.caption("No internal imports detected.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🚪 Entry points")
        eps = cached(insights.find_entry_points)
        ui.df(eps) if eps else st.caption("None detected.")
    with c2:
        st.markdown("#### 🌐 HTTP routes")
        routes = cached(insights.find_routes)
        ui.df(routes) if routes else st.caption("No routes detected.")
    st.markdown("#### 📦 Dependencies")
    deps = cached(insights.detect_dependencies)
    if deps:
        for f, names in deps.items():
            st.markdown(f"**`{f}`** — " + " ".join(ui.pill(n) for n in names[:40]), unsafe_allow_html=True)
    else:
        st.caption("No dependency manifests found.")
    with st.expander("Skipped files"):
        ui.df([{"reason": k, "count": v} for k, v in index.skipped.items()]) if index.skipped else st.caption("Nothing skipped.")
    if index.warnings:
        with st.expander(f"Parser warnings ({len(index.warnings)})"):
            st.code("\n".join(index.warnings[:100]), language="text")


# ============================================================================ graphs
def _function_names(index) -> list[str]:
    return sorted({c.symbol_name for c in index.chunks if c.symbol_type in ("function", "method") and c.part == 0})


def tab_graphs() -> None:
    index = st.session_state.index
    kind = st.radio("Diagram", ["Module imports", "Folder dependencies", "Call graph"], horizontal=True)
    if kind == "Module imports":
        n = st.slider("Max nodes", 8, 60, 30, key="mod_nodes")
        ui.mermaid(cached(insights.module_graph_mermaid, n, False), 560)
        st.caption("Internal imports only (Python, JS/TS, Java, C/C++). Edges point from importer → imported.")
    elif kind == "Folder dependencies":
        ui.mermaid(cached(insights.module_graph_mermaid, 30, True), 480)
    else:
        names = cached(_function_names)
        focus = st.selectbox("Focus on a function (empty = most connected)", [""] + names, format_func=lambda x: x or "(top connected)")
        depth = st.slider("Depth", 1, 3, 2)
        ui.mermaid(cached(insights.call_graph_mermaid, focus or None, depth, 40), 560)
        st.caption("Calls are resolved by name (static, best-effort) — dynamic dispatch isn't tracked.")


# ============================================================================ quality & security
def tab_quality() -> None:
    index = st.session_state.index
    findings = cached(insights.security_scan, 300)
    score = insights.security_score(findings)
    cov = cached(insights.docstring_coverage)
    hot = cached(insights.complexity_hotspots, 15)
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        ui.plot(ui.gauge(score, "Security score"), key="sec_gauge")
    with c2:
        ui.plot(ui.gauge(cov["pct"], "Docstring coverage"), key="doc_gauge")
    with c3:
        sev = pd.Series([f["severity"] for f in findings]).value_counts() if findings else pd.Series(dtype=int)
        st.markdown("**Findings by severity**")
        for s in ("critical", "high", "medium", "low"):
            st.markdown(f"<span class='rl-sev-{s}'>{s.title()}</span>: {int(sev.get(s, 0))}", unsafe_allow_html=True)
        st.download_button("⬇ Report (.md)", security_report_markdown(index, findings, score),
                           file_name=f"{index.name}-security.md", mime="text/markdown")
    st.caption("Static, regex-based heuristics — expect false positives. Secrets are masked in the table.")

    t1, t2, t3, t4, t5 = st.tabs(["🛡️ Security", "🔥 Complexity", "🧟 Dead-code candidates", "📝 Docs coverage", "✅ TODOs"])
    with t1:
        if not findings:
            st.success("No issues matched the built-in rules.")
        else:
            pick = st.multiselect("Severity", ["critical", "high", "medium", "low"], default=["critical", "high", "medium", "low"])
            ui.df([{**f, "location": f"{f['file']}:{f['line']}"} for f in findings if f["severity"] in pick][:400])
    with t2:
        ui.df(hot) if hot else st.caption("No functions with complexity data.")
        st.caption("Cyclomatic complexity estimate: >10 medium, >20 high.")
    with t3:
        dead = cached(insights.dead_code_candidates, 50)
        ui.df(dead) if dead else st.caption("No obviously unused functions found.")
        st.caption("Names that appear only once in the whole repo. Entry points, dunders, decorators and tests are excluded; dynamic use can still fool this.")
    with t4:
        st.markdown(f"**{cov['documented']} / {cov['total']}** public symbols documented")
        ui.df(cov["missing"]) if cov["missing"] else st.caption("Nothing notable is missing docs.")
    with t5:
        todos = cached(insights.todo_scan, 200)
        ui.df(todos) if todos else st.caption("No TODO/FIXME markers.")


# ============================================================================ retrieval lab
def tab_lab(flt: Filters) -> None:
    index, retr = st.session_state.index, st.session_state.retriever
    s1, s2 = st.tabs(["🔬 Search playground", "📈 Benchmark"])
    with s1:
        q = st.text_input("Query", placeholder="e.g. how are zip files extracted safely", key="lab_q")
        if q:
            plan = heuristic_rewrite(q)
            cols = st.columns(3)
            for col, (label, mode) in zip(cols, [("Vector", "vector"), ("BM25", "bm25"), ("Hybrid (RRF)", "hybrid")]):
                res = retr.search(q, k=6, mode=mode, filters=flt, rerank=(mode == "hybrid"), plan=plan)
                with col:
                    st.markdown(f"**{label}** · {res.timings_ms.get('total', 0):.1f} ms")
                    for r, h in enumerate(res.hits, 1):
                        st.markdown(f"{r}. `{h.chunk.qualname}`  \n<span class='rl-src-head'>{h.chunk.citation}</span>",
                                    unsafe_allow_html=True)
            st.caption(f"Expanded query: `{plan.keywords}`")
    with s2:
        st.markdown("Generates questions from this repo's own docstrings, then checks whether each retrieval strategy "
                    "finds the right function. Gives **Recall@k** and **MRR** you can quote.")
        n = st.slider("Number of test questions", 10, 150, 50, step=10)
        if st.button("▶ Run benchmark", type="primary"):
            cases = generate_cases(index, n=n)
            if len(cases) < 5:
                st.warning("Not enough documented functions to build a meaningful benchmark.")
            else:
                bar = st.progress(0.0)
                use = dict(STRATEGIES)
                if st.session_state.get("cross"):
                    use["Hybrid + cross-encoder"] = dict(mode="hybrid", rerank=True, cross_encoder=True)
                bench = run_benchmark(retr, cases, strategies=use, progress=lambda f, m: bar.progress(min(1.0, f), text=m))
                bar.empty()
                st.session_state.bench = bench
        bench = st.session_state.bench
        if bench:
            ui.plot(ui.benchmark_chart(bench["metrics"]), key="bench_chart")
            table = pd.DataFrame(bench["metrics"]).T
            st.dataframe(table.style.format({c: "{:.1%}" for c in table.columns if c.startswith("Recall")} |
                                            {"MRR": "{:.3f}", "Avg latency (ms)": "{:.2f}"}))
            st.success(benchmark_summary(bench))
            with st.expander("Per-question results"):
                ui.df(bench["detail"][:600])


# ============================================================================ main
side = sidebar()
index = st.session_state.index
ui.hero(index, _client().available)
if index is None:
    ui.onboarding()
else:
    if index.vector_backend == "chroma" and not chroma_available():
        st.error("This index needs ChromaDB (`pip install chromadb`).")
        st.stop()
    tabs = st.tabs(["💬 Chat", "📂 Explorer", "📊 Insights", "🕸️ Graphs", "🛡️ Quality & Security", "🧪 Retrieval Lab"])
    with tabs[0]:
        tab_chat(side["filters"])
    with tabs[1]:
        tab_explorer()
    with tabs[2]:
        tab_insights()
    with tabs[3]:
        tab_graphs()
    with tabs[4]:
        tab_quality()
    with tabs[5]:
        tab_lab(side["filters"])
