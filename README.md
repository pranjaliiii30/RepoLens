# 🔎 RepoLens — Chat with any GitHub repo

Paste a GitHub URL (or upload a ZIP), and ask questions like *"where is authentication handled?"*.
Answers cite exact files and line numbers, and **every citation is verified** against the code that was actually retrieved.

Beyond chat, RepoLens is a small code-intelligence workbench: call graph, import graph, entry points, HTTP routes,
complexity hotspots, a security scan, dead-code candidates, and a retrieval benchmark that measures its own search quality.

![RepoLens](docs/screenshots/chat.png)
<sub>Add screenshots to `docs/screenshots/` (chat.png, insights.png, graphs.png, security.png, lab.png).</sub>

## Features

| Area | What you get |
|---|---|
| **Ingestion** | GitHub URL (`owner/repo`, `/tree/branch` URLs), ZIP upload, optional branch + token for private repos. Shallow clone (depth 1), size / file-count warnings with *Continue anyway*, safe ZIP extraction (zip-slip, symlinks, zip-bomb limits), token redacted from every error. |
| **Code-aware chunking** | Python via `ast`; JS/TS/Java/Go/C/C++/Rust/C# via tree-sitter (when installed) or a built-in string/comment-aware brace extractor; Markdown by heading; sliding window (60 lines / 10 overlap) for everything else. One chunk per function/method, a summary chunk per class, module-level runs, long symbols split into parts with exact line ranges. |
| **Hybrid retrieval** | Vector (ChromaDB, cosine) + BM25 with a code-aware tokenizer (`verifyToken` ≙ `verify_token`), fused with Reciprocal Rank Fusion, tie-break reranking, optional cross-encoder rerank, per-file diversity cap. |
| **Query understanding** | Gemini rewrites your question into identifier-rich keywords (with an offline synonym-expansion fallback); follow-up questions resolve pronouns from chat history. |
| **Answers** | Streaming, strictly grounded prompt, `[file:start-end]` citations rendered as badges and **verified** (hallucinated ones are struck through), sources panel with scores + highlighted code + copy button, follow-up chips, quick actions, 👍/👎 feedback, download chat as Markdown. |
| **Works offline** | No API key? You still get ranked code locations with citations. No model download? An offline hashing embedder keeps everything running. |
| **Explorer** | File tree, searchable file picker, syntax-highlighted viewer, per-file symbol table, callers/callees for any function, scope chat to one file. |
| **Insights** | Language donut, largest files, most-imported modules, tech-stack detection, entry points, routes (Flask/FastAPI/Express/Spring/Gin/Django), dependency manifests, AI or offline repo overview. |
| **Graphs** | Mermaid module-import graph, folder dependency graph, call graph (focus on any function). |
| **Quality & Security** | Security score + findings (secrets, SQL injection patterns, `eval`, weak hashes, TLS off, …), complexity hotspots, dead-code candidates, docstring coverage, TODO/FIXME scan, Markdown report export. |
| **Retrieval Lab** | Side-by-side vector / BM25 / hybrid playground, plus an auto-generated benchmark (Recall@k, MRR) so retrieval changes are measured, not guessed. |
| **Ops** | Per-commit index cache (re-opening is instant), atomic index writes, debug log panel, Docker, GitHub Actions CI, terminal CLI. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt                # optional: tree-sitter parsing
cp .env.example .env                                    # add GEMINI_API_KEY (free: aistudio.google.com/apikey)
streamlit run app.py
```

Requires Python 3.10+ and `git` on PATH. The first index downloads the embedding model (~90 MB for `all-MiniLM-L6-v2`).
Pick `offline-hashing` in the sidebar to skip the download (keyword-style matching only).

**Docker:** `docker compose up --build` → http://localhost:8501

**CLI:**
```bash
python repolens_cli.py index https://github.com/pallets/click
python repolens_cli.py ask   pallets-click "where are command groups resolved?"
python repolens_cli.py bench pallets-click -n 80
```

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Answers, query rewriting, overview. Optional; can also be pasted in the sidebar. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any Gemini model id your key can use (also selectable in the sidebar). |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model. |
| `REPOLENS_DATA_DIR` | `.repolens_data` | Index / clone-free cache location. |
| `REPOLENS_ALLOW_LOCAL` | `0` | `1` enables indexing a server-side folder path. Keep `0` on any shared deployment. |

## Architecture

```mermaid
flowchart LR
  A[GitHub URL / ZIP] --> B[ingest.py<br/>clone · safe unzip · filter]
  B --> C[chunker.py<br/>AST · tree-sitter · brace · markdown · window]
  C --> D[indexer.py<br/>embeddings · ChromaDB · BM25]
  D --> S[(index cache<br/>per repo + commit)]
  Q[Question] --> R1[retriever.py<br/>query rewrite]
  R1 --> R2[vector + BM25]
  S --> R2
  R2 --> R3[RRF fusion · rerank]
  R3 --> L[llm.py<br/>grounded prompt · Gemini stream]
  L --> V[citation verification]
  V --> UI[Streamlit UI]
  S --> I[insights.py<br/>call graph · imports · security · hotspots]
  I --> UI
  S --> E[evaluation.py<br/>Recall@k · MRR]
  E --> UI
```

```
repolens/
├── app.py                  Streamlit UI (Chat · Explorer · Insights · Graphs · Quality & Security · Retrieval Lab)
├── repolens_cli.py         Terminal interface
├── core/
│   ├── ingest.py           URL parsing, clone, safe ZIP, file filtering
│   ├── chunker.py          Code-aware chunking
│   ├── indexer.py          Embedders, ChromaDB/numpy stores, BM25, cache lifecycle
│   ├── retriever.py        Hybrid search, RRF, rerank, query rewriting
│   ├── llm.py              Gemini client, prompts, retries, citation checks, offline answers
│   ├── insights.py         Static analysis + security scan
│   ├── evaluation.py       Retrieval benchmark
│   ├── service.py          Cache-aware index-or-load, repo overview
│   ├── render.py           Streamlit-free rendering helpers (HTML-safe markdown, Mermaid, exports)
│   ├── models.py · config.py · utils.py
├── ui/                     CSS + reusable components
└── tests/                  pytest suite
```

## Design decisions

* **Why code-aware chunking?** Fixed-size windows cut functions in half and mix unrelated code, which hurts both embeddings and citations. One chunk per symbol means a hit *is* a function with a real line range, and each embedded chunk is prefixed with `File | Class | Method` so the model knows where it lives.
* **Why hybrid search?** Embeddings capture intent ("where do we check login?"), BM25 nails exact identifiers (`verify_token`). RRF fuses *ranks*, so there is no score normalisation to tune. The tokenizer splits snake/camel case so both signals see `verifyToken` and `verify_token` alike.
* **Why verify citations?** LLMs sometimes cite lines they never saw. RepoLens checks each `[file:a-b]` against the retrieved chunk ranges and visibly flags the ones that do not match.
* **Why a tiny rerank boost?** Early on, larger symbol/path boosts *hurt* the benchmark (MRR 0.92 → 0.67) because RRF scores for adjacent ranks differ by only ~0.0005. The boosts are now tie-breakers (`BOOST_SCALE` in `retriever.py`), and the Retrieval Lab exists so such changes are measured.
* **Security posture:** model output is HTML-escaped before rendering (repo content is untrusted); ZIPs are extracted defensively; clones never prompt for credentials; tokens are redacted from errors and never stored; local-path indexing is off by default.

## Retrieval benchmark

The Retrieval Lab generates questions from a repo's own docstrings (with the symbol's name removed so string matching can't cheat) plus name-derived questions, then checks whether each strategy returns the right function.

Measured while developing, on RepoLens's own source (36 files, 425 chunks, 100 questions) using the **offline hashing embedder** — a lexical stand-in, *not* a semantic model:

| Strategy | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| Vector only | 0.55 | 0.91 | 0.95 | 0.692 |
| BM25 only | 0.68 | 0.97 | 0.98 | 0.805 |
| Hybrid (RRF) | 0.62 | 0.97 | 0.99 | 0.770 |
| **Hybrid + rerank** | **0.74** | **0.98** | **0.99** | **0.846** |

Read this honestly: with a lexical "vector" signal, plain RRF did not beat BM25 alone. A real semantic embedder behaves differently, so **run the benchmark on your own repos with your own embedder** and quote *those* numbers.

Scale check (synthetic, offline embedder): 2,000 files / 184k lines / 22k chunks indexed in ~6 s, hybrid search median ~6 ms, ~210 MB peak memory. With a real embedding model, embedding time dominates indexing.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

93 tests cover the chunker (Python AST, nested classes, decorators, syntax-error fallback, long-symbol splitting, BOM/CRLF, JS/Java/Go/Rust/C), ingestion safety (zip-slip, symlinks, bombs, URL validation), index lifecycle and caching, retrieval quality and filters, LLM retries and citation verification, static analysis and security rules, HTML-injection safety, and the benchmark. An `AppTest` smoke test renders the whole Streamlit app when Streamlit is installed.

## Verification status & limitations

Be aware of what has and has not been exercised:

* **Verified in development (offline sandbox):** chunking, ingestion filters/ZIP safety, indexing + cache, BM25/RRF/hybrid search, insights, security rules, rendering safety, CLI, the benchmark, and a headless run of `app.py` against a stub of the Streamlit API.
* **Not exercised there (no network / packages):** the real Streamlit renderer, live `git clone`, ChromaDB, sentence-transformers, tree-sitter, and the Gemini API. Those code paths have graceful fallbacks (numpy store, offline embedder, brace parser, extractive answers) and the retry/error mapping is unit-tested with fakes, but run the app once and `pytest` (which includes the `AppTest` smoke test) on your machine to confirm your versions.
* `requirements.txt` uses version *ranges*; run `pip freeze > requirements.lock` after a good install to pin exactly.
* Static analysis is heuristic: call resolution is by name (no type inference), dead-code and security findings can be false positives/negatives, and routes are detected by pattern.
* Very large monorepos: the default soft limits are 200 MB / 5,000 files; raise `MAX_FILES` / `MAX_REPO_MB` in `core/config.py` if you need more.

## Ideas to extend

* Incremental re-indexing: re-embed only files changed since the last commit.
* Git-history questions ("who last changed this and why?") via `git blame` / `git log` chunks.
* Multi-repo chat and cross-repo search.
* Pull-request review mode: retrieve context for a diff and explain risk.
* Evaluation with human-written questions and answer-faithfulness scoring.

## Resume bullets

> Built **RepoLens**, a RAG assistant for GitHub repositories: AST/tree-sitter code-aware chunking, hybrid BM25 + vector retrieval fused with RRF, and line-level citations verified against retrieved code to detect hallucinated references.
>
> Added an in-app retrieval benchmark (auto-generated queries, Recall@k / MRR) and used it to tune reranking — [**fill in your own measured numbers from the Retrieval Lab**].
>
> Implemented static analysis (call/import graphs, complexity hotspots, security scan), safe ingestion (zip-slip/symlink/bomb protection, token redaction), and a 90+ test pytest suite with CI.

## License

MIT — add a `LICENSE` file before publishing.
