"""Repo analytics: stats, hotspots, call graph, import graph, entry points, routes, dependencies,
TODOs, dead-code candidates, and a security scan. All static and offline (no LLM needed)."""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .models import CODE_SYMBOL_TYPES, Chunk, RepoIndex
from .utils import is_test_path, posix

# ============================================================================ stats
def repo_stats(index: RepoIndex) -> dict:
    types = Counter(c.symbol_type for c in index.chunks)
    by_file_symbols: Counter = Counter()
    for c in index.chunks:
        if c.symbol_type in ("function", "method") and c.part == 0:
            by_file_symbols[c.file_path] += 1
    code_langs = {f.language for f in index.files}
    return {
        "files": len(index.files),
        "loc": index.loc,
        "chunks": len(index.chunks),
        "languages": index.languages,
        "language_files": dict(Counter(f.language for f in index.files)),
        "symbol_types": dict(types),
        "largest_files": sorted(((f.path, f.lines) for f in index.files), key=lambda x: -x[1])[:10],
        "most_functions": by_file_symbols.most_common(10),
        "n_languages": len(code_langs),
        "test_files": sum(1 for f in index.files if is_test_path(f.path)),
    }


def complexity_hotspots(index: RepoIndex, top: int = 15) -> list[dict]:
    rows = []
    for c in index.chunks:
        if c.symbol_type in ("function", "method") and c.part == 0 and c.complexity:
            rows.append({"symbol": c.qualname, "file": c.file_path, "lines": f"{c.start_line}-{c.end_line}",
                         "length": c.n_lines, "complexity": c.complexity,
                         "risk": "high" if c.complexity > 20 else "medium" if c.complexity > 10 else "low"})
    rows.sort(key=lambda r: (-r["complexity"], -r["length"]))
    return rows[:top]


def todo_scan(index: RepoIndex, limit: int = 200) -> list[dict]:
    pat = re.compile(r"(?:#|//|/\*|\*|--|<!--)\s*(TODO|FIXME|HACK|XXX|BUG)\b[:\s(]*(.*)", re.I)
    out = []
    for f in index.files:
        if f.language in ("markdown", "text"):
            continue
        try:
            text = index.read_file(f.path)
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = pat.search(line)
            if m:
                out.append({"file": f.path, "line": i, "tag": m.group(1).upper(), "text": m.group(2).strip()[:140]})
                if len(out) >= limit:
                    return out
    return out


# ============================================================================ symbols & call graph
@dataclass
class SymbolTable:
    by_name: dict[str, list[Chunk]] = field(default_factory=lambda: defaultdict(list))
    funcs: list[Chunk] = field(default_factory=list)


def build_symbol_table(index: RepoIndex) -> SymbolTable:
    st = SymbolTable()
    for c in index.chunks:
        if c.symbol_type in CODE_SYMBOL_TYPES and c.part == 0:
            st.by_name[c.symbol_name].append(c)
            if c.symbol_type in ("function", "method"):
                st.funcs.append(c)
    return st


def build_call_graph(index: RepoIndex) -> dict[str, set[str]]:
    """Edges ``caller_id -> {callee_id}``. A call resolves to a symbol with the same name; ambiguous
    names (defined more than 3 times) are skipped to avoid noise from generic names like ``get``."""
    st = build_symbol_table(index)
    graph: dict[str, set[str]] = defaultdict(set)
    for c in st.funcs:
        for name in c.calls:
            targets = st.by_name.get(name, [])
            if not targets or len(targets) > 3:
                continue
            same_file = [t for t in targets if t.file_path == c.file_path]
            for t in (same_file or targets):
                if t.id != c.id and t.symbol_type in ("function", "method"):
                    graph[c.id].add(t.id)
    return graph


def callers_of(index: RepoIndex, symbol: str) -> list[Chunk]:
    graph = build_call_graph(index)
    targets = {c.id for c in index.chunks if c.symbol_name == symbol or c.qualname == symbol}
    return [index.by_id[cid] for cid, outs in graph.items() if outs & targets and cid in index.by_id]


def callees_of(index: RepoIndex, symbol: str) -> list[Chunk]:
    graph = build_call_graph(index)
    out: dict[str, Chunk] = {}
    for c in index.chunks:
        if c.symbol_name == symbol or c.qualname == symbol:
            for cid in graph.get(c.id, ()):
                if cid in index.by_id:
                    out[cid] = index.by_id[cid]
    return list(out.values())


def _mm_id(text: str) -> str:
    return "n" + re.sub(r"\W", "_", text)


def _mm_label(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")[:48]


def call_graph_mermaid(index: RepoIndex, focus: str | None = None, depth: int = 2, max_nodes: int = 40) -> str:
    graph = build_call_graph(index)
    by_id = index.by_id
    rev: dict[str, set[str]] = defaultdict(set)
    for a, bs in graph.items():
        for b in bs:
            rev[b].add(a)
    if focus:
        roots = {c.id for c in index.chunks if c.symbol_name == focus or c.qualname == focus}
        if not roots:
            return ""
        keep, frontier = set(roots), set(roots)
        for _ in range(depth):
            nxt = set()
            for n in frontier:
                nxt |= graph.get(n, set()) | rev.get(n, set())
            frontier = nxt - keep
            keep |= nxt
            if len(keep) > max_nodes:
                break
    else:
        degree = Counter()
        for a, bs in graph.items():
            degree[a] += len(bs)
            for b in bs:
                degree[b] += 1
        keep = {n for n, _ in degree.most_common(max_nodes)}
    keep = {k for k in list(keep)[: max_nodes] if k in by_id}
    if not keep:
        return ""
    lines = ["flowchart LR"]
    for n in keep:
        c = by_id[n]
        lines.append(f'  {_mm_id(n)}["{_mm_label(c.qualname)}"]')
    for a in keep:
        for b in graph.get(a, ()):
            if b in keep:
                lines.append(f"  {_mm_id(a)} --> {_mm_id(b)}")
    return "\n".join(lines)


def dead_code_candidates(index: RepoIndex, limit: int = 50) -> list[dict]:
    """Functions never referenced by name anywhere else (heuristic; dynamic dispatch can fool it)."""
    corpus_tokens: Counter = Counter()
    for c in index.chunks:
        corpus_tokens.update(re.findall(r"[A-Za-z_]\w*", c.code))
    out = []
    for c in index.chunks:
        if c.symbol_type not in ("function", "method") or c.part != 0:
            continue
        n = c.symbol_name
        if n.startswith("__") or n in ("main", "setUp", "tearDown", "constructor", "<anonymous>") or n.startswith("test"):
            continue
        if is_test_path(c.file_path) or any(d.startswith(("@app.", "@router.", "@pytest", "@property", "@classmethod", "@staticmethod", "@abstractmethod", "@override", "@Override")) for d in c.decorators):
            continue
        if corpus_tokens[n] <= 1:
            out.append({"symbol": c.qualname, "file": c.file_path, "lines": f"{c.start_line}-{c.end_line}",
                        "length": c.n_lines})
    out.sort(key=lambda r: -r["length"])
    return out[:limit]


# ============================================================================ imports / module graph
_JS_IMPORT = re.compile(r"""(?:import\s+(?:[^'"]*?\s+from\s+)?|require\(\s*|import\(\s*|export\s+[^'"]*?\s+from\s+)['"]([^'"]+)['"]""")
_GO_IMPORT_BLOCK = re.compile(r"import\s*\(([^)]*)\)", re.S)
_JAVA_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.M)
_C_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
_RUST_USE = re.compile(r"^\s*(?:pub\s+)?(?:use|mod)\s+(?:crate::)?(\w+)", re.M)


def python_imports(text: str) -> list[tuple[str, int]]:
    """Return ``(module, level)`` pairs; level>0 means a relative import."""
    out: list[tuple[str, int]] = []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.extend((a.name, 0) for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level and not base:
                out.extend((a.name, n.level) for a in n.names)
            else:
                out.append((base, n.level))
    return out


def _py_module_map(index: RepoIndex) -> dict[str, str]:
    mods: dict[str, str] = {}
    for f in index.files:
        if f.language != "python":
            continue
        p = posix(f.path)[:-3]
        if p.endswith("/__init__"):
            p = p[: -len("/__init__")]
        mods[p.replace("/", ".")] = f.path
        # support src/ layouts: also register without the first directory
        if "/" in p:
            mods.setdefault(p.split("/", 1)[1].replace("/", "."), f.path)
    return mods


def build_import_graph(index: RepoIndex) -> dict[str, set[str]]:
    """File -> set of repo files it imports (best-effort, internal imports only)."""
    graph: dict[str, set[str]] = defaultdict(set)
    files = {f.path for f in index.files}
    py_mods = _py_module_map(index)
    for f in index.files:
        try:
            text = index.read_file(f.path)
        except Exception:
            continue
        here = posix(f.path)
        folder = here.rsplit("/", 1)[0] if "/" in here else ""
        if f.language == "python":
            pkg_parts = folder.split("/") if folder else []
            for mod, level in python_imports(text):
                if level:
                    base = pkg_parts[: max(0, len(pkg_parts) - (level - 1))]
                    full = ".".join(base + ([mod] if mod else []))
                else:
                    full = mod
                parts = full.split(".")
                for i in range(len(parts), 0, -1):
                    hit = py_mods.get(".".join(parts[:i]))
                    if hit and hit != f.path:
                        graph[f.path].add(hit)
                        break
        elif f.language in ("javascript", "typescript"):
            for spec in _JS_IMPORT.findall(text):
                if not spec.startswith("."):
                    continue
                target = posix(str(__import__("os").path.normpath(f"{folder}/{spec}"))) if folder else posix(__import__("os").path.normpath(spec))
                for cand in (target, *(f"{target}{e}" for e in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")),
                             *(f"{target}/index{e}" for e in (".js", ".ts", ".tsx", ".jsx"))):
                    if cand in files and cand != f.path:
                        graph[f.path].add(cand)
                        break
        elif f.language in ("c", "cpp"):
            for inc in _C_INCLUDE.findall(text):
                for cand in (posix(__import__("os").path.normpath(f"{folder}/{inc}")), inc):
                    if cand in files and cand != f.path:
                        graph[f.path].add(cand)
                        break
        elif f.language == "java":
            by_suffix = {p: p for p in files if p.endswith(".java")}
            for imp in _JAVA_IMPORT.findall(text):
                suffix = imp.replace(".", "/") + ".java"
                for p in by_suffix:
                    if p.endswith(suffix) and p != f.path:
                        graph[f.path].add(p)
                        break
    return graph


def module_graph_mermaid(index: RepoIndex, max_nodes: int = 35, by_folder: bool = False) -> str:
    graph = build_import_graph(index)
    if by_folder:
        agg: dict[str, set[str]] = defaultdict(set)
        for a, bs in graph.items():
            fa = posix(a).rsplit("/", 1)[0] if "/" in a else "(root)"
            for b in bs:
                fb = posix(b).rsplit("/", 1)[0] if "/" in b else "(root)"
                if fa != fb:
                    agg[fa].add(fb)
        graph = agg
    degree: Counter = Counter()
    for a, bs in graph.items():
        degree[a] += len(bs)
        for b in bs:
            degree[b] += 1
    keep = {n for n, _ in degree.most_common(max_nodes)}
    if not keep:
        return ""
    lines = ["flowchart LR"]
    for n in sorted(keep):
        lines.append(f'  {_mm_id(n)}["{_mm_label(n)}"]')
    for a in sorted(keep):
        for b in sorted(graph.get(a, ())):
            if b in keep:
                lines.append(f"  {_mm_id(a)} --> {_mm_id(b)}")
    return "\n".join(lines)


def most_imported(index: RepoIndex, top: int = 10) -> list[tuple[str, int]]:
    cnt: Counter = Counter()
    for bs in build_import_graph(index).values():
        cnt.update(bs)
    return cnt.most_common(top)


# ============================================================================ entry points & routes
_ROUTE_PATTERNS = [
    ("flask/fastapi", re.compile(r"""@\s*(?P<obj>\w+)\.(?P<verb>get|post|put|delete|patch|route|head|options|websocket)\s*\(\s*(?P<q>['"])(?P<path>[^'"]*)(?P=q)(?P<rest>[^)]*)""", re.I)),
    ("express", re.compile(r"""\b(?P<obj>app|router|api|server|route)\.(?P<verb>get|post|put|delete|patch|all|use)\s*\(\s*(?P<q>['"`])(?P<path>[^'"`]*)(?P=q)""", re.I)),
    ("spring", re.compile(r"""@(?P<verb>Get|Post|Put|Delete|Patch|Request)Mapping\s*\(\s*(?:value\s*=\s*)?\{?\s*(?P<q>")(?P<path>[^"]*)(?P=q)""")),
    ("gin/echo", re.compile(r"""\b(?P<obj>\w+)\.(?P<verb>GET|POST|PUT|DELETE|PATCH|Any)\s*\(\s*(?P<q>")(?P<path>[^"]*)(?P=q)""")),
    ("go-http", re.compile(r"""\b(?:http|mux|r)\.(?P<verb>HandleFunc|Handle)\s*\(\s*(?P<q>")(?P<path>[^"]*)(?P=q)""")),
    ("django", re.compile(r"""\b(?P<verb>path|re_path)\s*\(\s*(?P<q>r?['"])(?P<path>[^'"]*)['"]""")),
]


def find_routes(index: RepoIndex, limit: int = 300) -> list[dict]:
    out = []
    for f in index.files:
        if f.language not in ("python", "javascript", "typescript", "java", "go", "kotlin", "csharp"):
            continue
        if is_test_path(f.path):
            continue
        try:
            text = index.read_file(f.path)
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for fw, pat in _ROUTE_PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                verb = m.group("verb").upper()
                if fw == "express" and verb == "USE" and not m.group("path").startswith("/"):
                    continue
                if fw == "flask/fastapi" and m.group("obj").lower() in ("self", "cls"):
                    continue
                out.append({"method": verb, "path": m.group("path") or "/", "file": f.path, "line": i,
                            "framework": fw})
                break
            if len(out) >= limit:
                return out
    return out


def find_entry_points(index: RepoIndex) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, path: str, line: int, note: str = "") -> None:
        if (kind, path) in seen:
            return
        seen.add((kind, path))
        out.append({"kind": kind, "file": path, "line": line, "note": note})

    names = {f.path.lower(): f.path for f in index.files}
    for f in index.files:
        base = posix(f.path).rsplit("/", 1)[-1].lower()
        depth = posix(f.path).count("/")
        if base in ("__main__.py", "main.py", "app.py", "manage.py", "wsgi.py", "asgi.py", "cli.py", "run.py",
                    "server.py", "index.js", "server.js", "app.js", "main.js", "index.ts", "main.ts", "server.ts",
                    "main.go", "main.rs", "main.c", "main.cpp", "program.cs", "application.java", "main.java") and depth <= 2:
            add("conventional entry file", f.path, 1, base)
    for c in index.chunks:
        if c.symbol_type in ("function", "method") and c.symbol_name == "main" and not is_test_path(c.file_path):
            add("main()", c.file_path, c.start_line, c.signature[:80])
        if c.language == "python" and c.symbol_type == "module" and re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", c.code):
            add("__main__ guard", c.file_path, c.start_line)
        if c.language == "java" and "public static void main" in c.code and c.part == 0:
            add("main()", c.file_path, c.start_line)
    for key in ("pyproject.toml", "package.json", "setup.py", "setup.cfg"):
        path = names.get(key)
        if not path:
            continue
        try:
            text = index.read_file(path)
        except Exception:
            continue
        if key == "package.json":
            try:
                data = json.loads(text)
                for k, v in (data.get("scripts") or {}).items():
                    if k in ("start", "dev", "serve", "build", "test"):
                        add(f"npm script: {k}", path, 1, str(v)[:80])
                if data.get("main"):
                    add("package main", path, 1, str(data["main"]))
                bins = data.get("bin")
                if bins:
                    add("package bin", path, 1, str(bins)[:80])
            except (ValueError, AttributeError):
                pass
        else:
            for m in re.finditer(r"^\s*([\w.-]+)\s*=\s*['\"]([\w.]+:[\w.]+)['\"]", text, re.M):
                add("console script", path, text[:m.start()].count("\n") + 1, f"{m.group(1)} → {m.group(2)}")
    return out[:40]


# ============================================================================ dependencies
def detect_dependencies(index: RepoIndex) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {}
    names = {posix(f.path): f.path for f in index.files}
    for rel in names:
        base = rel.rsplit("/", 1)[-1]
        if base not in ("requirements.txt", "package.json", "pyproject.toml", "go.mod", "Cargo.toml", "pom.xml",
                        "build.gradle", "Gemfile", "composer.json", "Pipfile"):
            continue
        if rel.count("/") > 2:
            continue
        try:
            text = index.read_file(rel)
        except Exception:
            continue
        found: list[str] = []
        if base == "requirements.txt":
            found = [re.split(r"[<>=!~\[; ]", ln.strip(), 1)[0] for ln in text.splitlines()
                     if ln.strip() and not ln.strip().startswith(("#", "-"))]
        elif base == "package.json":
            try:
                data = json.loads(text)
                found = list((data.get("dependencies") or {}).keys()) + [f"{k} (dev)" for k in (data.get("devDependencies") or {})]
            except ValueError:
                pass
        elif base == "pyproject.toml":
            m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
            if m:
                found = [re.split(r"[<>=!~\[; ]", s.strip(" \"',\n"), 1)[0] for s in m.group(1).split(",") if s.strip(" \"'\n")]
            found += re.findall(r"^([A-Za-z0-9_.-]+)\s*=\s*['\"{]", text.split("[tool.poetry.dependencies]")[-1].split("[")[0], re.M) if "[tool.poetry.dependencies]" in text else []
        elif base == "go.mod":
            found = re.findall(r"^\s*([\w./-]+\.[\w./-]+)\s+v[\w.+-]+", text, re.M)
        elif base == "Cargo.toml":
            sect = text.split("[dependencies]")[-1].split("\n[")[0] if "[dependencies]" in text else ""
            found = re.findall(r"^([A-Za-z0-9_-]+)\s*=", sect, re.M)
        elif base == "pom.xml":
            found = re.findall(r"<artifactId>([^<]+)</artifactId>", text)
        elif base == "Gemfile":
            found = re.findall(r"^\s*gem\s+['\"]([^'\"]+)", text, re.M)
        if found:
            deps[rel] = [d for d in found if d][:60]
    return deps


def detect_tech_stack(index: RepoIndex) -> list[str]:
    stack = [lang for lang, _ in list(index.languages.items())[:4] if lang not in ("markdown", "text", "json", "yaml")]
    blob = " ".join(d.lower() for ds in detect_dependencies(index).values() for d in ds)
    known = {"django": "Django", "flask": "Flask", "fastapi": "FastAPI", "streamlit": "Streamlit", "react": "React",
             "vue": "Vue", "next": "Next.js", "express": "Express", "spring": "Spring", "pytest": "pytest",
             "jest": "Jest", "numpy": "NumPy", "pandas": "pandas", "torch": "PyTorch", "tensorflow": "TensorFlow",
             "sqlalchemy": "SQLAlchemy", "gin": "Gin", "tokio": "Tokio", "angular": "Angular", "svelte": "Svelte"}
    for k, v in known.items():
        if re.search(rf"\b{k}", blob):
            stack.append(v)
    files = {posix(f.path).rsplit("/", 1)[-1].lower() for f in index.files}
    if "dockerfile" in files:
        stack.append("Docker")
    if any(f.path.startswith(".github/workflows") for f in index.files):
        stack.append("GitHub Actions")
    return list(dict.fromkeys(stack))


# ============================================================================ security scan
SECRET_RULES: list[tuple[str, str, re.Pattern]] = [
    ("critical", "AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("critical", "Private key block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("critical", "GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("critical", "Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("critical", "Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("critical", "Stripe live key", re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b")),
    ("high", "Hard-coded credential", re.compile(r"""(?i)\b\w*(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)\w*["']?\s*[:=]\s*['"](?!\s*['"])(?!.*(?:\$\{|\{\{|<|your|example|changeme|xxx|\*\*\*|placeholder|dummy|todo|none|null|environ|getenv))[^'"\s]{8,}['"]""")),
    ("high", "Credentials in URL", re.compile(r"[a-z]+://[^/\s:@'\"]+:[^/\s:@'\"]{3,}@[\w.-]+")),
    ("high", "Shell injection risk", re.compile(r"\bos\.system\s*\(|subprocess\.\w+\([^)]*shell\s*=\s*True")),
    ("high", "Unsafe deserialization", re.compile(r"\bpickle\.loads?\s*\(|\byaml\.load\s*\((?![^)]*Loader\s*=\s*(?:yaml\.)?SafeLoader)|\bmarshal\.loads?\s*\(")),
    ("high", "eval/exec of dynamic code", re.compile(r"(?<![\w.])(?:eval|exec)\s*\((?!\s*['\"]\s*['\"]?\))")),
    ("high", "SQL built with string formatting", re.compile(r"""(?i)(?:execute|query|raw)\s*\(\s*(?:f['"]|['"][^'"]*(?:select|insert|update|delete)[^'"]*['"]\s*(?:%|\+|\.format))""")),
    ("medium", "TLS verification disabled", re.compile(r"verify\s*=\s*False|rejectUnauthorized\s*:\s*false|InsecureSkipVerify\s*:\s*true|CURLOPT_SSL_VERIFYPEER\s*,\s*(?:0|false)", re.I)),
    ("medium", "Weak hash (MD5/SHA1)", re.compile(r"hashlib\.(?:md5|sha1)\s*\(|createHash\(\s*['\"](?:md5|sha1)['\"]")),
    ("medium", "Debug mode enabled", re.compile(r"\bdebug\s*=\s*True\b|app\.run\([^)]*debug\s*=\s*True", re.I)),
    ("medium", "innerHTML / dangerouslySetInnerHTML", re.compile(r"\.innerHTML\s*=|dangerouslySetInnerHTML")),
    ("low", "Wildcard CORS", re.compile(r"""(?i)allow[_-]?origins?["']?\s*[:=]\s*\[?\s*["']\*["']|Access-Control-Allow-Origin["']?\s*[:,]\s*["']\*""")),
    ("low", "Bare except", re.compile(r"^\s*except\s*:\s*(?:#.*)?$")),
]
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_SECRET_SKIP_LANGS = {"markdown", "text"}


def _mask(line: str) -> str:
    return re.sub(r"""(['"])([^'"\s]{6})[^'"\s]{4,}(['"])""", r"\1\2…\3", line.strip())[:160]


def security_scan(index: RepoIndex, limit: int = 300) -> list[dict]:
    findings: list[dict] = []
    for f in index.files:
        try:
            text = index.read_file(f.path)
        except Exception:
            continue
        is_test = is_test_path(f.path)
        is_example = bool(re.search(r"(\.example|\.sample|\.template)$|(^|/)(examples?|docs?|fixtures?)/", f.path))
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > 2000:
                continue
            for sev, title, pat in SECRET_RULES:
                if not pat.search(line):
                    continue
                if f.language in _SECRET_SKIP_LANGS and sev != "critical":
                    continue
                if title == "Bare except" and f.language != "python":
                    continue
                if title == "Hard-coded credential" and f.language not in ("python", "javascript", "typescript", "java", "go", "ruby", "php", "csharp", "kotlin", "shell", "yaml", "json", "ini", "toml", "text"):
                    continue
                eff = sev
                if (is_test or is_example) and sev in ("high", "medium"):
                    eff = "low"
                findings.append({"severity": eff, "issue": title, "file": f.path, "line": i,
                                 "snippet": _mask(line) if "credential" in title.lower() or sev == "critical" else line.strip()[:160],
                                 "context": "test/example" if (is_test or is_example) else ""})
    findings.sort(key=lambda r: (SEVERITY_ORDER[r["severity"]], r["file"], r["line"]))
    return findings[:limit]


def security_score(findings: list[dict]) -> int:
    penalty = sum({"critical": 25, "high": 8, "medium": 3, "low": 1}[f["severity"]] for f in findings)
    return max(0, 100 - penalty)


# ============================================================================ docs coverage / quality
def docstring_coverage(index: RepoIndex) -> dict:
    total = documented = 0
    missing: list[dict] = []
    for c in index.chunks:
        if c.symbol_type in CODE_SYMBOL_TYPES and c.part == 0 and not is_test_path(c.file_path):
            if c.symbol_name.startswith("_") and not c.symbol_name.startswith("__init__"):
                continue
            total += 1
            if c.docstring.strip():
                documented += 1
            elif len(missing) < 40 and c.n_lines >= 8:
                missing.append({"symbol": c.qualname, "file": c.file_path, "lines": f"{c.start_line}-{c.end_line}"})
    return {"total": total, "documented": documented, "pct": round(100 * documented / total, 1) if total else 0.0,
            "missing": missing}


def folder_tree(index: RepoIndex) -> dict:
    tree: dict = {}
    for f in sorted(index.files, key=lambda x: x.path):
        node = tree
        parts = posix(f.path).split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = f
    return tree


def folder_summary(index: RepoIndex, depth: int = 2, max_lines: int = 60) -> str:
    counts: Counter = Counter()
    for f in index.files:
        parts = posix(f.path).split("/")
        counts["/".join(parts[:depth]) if len(parts) > depth else ("/".join(parts[:-1]) or ".")] += 1
    return "\n".join(f"{k}/  ({v} files)" for k, v in counts.most_common(max_lines))
