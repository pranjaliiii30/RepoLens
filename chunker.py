"""Code-aware chunking.

Strategy per file type
----------------------
* Python            -> ``ast``: one chunk per function / async function / method, a *summary* chunk per
                       class (header + docstring + leading attributes + member list) and contiguous
                       module-level runs (imports, constants, ``__main__`` blocks).
* JS/TS/Java/Go/C/C++/Rust/C# -> tree-sitter when installed; otherwise a brace-matching extractor that
                       understands strings and comments. Both produce the same chunk shapes.
* Markdown          -> one chunk per heading section.
* everything else   -> sliding window (60 lines, 10 overlap) that keeps real line numbers.

Symbols longer than ``LONG_SYMBOL_LINES`` are split into parts that share the parent symbol name and keep
correct absolute line ranges. Every chunk's ``code`` is exactly ``lines[start-1:end]`` of the source file.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from . import config
from .models import Chunk, make_chunk_id

log = logging.getLogger("repolens.chunker")

BRACE_LANGS = {"javascript", "typescript", "java", "go", "rust", "csharp", "c", "cpp"}
TREE_SITTER_NAMES = {"javascript": "javascript", "typescript": "typescript", "java": "java", "go": "go",
                     "rust": "rust", "csharp": "c_sharp", "c": "c", "cpp": "cpp"}
CONFIG_LANGS = {"json", "yaml", "toml", "ini", "xml", "hcl", "protobuf", "dockerfile", "makefile", "cmake", "groovy"}


# ----------------------------------------------------------------------------- helpers
def split_lines(text: str) -> list[str]:
    """Split like Python's ``ast`` counts lines (only \\n, \\r\\n, \\r)."""
    text = text.lstrip("﻿")
    parts = re.split(r"\r\n|\r|\n", text)
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def split_ranges(start: int, end: int, lines: list[str], window: int, overlap: int) -> list[tuple[int, int]]:
    """Split the inclusive 1-based range into overlapping windows, preferring blank-line boundaries."""
    ranges: list[tuple[int, int]] = []
    s = start
    while s <= end:
        e = min(s + window - 1, end)
        if e < end:
            for j in range(e, max(s + window // 2, e - 15), -1):
                if 0 < j <= len(lines) and not lines[j - 1].strip():
                    e = j
                    break
        ranges.append((s, e))
        if e >= end:
            break
        s = max(e - overlap + 1, s + 1)
    return ranges


_COMPLEX_RE = re.compile(r"\b(?:if|for|while|case|catch|foreach|elif|except|when|guard)\b|&&|\|\||\?(?![?.:])")
_CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
_CALL_KEYWORDS = frozenset(
    """if for while switch catch return sizeof typeof new delete throw function def class else elif
    foreach using lock when match with assert await yield print range len super""".split()
)


def _approx_complexity(code: str) -> int:
    return 1 + len(_COMPLEX_RE.findall(code))


def _regex_calls(code: str) -> list[str]:
    body = code.split("\n", 1)[1] if "\n" in code else ""
    names = {m for m in _CALL_RE.findall(body) if m not in _CALL_KEYWORDS}
    return sorted(names)[:120]


class _Out:
    """Accumulates chunks for a single file."""

    def __init__(self, path: str, language: str, lines: list[str]):
        self.path = path
        self.language = language
        self.lines = lines
        self.chunks: list[Chunk] = []

    def emit(self, name: str, stype: str, start: int, end: int, *, parent: str = "", docstring: str = "",
             decorators: list[str] | None = None, signature: str = "", complexity: int = 0,
             calls: list[str] | None = None, members: list[str] | None = None) -> None:
        start = max(1, start)
        end = min(len(self.lines), end)
        if end < start or not "\n".join(self.lines[start - 1:end]).strip():
            return
        if end - start + 1 > config.LONG_SYMBOL_LINES:
            ranges = split_ranges(start, end, self.lines, config.SPLIT_WINDOW, config.SPLIT_OVERLAP)
        else:
            ranges = [(start, end)]
        total = len(ranges)
        for i, (s, e) in enumerate(ranges):
            code = "\n".join(self.lines[s - 1:e])
            if not code.strip():
                continue
            self.chunks.append(Chunk(
                id=make_chunk_id(self.path, s, e, name, i), file_path=self.path, language=self.language,
                symbol_name=name, symbol_type=stype, start_line=s, end_line=e, code=code,
                docstring=docstring if i == 0 else "", decorators=list(decorators or []), parent=parent,
                signature=signature, part=i, parts=total, complexity=complexity if i == 0 else 0,
                calls=list(calls or []) if i == 0 else [], members=list(members or []) if i == 0 else [],
            ))


def _trim_blank(lines: list[str], lo: int, hi: int) -> tuple[int, int]:
    while lo <= hi and not lines[lo - 1].strip():
        lo += 1
    while hi >= lo and not lines[hi - 1].strip():
        hi -= 1
    return lo, hi


# ----------------------------------------------------------------------------- Python
_PY_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _py_start(node: ast.AST, lines: list[str]) -> int:
    start = node.lineno  # type: ignore[attr-defined]
    decs = getattr(node, "decorator_list", [])
    if decs:
        start = min(start, min(d.lineno for d in decs))
    indent = len(lines[start - 1]) - len(lines[start - 1].lstrip())
    while start > 1:
        prev = lines[start - 2]
        if prev.lstrip().startswith("#") and (len(prev) - len(prev.lstrip())) == indent:
            start -= 1
        else:
            break
    return start


def _py_signature(node: ast.AST, lines: list[str]) -> str:
    first = node.body[0].lineno if getattr(node, "body", None) else node.lineno  # type: ignore[attr-defined]
    s = node.lineno  # type: ignore[attr-defined]
    end = max(s, first - 1) if first > s else s
    parts = [ln.strip() for ln in lines[s - 1:end] if not ln.strip().startswith("#")]
    return " ".join(parts)[:300]


def _py_complexity(node: ast.AST) -> int:
    c = 1
    match_case = getattr(ast, "match_case", None)
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)):
            c += 1
        elif isinstance(n, ast.BoolOp):
            c += len(n.values) - 1
        elif isinstance(n, ast.comprehension):
            c += 1 + len(n.ifs)
        elif match_case is not None and isinstance(n, match_case):
            c += 1
    return c


def _py_calls(node: ast.AST) -> list[str]:
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return sorted(names)[:120]


def _py_decorators(node: ast.AST) -> list[str]:
    out = []
    for d in getattr(node, "decorator_list", []):
        try:
            out.append("@" + ast.unparse(d))
        except Exception:  # pragma: no cover
            out.append("@<decorator>")
    return out


def _py_function(out: _Out, node: ast.AST, parent: str) -> None:
    out.emit(node.name, "method" if parent else "function",  # type: ignore[attr-defined]
             _py_start(node, out.lines), node.end_lineno,  # type: ignore[attr-defined]
             parent=parent, docstring=ast.get_docstring(node) or "",  # type: ignore[arg-type]
             decorators=_py_decorators(node), signature=_py_signature(node, out.lines),
             complexity=_py_complexity(node), calls=_py_calls(node))


def _py_class(out: _Out, node: ast.ClassDef, parent: str) -> None:
    qual = f"{parent}.{node.name}" if parent else node.name
    lines = out.lines
    start = _py_start(node, lines)
    first_body = node.body[0].lineno if node.body else node.lineno
    header_end = max(node.lineno, first_body - 1) if first_body > node.lineno else node.lineno
    summary_end = header_end
    for stmt in node.body:
        if isinstance(stmt, _PY_DEFS):
            break
        summary_end = max(summary_end, stmt.end_lineno or stmt.lineno)
    members = [s.name for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
    out.emit(node.name, "class", start, summary_end, parent=parent, docstring=ast.get_docstring(node) or "",
             decorators=_py_decorators(node), signature=_py_signature(node, lines), members=members)

    run: list[ast.stmt] = []
    seen_def = False

    def flush_run() -> None:
        if run and seen_def:
            lo, hi = _trim_blank(lines, run[0].lineno, run[-1].end_lineno or run[-1].lineno)
            out.emit("<class body>", "block", lo, hi, parent=qual)
        run.clear()

    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_run()
            seen_def = True
            _py_function(out, stmt, qual)
        elif isinstance(stmt, ast.ClassDef):
            flush_run()
            seen_def = True
            _py_class(out, stmt, qual)
        elif seen_def:
            run.append(stmt)
    flush_run()


def _py_chunks(out: _Out, text: str) -> None:
    tree = ast.parse(text)
    lines = out.lines
    module_doc = ast.get_docstring(tree) or ""
    prev_end = 0
    run: list[ast.stmt] = []
    first_run = True

    def flush(hi: int) -> None:
        nonlocal first_run
        if not run:
            return
        lo, hi2 = _trim_blank(lines, prev_end + 1, hi)
        out.emit("<module>", "module", lo, hi2, docstring=module_doc if first_run else "")
        first_run = False
        run.clear()

    for node in tree.body:
        if isinstance(node, _PY_DEFS):
            start = _py_start(node, lines)
            flush(start - 1)
            if isinstance(node, ast.ClassDef):
                _py_class(out, node, "")
            else:
                _py_function(out, node, "")
            prev_end = node.end_lineno or node.lineno
        else:
            run.append(node)
    flush(len(lines))


# ----------------------------------------------------------------------------- brace languages
def _brace_depths(lines: list[str]) -> tuple[list[int], list[int]]:
    """Per-line brace depth at line start and max depth reached on the line (strings/comments aware)."""
    n = len(lines)
    d_start = [0] * (n + 1)
    d_max = [0] * n
    depth = 0
    in_block = False
    in_tpl = False
    for i, line in enumerate(lines):
        d_start[i] = depth
        mx = depth
        j, length = 0, len(line)
        while j < length:
            ch = line[j]
            if in_block:
                if ch == "*" and line.startswith("*/", j):
                    in_block = False
                    j += 2
                else:
                    j += 1
                continue
            if in_tpl:
                if ch == "\\":
                    j += 2
                    continue
                if ch == "`":
                    in_tpl = False
                j += 1
                continue
            if ch == "/" and line.startswith("//", j):
                break
            if ch == "/" and line.startswith("/*", j):
                in_block = True
                j += 2
                continue
            if ch == "`":
                in_tpl = True
                j += 1
                continue
            if ch in "\"'":
                k = j + 1
                while k < length:
                    if line[k] == "\\":
                        k += 2
                        continue
                    if line[k] == ch:
                        break
                    k += 1
                j = k + 1 if k < length else j + 1
                continue
            if ch == "{":
                depth += 1
                mx = max(mx, depth)
            elif ch == "}":
                depth = max(0, depth - 1)
            j += 1
        d_max[i] = mx
    d_start[n] = depth
    return d_start, d_max


def _block_end(i: int, d_start: list[int], d_max: list[int], lines: list[str], lookahead: int = 10) -> int | None:
    """0-based index of the line that closes the block opened at/after line ``i`` (or None)."""
    base = d_start[i]
    started = False
    for k in range(i, len(lines)):
        if d_max[k] > base:
            started = True
        if started and d_start[k + 1] <= base:
            return k
        if not started:
            if k - i >= lookahead:
                return None
            if ";" in lines[k]:
                return None
    return None


_KEYWORDS = frozenset(
    """if for while switch return else do catch sizeof typedef using new delete throw case goto try finally
    function foreach lock when elseif synchronized await yield assert""".split()
)
_CLIKE_COMMENT_PREFIXES = ("//", "/*", "*", "*/", "///")


@dataclass
class _Rules:
    class_re: re.Pattern | None = None
    func_res: list[tuple[re.Pattern, str]] = field(default_factory=list)
    member_res: list[re.Pattern] = field(default_factory=list)       # only valid in class scope
    containers: list[tuple[re.Pattern, bool]] = field(default_factory=list)  # (pattern, sets_parent)
    validate: Callable[[re.Match, str, str], bool] | None = None
    attr_prefixes: tuple[str, ...] = ("@",)


def _java_like_rules(extra_mods: str = "", kinds: str = "class|interface|enum|record|@interface") -> _Rules:
    mods = (r"(?:(?:public|protected|private|abstract|static|final|sealed|non-sealed|strictfp|synchronized|native|"
            r"default|transient|volatile|internal|partial|override|virtual|async|extern|unsafe|readonly" + extra_mods + r")\s+)")
    cls = re.compile(rf"^\s*{mods}*(?:{kinds})\s+(?P<name>\w+)")
    typed = re.compile(rf"^\s*{mods}*(?:<[^>]+>\s+)?[\w.<>\[\],?&\s]+?\s+(?P<name>\w+)\s*(?:<[^>(]*>)?\s*\(")
    ctor = re.compile(rf"^\s*{mods}+(?P<name>\w+)\s*\(")

    def validate(m: re.Match, line: str, parent: str) -> bool:
        name = m.group("name")
        prefix = line[: m.start("name")]
        if name in _KEYWORDS or "=" in prefix or "(" in prefix:
            return False
        if re.search(r"\b(return|new|throw|else|case|yield|assert|await)\b", prefix):
            return False
        if m.re is ctor and name != parent.split(".")[-1]:
            return False
        return True

    containers = [(re.compile(r"^\s*namespace\s+(?P<name>[\w.]+)"), False)]
    return _Rules(class_re=cls, func_res=[(typed, "func"), (ctor, "func")], containers=containers, validate=validate)


_RULES: dict[str, _Rules] = {
    "javascript": _Rules(
        class_re=re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(?P<name>[A-Za-z_$][\w$]*)"),
        func_res=[
            (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)"), "func"),
            (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::\s*[^=]+?)?=>|[A-Za-z_$][\w$]*\s*=>)"), "arrow"),
        ],
        member_res=[re.compile(r"^\s+(?:(?:public|private|protected|static|readonly|override|abstract|async|get|set|declare)\s+)*\*?\s*(?P<name>[A-Za-z_$#][\w$]*)\s*(?:<[^>(]*>)?\s*\(")],
        validate=lambda m, line, parent: m.group("name") not in _KEYWORDS,
    ),
    "java": _java_like_rules(),
    "csharp": _java_like_rules(kinds="class|interface|enum|record|struct"),
    "go": _Rules(
        class_re=re.compile(r"^type\s+(?P<name>\w+)\s+(?:struct|interface)\b"),
        func_res=[(re.compile(r"^func\s+(?:\(\s*\w*\s*\*?\s*(?P<recv>[\w.]+)(?:\[[^\]]*\])?\s*\)\s*)?(?P<name>\w+)\s*(?:\[[^\]]*\])?\s*\("), "func")],
        attr_prefixes=(),
    ),
    "rust": _Rules(
        class_re=re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union)\s+(?P<name>\w+)"),
        func_res=[(re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+(?P<name>\w+)"), "func")],
        containers=[
            (re.compile(r"^\s*impl\b(?:<[^>]*>)?\s+(?:[\w:<>,\s&']+?\s+for\s+)?(?P<name>[A-Za-z_][\w:]*)"), True),
            (re.compile(r"^\s*(?:pub\s+)?mod\s+(?P<name>\w+)\s*\{"), False),
        ],
        attr_prefixes=("#[",),
    ),
}
_C_NOT = r"(?!\s*(?:if|for|while|switch|return|else|do|catch|sizeof|typedef|using|new|delete|throw|case|goto|namespace|class|struct|enum|union|template)\b)"
_RULES["c"] = _Rules(
    class_re=re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?(?:typedef\s+)?(?:class|struct|union|enum(?:\s+class)?)\s+(?:\w+\s+)?(?P<name>\w+)[^;(]*$"),
    func_res=[(re.compile(rf"^{_C_NOT}\s*(?:[\w:<>,~*&\[\]\s]+?[\s*&]+)?(?P<name>(?:\w+::)*~?\w+)\s*\([^;]*$"), "func")],
    containers=[(re.compile(r"^\s*namespace\s*(?P<name>\w*)\s*\{?"), False), (re.compile(r'^\s*extern\s+"C(?:\+\+)?"\s*(?P<name>)\{'), False)],
    validate=lambda m, line, parent: m.group("name").split("::")[-1] not in _KEYWORDS,
    attr_prefixes=(),
)
_RULES["cpp"] = _RULES["c"]
_RULES["typescript"] = _RULES["javascript"]


def _clean_comment(raw: list[str]) -> str:
    out = []
    for ln in raw:
        s = ln.strip()
        if s.startswith(("@", "#[")):
            continue
        s = re.sub(r"^(/\*\*?|\*/|\*|///?|//!|#)\s?", "", s)
        s = re.sub(r"\*/\s*$", "", s).strip()
        if s:
            out.append(s)
    return "\n".join(out)


def _brace_extract(out: _Out, rules: _Rules, d_start: list[int], d_max: list[int]) -> None:
    lines = out.lines
    n = len(lines)
    covered = [False] * n

    def mark(s1: int, e1: int) -> None:
        for x in range(max(0, s1 - 1), min(n, e1)):
            covered[x] = True

    def lead_start(i0: int) -> tuple[int, str]:
        """Walk upward over doc comments/annotations directly above line ``i0`` (0-based)."""
        s = i0
        comment_lines: list[str] = []
        while s > 0 and not covered[s - 1]:
            stripped = lines[s - 1].strip()
            if stripped.startswith(_CLIKE_COMMENT_PREFIXES) or stripped.startswith(rules.attr_prefixes or ("\0",)):
                comment_lines.insert(0, stripped)
                s -= 1
            else:
                break
        return s + 1, _clean_comment(comment_lines)

    def is_comment_or_blank(line: str) -> bool:
        s = line.strip()
        return not s or s.startswith(_CLIKE_COMMENT_PREFIXES)

    def signature_of(i0: int, end0: int) -> str:
        parts = []
        for k in range(i0, min(end0, i0 + 6) + 1):
            parts.append(lines[k].strip())
            if "{" in lines[k]:
                break
        return " ".join(parts)[:300]

    def scan(lo: int, hi: int, depth: int, parent: str) -> None:
        i = lo
        while i <= hi:
            if covered[i] or d_start[i] != depth or is_comment_or_blank(lines[i]):
                i += 1
                continue
            line = lines[i]

            # containers (namespace / impl / mod / extern "C")
            handled = False
            for pat, sets_parent in rules.containers:
                m = pat.match(line)
                if m:
                    end = _block_end(i, d_start, d_max, lines)
                    if end is not None:
                        name = m.group("name") or ""
                        scan(i + 1, end - 1, depth + 1, name if (sets_parent and name) else parent)
                        mark(i + 1, end + 1)
                        i = end + 1
                        handled = True
                    break
            if handled:
                continue

            # classes / structs / interfaces
            m = rules.class_re.match(line) if rules.class_re else None
            if m and (rules.validate is None or "(" not in line[: m.end()]):
                end = _block_end(i, d_start, d_max, lines)
                if end is not None:
                    name = m.group("name")
                    qual = f"{parent}.{name}" if parent else name
                    start1, doc = lead_start(i)
                    saved, out.chunks = out.chunks, []
                    scan(i + 1, end - 1, depth + 1, qual)
                    children, out.chunks = out.chunks, saved
                    first_child = min((c.start_line for c in children), default=None)
                    summary_end = max((first_child - 1) if first_child else end + 1, i + 1)
                    members = [c.symbol_name for c in children
                               if c.symbol_type in ("method", "function") and c.parent == qual]
                    out.emit(name, "class", start1, summary_end, parent=parent, docstring=doc,
                             signature=signature_of(i, end), members=members)
                    out.chunks.extend(children)
                    mark(start1, end + 1)
                    i = end + 1
                    continue

            # functions / methods
            candidates = list(rules.func_res)
            if parent:
                candidates += [(p, "func") for p in rules.member_res]
            for pat, kind in candidates:
                m = pat.match(line)
                if not m:
                    continue
                if rules.validate is not None and not rules.validate(m, line, parent):
                    continue
                name = m.group("name")
                recv = m.groupdict().get("recv") or ""
                fn_parent = parent
                if "::" in name:
                    *owner, name = name.split("::")
                    fn_parent = "::".join(owner) if owner else parent
                if recv:
                    fn_parent = recv
                end = _block_end(i, d_start, d_max, lines)
                if end is None and kind == "arrow":
                    end = i
                    while end + 1 < n and ";" not in lines[end] and end - i < 12 and lines[end + 1].strip():
                        end += 1
                if end is None:
                    continue
                start1, doc = lead_start(i)
                code = "\n".join(lines[i:end + 1])
                out.emit(name, "method" if fn_parent else "function", start1, end + 1, parent=fn_parent,
                         docstring=doc, signature=signature_of(i, end),
                         complexity=_approx_complexity(code), calls=_regex_calls(code))
                mark(start1, end + 1)
                i = end
                break
            i += 1

    scan(0, n - 1, 0, "")

    # leftover module-level code (imports, constants, scripts) as contiguous runs
    i = 0
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not covered[j + 1]:
            j += 1
        lo, hi = _trim_blank(lines, i + 1, j + 1)
        if lo <= hi and not all(is_comment_or_blank(lines[k - 1]) for k in range(lo, hi + 1)):
            out.emit("<module>", "module", lo, hi)
        i = j + 1


# ----------------------------------------------------------------------------- tree-sitter (optional)
_TS_CLASS_TYPES = {"class_declaration", "class_definition", "interface_declaration", "enum_declaration",
                   "struct_specifier", "class_specifier", "struct_item", "enum_item", "trait_item",
                   "record_declaration", "struct_declaration", "abstract_class_declaration"}
_TS_FUNC_TYPES = {"function_declaration", "function_definition", "method_definition", "method_declaration",
                  "function_item", "constructor_declaration", "generator_function_declaration"}
_TS_ID_TYPES = {"identifier", "field_identifier", "qualified_identifier", "destructor_name", "property_identifier",
                "type_identifier", "name"}
_ts_parsers: dict[str, object] = {}


def _ts_parser(language: str):
    name = TREE_SITTER_NAMES.get(language)
    if not name:
        return None
    if name in _ts_parsers:
        return _ts_parsers[name]
    parser = None
    try:
        try:
            from tree_sitter_language_pack import get_parser  # maintained successor
        except ImportError:
            from tree_sitter_languages import get_parser  # legacy package
        parser = get_parser(name)
    except Exception as exc:  # not installed / unsupported platform / API drift
        log.info("tree-sitter unavailable for %s (%s); using brace extractor", language, exc)
    _ts_parsers[name] = parser
    return parser


def _ts_node_name(node) -> str:
    named = node.child_by_field_name("name")
    if named is not None:
        return named.text.decode("utf-8", "replace")
    cur = node.child_by_field_name("declarator")
    while cur is not None:
        if cur.type in _TS_ID_TYPES:
            return cur.text.decode("utf-8", "replace")
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            for ch in cur.children:
                if ch.type in _TS_ID_TYPES:
                    return ch.text.decode("utf-8", "replace")
            break
        cur = nxt
    return ""


def _ts_extract(out: _Out, parser) -> bool:
    """Returns False when tree-sitter couldn't produce a trustworthy result (caller falls back)."""
    src = "\n".join(out.lines).encode("utf-8")
    tree = parser.parse(src)
    root = tree.root_node
    if root.has_error:
        return False
    lines = out.lines
    covered_ranges: list[tuple[int, int]] = []

    def lead(start_row: int) -> tuple[int, str]:
        s = start_row
        buf: list[str] = []
        while s > 0 and lines[s - 1].strip().startswith(_CLIKE_COMMENT_PREFIXES + ("@", "#[")):
            buf.insert(0, lines[s - 1].strip())
            s -= 1
        return s + 1, _clean_comment(buf)

    def walk(node, parent: str) -> None:
        for child in node.children:
            t = child.type
            if t in _TS_CLASS_TYPES:
                name = _ts_node_name(child) or "<anonymous>"
                qual = f"{parent}.{name}" if parent else name
                s1, doc = lead(child.start_point[0])
                e1 = child.end_point[0] + 1
                saved, out.chunks = out.chunks, []
                walk(child, qual)
                kids, out.chunks = out.chunks, saved
                first = min((c.start_line for c in kids), default=None)
                members = [c.symbol_name for c in kids if c.symbol_type in ("method", "function") and c.parent == qual]
                out.emit(name, "class", s1, max((first - 1) if first else e1, child.start_point[0] + 1),
                         parent=parent, docstring=doc, signature=lines[child.start_point[0]].strip()[:300],
                         members=members)
                out.chunks.extend(kids)
                covered_ranges.append((s1, e1))
            elif t in _TS_FUNC_TYPES:
                name = _ts_node_name(child) or "<anonymous>"
                s1, doc = lead(child.start_point[0])
                e1 = child.end_point[0] + 1
                code = "\n".join(lines[child.start_point[0]:e1])
                out.emit(name, "method" if parent else "function", s1, e1, parent=parent, docstring=doc,
                         signature=lines[child.start_point[0]].strip()[:300],
                         complexity=_approx_complexity(code), calls=_regex_calls(code))
                covered_ranges.append((s1, e1))
            elif t in ("lexical_declaration", "variable_declaration"):
                for decl in child.children:
                    value = decl.child_by_field_name("value") if decl.type == "variable_declarator" else None
                    if value is not None and value.type in ("arrow_function", "function_expression", "function"):
                        name = _ts_node_name(decl) or "<anonymous>"
                        s1, doc = lead(child.start_point[0])
                        e1 = child.end_point[0] + 1
                        code = "\n".join(lines[child.start_point[0]:e1])
                        out.emit(name, "function", s1, e1, parent=parent, docstring=doc,
                                 signature=lines[child.start_point[0]].strip()[:300],
                                 complexity=_approx_complexity(code), calls=_regex_calls(code))
                        covered_ranges.append((s1, e1))
            elif child.named_child_count:
                walk(child, parent)

    walk(root, "")
    if not out.chunks:
        return False
    covered = [False] * len(lines)
    for s, e in covered_ranges:
        for x in range(s - 1, min(e, len(lines))):
            covered[x] = True
    i = 0
    while i < len(lines):
        if covered[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(lines) and not covered[j + 1]:
            j += 1
        lo, hi = _trim_blank(lines, i + 1, j + 1)
        if lo <= hi and not all(not lines[k - 1].strip() or lines[k - 1].strip().startswith(_CLIKE_COMMENT_PREFIXES)
                                for k in range(lo, hi + 1)):
            out.emit("<module>", "module", lo, hi)
        i = j + 1
    return True


# ----------------------------------------------------------------------------- markdown & windows
def _md_chunks(out: _Out) -> None:
    lines = out.lines
    n = len(lines)
    sections: list[tuple[str, int]] = []
    in_fence = False
    for idx, ln in enumerate(lines, 1):
        if ln.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence:
            m = re.match(r"^(#{1,6})\s+(.*\S)\s*#*\s*$", ln)
            if m:
                sections.append((m.group(2), idx))
    if not sections:
        _window_chunks(out, "doc")
        return
    if sections[0][1] > 1 and "\n".join(lines[:sections[0][1] - 1]).strip():
        sections.insert(0, ("Introduction", 1))
    for k, (title, start) in enumerate(sections):
        end = (sections[k + 1][1] - 1) if k + 1 < len(sections) else n
        lo, hi = _trim_blank(lines, start, end)
        if lo <= hi:
            out.emit(title[:120], "doc", lo, hi, signature=title)


def _window_chunks(out: _Out, stype: str) -> None:
    lines = out.lines
    n = len(lines)
    base = out.path.rsplit("/", 1)[-1]
    if n <= int(config.WINDOW_LINES * 1.5):
        lo, hi = _trim_blank(lines, 1, n)
        if lo <= hi:
            out.chunks.append(Chunk(
                id=make_chunk_id(out.path, lo, hi, base, 0), file_path=out.path, language=out.language,
                symbol_name=base, symbol_type=stype, start_line=lo, end_line=hi,
                code="\n".join(lines[lo - 1:hi])))
        return
    s = 1
    while s <= n:
        e = min(s + config.WINDOW_LINES - 1, n)
        lo, hi = _trim_blank(lines, s, e)
        if lo <= hi:
            name = f"{base} (lines {lo}-{hi})"
            out.chunks.append(Chunk(
                id=make_chunk_id(out.path, lo, hi, name, 0), file_path=out.path, language=out.language,
                symbol_name=name, symbol_type=stype, start_line=lo, end_line=hi,
                code="\n".join(lines[lo - 1:hi])))
        if e >= n:
            break
        s = e - config.WINDOW_OVERLAP + 1


# ----------------------------------------------------------------------------- public API
def chunk_file(path: str, text: str, language: str, *, warnings: list[str] | None = None,
               info: dict | None = None, use_tree_sitter: bool = True) -> list[Chunk]:
    """Chunk one file. ``info['parser']`` reports which strategy was used."""
    text = text.lstrip("\ufeff")          # a UTF-8 BOM (common on Windows) makes ast.parse fail
    lines = split_lines(text)
    warnings = warnings if warnings is not None else []
    info = info if info is not None else {}
    if not any(ln.strip() for ln in lines):
        info["parser"] = "empty"
        return []
    out = _Out(path, language, lines)
    window_type = "config" if language in CONFIG_LANGS else ("doc" if language in ("text",) else "block")
    try:
        if language == "python":
            _py_chunks(out, text)
            info["parser"] = "ast"
        elif language in BRACE_LANGS:
            done = False
            if use_tree_sitter:
                parser = _ts_parser(language)
                if parser is not None:
                    try:
                        saved = out.chunks
                        out.chunks = []
                        done = _ts_extract(out, parser)
                        if not done:
                            out.chunks = saved
                        else:
                            info["parser"] = "tree-sitter"
                    except Exception as exc:  # grammar/API mismatch -> robust fallback
                        log.info("tree-sitter failed on %s: %s", path, exc)
                        out.chunks = []
            if not done:
                d_start, d_max = _brace_depths(lines)
                _brace_extract(out, _RULES[language], d_start, d_max)
                info["parser"] = "brace"
        elif language == "markdown":
            _md_chunks(out)
            info["parser"] = "markdown"
        else:
            _window_chunks(out, window_type)
            info["parser"] = "window"
    except (SyntaxError, ValueError, RecursionError) as exc:
        warnings.append(f"{path}: could not parse as {language} ({exc.__class__.__name__}); used sliding window")
        out.chunks = []
        _window_chunks(out, "block")
        info["parser"] = "window"
    if not out.chunks:
        _window_chunks(out, window_type if language not in BRACE_LANGS | {"python"} else "block")
        info.setdefault("parser", "window")
    return out.chunks


def chunk_texts(items: Iterable[tuple[str, str, str]], *, warnings: list[str] | None = None,
                use_tree_sitter: bool = True) -> list[Chunk]:
    """Convenience for tests: chunk many ``(path, text, language)`` triples."""
    result: list[Chunk] = []
    for path, text, language in items:
        result.extend(chunk_file(path, text, language, warnings=warnings, use_tree_sitter=use_tree_sitter))
    return result
