"""Chunker tests: Python AST chunking (core feature), brace-language extraction and fallbacks."""
from __future__ import annotations

from core.chunker import chunk_file, split_lines


def _chunks(path, text, lang="python", **kw):
    warnings: list[str] = []
    info: dict = {}
    out = chunk_file(path, text, lang, warnings=warnings, info=info, **kw)
    return out, warnings, info


def _assert_exact(chunks, text):
    """Core invariant: every chunk's code is exactly the source lines it claims to cover."""
    lines = split_lines(text)
    for c in chunks:
        assert 1 <= c.start_line <= c.end_line <= len(lines), c
        assert c.code == "\n".join(lines[c.start_line - 1:c.end_line]), f"{c.qualname} range mismatch"


def by_name(chunks):
    return {c.qualname: c for c in chunks}


# ----------------------------------------------------------------------------- python
PY = '''"""Module doc."""
import os

LIMIT = 10


def top(a, b=2):
    """Add things."""
    if a and b:
        return a + b
    return 0


async def fetch(url):
    return await get(url)


class Service(Base):
    """A service."""

    retries = 3

    def __init__(self, db):
        self.db = db

    @staticmethod
    def make():
        return Service(None)

    @property
    def name(self):
        return "svc"

    class Inner:
        def deep(self):
            return 1


if __name__ == "__main__":
    top(1)
'''


def test_python_extracts_functions_methods_and_classes():
    chunks, warnings, info = _chunks("pkg/mod.py", PY)
    assert info["parser"] == "ast" and not warnings
    n = by_name(chunks)
    assert n["top"].symbol_type == "function"
    assert n["fetch"].symbol_type == "function"
    assert n["Service"].symbol_type == "class"
    assert n["Service.__init__"].symbol_type == "method"
    assert n["Service.make"].parent == "Service"
    assert n["Service.Inner"].symbol_type == "class"
    assert n["Service.Inner.deep"].symbol_type == "method"
    assert n["Service.Inner.deep"].parent == "Service.Inner"
    _assert_exact(chunks, PY)


def test_python_metadata_docstring_decorators_signature():
    chunks, _, _ = _chunks("mod.py", PY)
    n = by_name(chunks)
    assert n["top"].docstring == "Add things."
    assert "def top(a, b=2)" in n["top"].signature
    assert n["Service.make"].decorators == ["@staticmethod"]
    assert n["Service.name"].decorators == ["@property"]
    assert n["Service"].docstring == "A service."
    assert n["Service.make"].start_line == PY.splitlines().index("    @staticmethod") + 1  # decorator included


def test_python_class_summary_chunk_lists_members_and_excludes_method_bodies():
    chunks, _, _ = _chunks("mod.py", PY)
    cls = by_name(chunks)["Service"]
    assert cls.members == ["__init__", "make", "name"]
    assert "retries = 3" in cls.code
    assert "self.db = db" not in cls.code


def test_python_module_header_and_main_guard_become_module_chunks():
    chunks, _, _ = _chunks("mod.py", PY)
    modules = [c for c in chunks if c.symbol_type == "module"]
    assert len(modules) >= 2
    assert "import os" in modules[0].code and modules[0].docstring == "Module doc."
    assert any("__main__" in m.code for m in modules)


def test_python_complexity_and_calls():
    chunks, _, _ = _chunks("mod.py", PY)
    n = by_name(chunks)
    assert n["top"].complexity >= 3          # base + if + `and`
    assert "get" in n["fetch"].calls
    assert "Service" in n["Service.make"].calls


def test_python_coverage_no_overlap_between_top_level_symbols():
    chunks, _, _ = _chunks("mod.py", PY)
    spans = sorted((c.start_line, c.end_line) for c in chunks if c.symbol_type in ("function",) and not c.parent)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 < s2


def test_python_syntax_error_falls_back_to_window_with_warning():
    text = "def broken(:\n    pass\n"
    chunks, warnings, info = _chunks("bad.py", text)
    assert info["parser"] == "window"
    assert len(chunks) == 1 and chunks[0].symbol_type == "block"
    assert warnings and "bad.py" in warnings[0]
    _assert_exact(chunks, text)


def test_python_long_function_is_split_with_correct_absolute_ranges():
    text = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(400)) + "\n"
    chunks, _, _ = _chunks("long.py", text)
    parts = [c for c in chunks if c.symbol_name == "big"]
    assert len(parts) > 1
    assert all(c.parts == len(parts) for c in parts)
    assert [c.part for c in parts] == list(range(len(parts)))
    assert parts[0].start_line == 1 and parts[-1].end_line == 401
    for a, b in zip(parts, parts[1:]):
        assert b.start_line <= a.end_line + 1          # overlap, never a gap
    assert all(c.n_lines <= 101 for c in parts)
    assert parts[0].docstring == "" and parts[0].complexity >= 1 and parts[1].complexity == 0
    _assert_exact(chunks, text)


def test_python_handles_crlf_bom_and_unicode():
    text = "﻿def grüße():\r\n    \"\"\"Hallo 世界.\"\"\"\r\n    return 'ñ'\r\n"
    chunks, warnings, _ = _chunks("u.py", text)
    assert not warnings
    c = by_name(chunks)["grüße"]
    assert c.docstring.startswith("Hallo")
    assert (c.start_line, c.end_line) == (1, 3)


def test_python_empty_and_whitespace_only_files_produce_no_chunks():
    assert _chunks("e.py", "")[0] == []
    assert _chunks("e.py", "\n\n   \n")[0] == []


def test_python_comment_directly_above_function_is_included():
    text = "x = 1\n\n# helper does a thing\ndef f():\n    return 1\n"
    f = by_name(_chunks("m.py", text)[0])["f"]
    assert f.start_line == 3


def test_chunk_ids_are_unique_and_stable():
    a, _, _ = _chunks("mod.py", PY)
    b, _, _ = _chunks("mod.py", PY)
    assert len({c.id for c in a}) == len(a)
    assert [c.id for c in a] == [c.id for c in b]


def test_embed_text_has_context_header():
    c = by_name(_chunks("auth/login.py", "class AuthService:\n    def verify_token(self):\n        return 1\n")[0])["AuthService.verify_token"]
    h = c.embed_text().splitlines()[0]
    assert h == "File: auth/login.py | Class: AuthService | Method: verify_token"


# ----------------------------------------------------------------------------- brace languages
JS = '''const express = require('express');

/**
 * Verify a bearer token.
 */
function verifyToken(req, res, next) {
  const header = req.headers["authorization"] || "{";
  if (!header) { return res.status(401).send("no token }"); }
  next();
}

const hash = (pw) => crypto.createHash('sha1').update(pw).digest('hex');

export class AuthService extends Base {
  constructor(db) {
    super();
    this.db = db;
  }

  async login(user, pw) {
    const tpl = `hello ${user} {`;
    return this.db.find(user);
  }
}
'''


def test_js_functions_arrows_classes_ignore_braces_in_strings_and_templates():
    chunks, _, info = _chunks("a.js", JS, "javascript", use_tree_sitter=False)
    assert info["parser"] == "brace"
    n = by_name(chunks)
    assert n["verifyToken"].docstring == "Verify a bearer token."
    assert n["verifyToken"].end_line - n["verifyToken"].start_line >= 6
    assert n["hash"].symbol_type == "function"
    assert n["AuthService"].members == ["constructor", "login"]
    assert n["AuthService.login"].symbol_type == "method"
    _assert_exact(chunks, JS)


def test_java_methods_constructors_nested_classes():
    text = '''package x;

/** Handles users. */
public class UserService {
    private final Repo repo = new Repo();

    public UserService(Repo r) {
        this.repo = r;
    }

    public List<User> findAll(int limit,
                              String q) throws Exception {
        if (limit > 0) { return repo.find(q); }
        return List.of();
    }

    static class Inner {
        void run() { System.out.println("hi"); }
    }
}
'''
    chunks, _, _ = _chunks("UserService.java", text, "java", use_tree_sitter=False)
    n = by_name(chunks)
    assert n["UserService"].docstring == "Handles users."
    assert {"UserService.UserService", "UserService.findAll", "UserService.Inner.run"} <= set(n)
    assert n["UserService.findAll"].start_line == 11 and n["UserService.findAll"].end_line == 15
    _assert_exact(chunks, text)


def test_go_receiver_methods_and_structs():
    text = '''package main

type Server struct {
	port int
}

// Start runs the server.
func (s *Server) Start() error {
	return nil
}

func main() {
	s := &Server{port: 8080}
	s.Start()
}
'''
    chunks, _, _ = _chunks("main.go", text, "go", use_tree_sitter=False)
    n = by_name(chunks)
    assert n["Server"].symbol_type == "class"
    assert n["Server.Start"].symbol_type == "method" and n["Server.Start"].docstring == "Start runs the server."
    assert n["main"].symbol_type == "function"


def test_rust_impl_blocks_and_c_functions():
    rs = "pub struct Config { pub port: u16 }\n\nimpl Config {\n    pub fn new(port: u16) -> Self { Config { port } }\n}\n\nfn main() {\n    let c = 1;\n}\n"
    n = by_name(_chunks("main.rs", rs, "rust", use_tree_sitter=False)[0])
    assert "Config.new" in n and n["main"].symbol_type == "function"
    c = "static int add(int a,\n               int b) {\n    return a + b;\n}\n\nint main(void) {\n    return add(1, 2);\n}\n"
    n = by_name(_chunks("main.c", c, "c", use_tree_sitter=False)[0])
    assert n["add"].start_line == 1 and n["add"].end_line == 4 and "main" in n


def test_control_flow_keywords_are_not_mistaken_for_functions():
    text = "int main(void) {\n    if (x) {\n        y();\n    }\n    while (z) { w(); }\n    return 0;\n}\n"
    names = {c.symbol_name for c in _chunks("a.c", text, "c", use_tree_sitter=False)[0] if c.symbol_type == "function"}
    assert names == {"main"}


def test_unbalanced_braces_degrade_gracefully():
    text = "function a() {\n  if (x) {\n}\n"
    chunks, _, _ = _chunks("a.js", text, "javascript", use_tree_sitter=False)
    assert chunks                                   # something is indexed, nothing crashes
    _assert_exact(chunks, text)


# ----------------------------------------------------------------------------- docs / windows
def test_markdown_sections_ignore_headings_inside_code_fences():
    text = "# Title\n\nintro\n\n## Install\n\n```bash\n# not a heading\npip install x\n```\n\n## Usage\ntext\n"
    chunks, _, info = _chunks("README.md", text, "markdown")
    assert info["parser"] == "markdown"
    assert [c.symbol_name for c in chunks] == ["Title", "Install", "Usage"]
    assert all(c.symbol_type == "doc" for c in chunks)
    _assert_exact(chunks, text)


def test_sliding_window_fallback_has_overlap_and_real_line_numbers():
    text = "\n".join(f"line {i}" for i in range(1, 201))
    chunks, _, info = _chunks("data.txt", text, "text")
    assert info["parser"] == "window"
    assert chunks[0].start_line == 1 and chunks[0].end_line == 60
    assert chunks[1].start_line == 51                       # 10-line overlap
    assert chunks[-1].end_line == 200
    _assert_exact(chunks, text)


def test_small_config_file_is_a_single_config_chunk():
    chunks, _, _ = _chunks("conf.yaml", "a: 1\nb: 2\n", "yaml")
    assert len(chunks) == 1 and chunks[0].symbol_type == "config"
