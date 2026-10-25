"""Static analysis: call graph, routes, entry points, imports, dependencies, security rules, dead code."""
from __future__ import annotations

from core import insights as I
from core.insights import SECRET_RULES


def test_stats(sample_index):
    st = I.repo_stats(sample_index)
    assert st["files"] == len(sample_index.files) and st["loc"] > 50
    assert st["test_files"] == 1 and "python" in st["languages"]


def test_call_graph_callers_and_callees(sample_index):
    assert [c.qualname for c in I.callers_of(sample_index, "hash_password")] == ["login_user"]
    assert {c.qualname for c in I.callees_of(sample_index, "login_user")} == {"hash_password", "lookup"}
    assert "login" in {c.qualname for c in I.callers_of(sample_index, "login_user")}
    assert I.callers_of(sample_index, "does_not_exist") == []


def test_call_graph_mermaid(sample_index):
    m = I.call_graph_mermaid(sample_index, focus="login_user")
    assert m.startswith("flowchart LR") and "-->" in m and "hash_password" in m
    assert I.call_graph_mermaid(sample_index, focus="nope") == ""


def test_routes_across_frameworks_ignore_tests(sample_index):
    got = {(r["method"], r["path"]) for r in I.find_routes(sample_index)}
    assert {("GET", "/users"), ("GET", "/health"), ("POST", "/api/items")} <= got
    assert ("ROUTE", "/login") in got


def test_entry_points(sample_index):
    kinds = {(e["kind"], e["file"]) for e in I.find_entry_points(sample_index)}
    assert ("__main__ guard", "app/main.py") in kinds and ("conventional entry file", "web/server.js") in kinds


def test_import_graph_resolves_relative_python_and_js_imports(sample_index):
    g = I.build_import_graph(sample_index)
    assert g["app/main.py"] >= {"app/auth.py", "app/db.py"}
    assert g["web/server.js"] == {"web/util.js"}
    assert "app/auth.py" in g["tests/test_auth.py"]
    assert I.most_imported(sample_index)[0][0] == "app/auth.py"


def test_module_graph_mermaid_has_edges_and_folder_mode(sample_index):
    m = I.module_graph_mermaid(sample_index)
    assert "napp_main_py --> napp_auth_py" in m
    assert I.module_graph_mermaid(sample_index, by_folder=True).startswith("flowchart")


def test_dependencies_and_stack(sample_index):
    assert I.detect_dependencies(sample_index)["requirements.txt"] == ["flask", "jwt", "pytest"]
    assert "Flask" in I.detect_tech_stack(sample_index)


def test_dead_code_candidates_exclude_used_tests_and_entry_points(sample_index):
    names = {d["symbol"] for d in I.dead_code_candidates(sample_index)}
    assert {"unused_helper", "find_user"} <= names
    assert "hash_password" not in names and "login" not in names and "test_verify" not in names


def test_security_scan_finds_the_seeded_problems(sample_index):
    findings = I.security_scan(sample_index)
    issues = {(f["issue"], f["file"]) for f in findings}
    assert ("Hard-coded credential", "app/auth.py") in issues
    assert ("SQL built with string formatting", "app/db.py") in issues
    assert ("Weak hash (MD5/SHA1)", "app/auth.py") in issues
    assert ("Debug mode enabled", "app/main.py") in issues
    assert ("Bare except", "app/auth.py") in issues
    sev = [I.SEVERITY_ORDER[f["severity"]] for f in findings]
    assert sev == sorted(sev)                                                   # most severe first
    secret = next(f for f in findings if f["issue"] == "Hard-coded credential")
    assert "supersecretvalue123" not in secret["snippet"]                       # masked
    assert 0 <= I.security_score(findings) < 100


def _rule(title):
    return next(p for _, t, p in SECRET_RULES if t == title)


def test_credential_rule_true_and_false_positives():
    r = _rule("Hard-coded credential")
    for line in ('SECRET_KEY = "supersecretvalue123"', 'const dbPassword = "hunter2hunter2"', "api_key: 'abcd1234efgh'"):
        assert r.search(line), line
    for line in ('password = os.environ["PW"]', 'api_key = "your-api-key-here"', 'token: "${TOKEN}"',
                 'x = "short"', "self.password = password", 'password = ""', 'secret = getenv("S")'):
        assert not r.search(line), line


def test_provider_key_patterns():
    assert _rule("AWS access key id").search("key = AKIAIOSFODNN7EXAMPLE")
    assert _rule("Private key block").search("-----BEGIN RSA PRIVATE KEY-----")
    assert _rule("GitHub token").search("ghp_" + "a" * 36)
    assert _rule("Google API key").search("AIza" + "B" * 35)
    assert not _rule("AWS access key id").search("AKIA-short")


def test_security_findings_in_tests_are_downgraded(tmp_path, data_dir):
    from core import config
    from core.indexer import IndexParams, build_index
    from core.ingest import SourceSpec, acquire

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text('def test_a():\n    h = hashlib.md5(b"x")\n')
    config.ALLOW_LOCAL_PATHS = True
    with acquire(SourceSpec(kind="local", value=str(tmp_path))) as snap:
        idx = build_index(snap, IndexParams(embedder=config.OFFLINE_EMBEDDER), data_dir=data_dir)
    f = I.security_scan(idx)[0]
    assert f["severity"] == "low" and f["context"] == "test/example"


def test_docstring_coverage_and_complexity(sample_index):
    cov = I.docstring_coverage(sample_index)
    assert cov["total"] > 0 and 0 < cov["pct"] < 100
    assert I.complexity_hotspots(sample_index)[0]["complexity"] >= 1


def test_todo_scan(tmp_path, data_dir):
    from core import config
    from core.indexer import IndexParams, build_index
    from core.ingest import SourceSpec, acquire

    (tmp_path / "a.py").write_text("x = 1  # TODO: fix this\n# FIXME broken\n")
    config.ALLOW_LOCAL_PATHS = True
    with acquire(SourceSpec(kind="local", value=str(tmp_path))) as snap:
        idx = build_index(snap, IndexParams(embedder=config.OFFLINE_EMBEDDER), data_dir=data_dir)
    assert [(t["tag"], t["line"]) for t in I.todo_scan(idx)] == [("TODO", 1), ("FIXME", 2)]
