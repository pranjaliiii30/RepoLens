"""Rendering safety, evaluation harness, and (when Streamlit is installed) an end-to-end app smoke test."""
from __future__ import annotations

import pytest

from core import config
from core.evaluation import STRATEGIES, generate_cases, run_benchmark
from core.render import (benchmark_summary, chat_to_markdown, mermaid_html, safe_markdown,
                         security_report_markdown, tree_lines)
from core.retriever import HybridRetriever


# ----------------------------------------------------------------------------- rendering safety
def test_safe_markdown_escapes_html_but_keeps_code_and_makes_badges():
    out = safe_markdown("<script>alert(1)</script> [app/a.py:3-5] `a<b`\n```html\n<div>x</div>\n```")
    assert "<script>" not in out and "&lt;script>" in out
    assert 'class="rl-cite"' in out and "a.py:3-5" in out
    assert "`a<b`" in out and "<div>x</div>" in out                         # code is untouched


def test_unverified_citations_are_marked():
    out = safe_markdown("[a.py:1-2] [b.py:1-2]", verified={("a.py", 1, 5)})
    assert out.count("rl-cite-bad") == 1 and out.count('class="rl-cite"') == 1


def test_citation_badge_escapes_paths():
    out = safe_markdown('[x"><img src=1>.py:1]')
    assert "<img" not in out


def test_mermaid_html_cannot_break_out_of_script_tag():
    h = mermaid_html('flowchart LR\n a["</script><script>alert(1)</script><!--"] --> b')
    assert h.count("</script>") == 1            # only the real closing tag; payload is written as <\/script>
    assert "<!--" not in h                      # no HTML-comment "double escape" trick either
    body = h.split('<script type="module">', 1)[1].split("</script>", 1)[0]
    assert "</script" not in body.lower()


def test_tree_and_exports(sample_index):
    tree = tree_lines(sample_index)
    assert tree.splitlines()[0] == "sample/" and "auth.py" in tree and "├──" in tree
    md = chat_to_markdown([{"role": "user", "content": "q?"},
                           {"role": "assistant", "content": "a [x.py:1-2]", "sources": ["x.py:1-2"]}], "demo")
    assert "# RepoLens chat — demo" in md and "x.py:1-2" in md
    from core.insights import security_scan, security_score

    f = security_scan(sample_index)
    rep = security_report_markdown(sample_index, f, security_score(f))
    assert "| Severity |" in rep and "supersecretvalue123" not in rep


# ----------------------------------------------------------------------------- evaluation
def test_generated_cases_hide_symbol_names_and_are_deterministic(sample_index):
    a = generate_cases(sample_index, n=20)
    b = generate_cases(sample_index, n=20)
    assert [c.question for c in a] == [c.question for c in b] and a
    for c in a:
        if c.kind == "docstring":
            for part in c.target_name.split(".")[-1].split("_"):
                if len(part) > 4:
                    assert part.lower() not in c.question.lower()


def test_benchmark_runs_and_metrics_are_sane(sample_stack, sample_index):
    cases = generate_cases(sample_index, n=12)
    res = run_benchmark(HybridRetriever(sample_stack), cases)
    assert res["n_cases"] == len(cases) and set(res["metrics"]) == set(STRATEGIES)
    for m in res["metrics"].values():
        assert 0 <= m["Recall@1"] <= m["Recall@3"] <= m["Recall@5"] <= m["Recall@10"] <= 1
        assert 0 <= m["MRR"] <= 1 and m["Recall@1"] <= m["MRR"] + 1e-9
    assert res["metrics"]["Hybrid (RRF)"]["Recall@10"] >= 0.8                   # tiny repo: should find nearly all
    assert "reached" in benchmark_summary(res)


# ----------------------------------------------------------------------------- full UI smoke test
def test_app_renders_without_exceptions():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("RepoLens" in m.value for m in at.markdown)
