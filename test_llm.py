"""LLM layer: grounded prompts, citation verification, retries/backoff, error mapping, offline fallback."""
from __future__ import annotations

import pytest

from core.llm import (GeminiClient, LLMError, LLMUnavailable, SYSTEM_PROMPT, _classify, build_prompt,
                      extractive_answer, format_context, heuristic_followups, history_text, parse_citations,
                      parse_followups, validate_citations)
from core.models import Chunk, SearchHit


def hit(path="app/auth.py", name="verify_token", s=8, e=13, code="def verify_token(t):\n    return t\n", typ="function"):
    return SearchHit(Chunk(f"{path}{s}", path, "python", name, typ, s, e, code, docstring="Validate a token."), 0.03)


# ----------------------------------------------------------------------------- citations
def test_parse_citations_handles_ranges_single_lines_and_nested_paths():
    text = "See [app/auth.py:8-13] and [src/a/b.ts:5] but not [link](http://x) or [1]."
    cs = parse_citations(text)
    assert [(c.path, c.start, c.end) for c in cs] == [("app/auth.py", 8, 13), ("src/a/b.ts", 5, 5)]


def test_validate_citations_flags_invented_references():
    hits = [hit()]
    ok, bad = validate_citations("Yes [app/auth.py:9-10]. Also [app/auth.py:100-120] and [nope/x.py:8-13].", hits)
    assert [(c.path, c.start) for c in ok] == [("app/auth.py", 9)]
    assert {c.path for c in bad} == {"app/auth.py", "nope/x.py"} and len(bad) == 2


def test_citation_overlapping_chunk_range_counts_as_verified():
    ok, bad = validate_citations("[app/auth.py:1-9]", [hit(s=8, e=13)])
    assert len(ok) == 1 and not bad


# ----------------------------------------------------------------------------- prompt
def test_context_has_headers_with_line_numbers_and_is_bounded():
    ctx = format_context([hit(code="a\nb\nc")])
    assert "### [app/auth.py:8-13]" in ctx and "    8| a" in ctx and "   10| c" in ctx
    many = [hit(path=f"f{i}.py", code="x" * 5000) for i in range(40)]
    assert len(format_context(many, max_chars=20_000)) < 30_000
    assert "no code context" in format_context([])


def test_prompt_contains_question_history_and_rules():
    p = build_prompt("what does it do?", [hit()], [{"role": "user", "content": "explain verify_token"},
                                                    {"role": "assistant", "content": "It validates JWTs."}], "demo")
    assert "QUESTION: what does it do?" in p and "USER: explain verify_token" in p and "CODE CONTEXT" in p
    for must in ("ONLY from the CODE CONTEXT", "[path/to/file.py:START-END]", "I couldn't find this in the indexed code",
                 "untrusted"):
        assert must in SYSTEM_PROMPT


def test_history_text_truncates_long_turns():
    h = [{"role": "user", "content": "x" * 5000}] * 10
    assert len(history_text(h)) < 3000


# ----------------------------------------------------------------------------- errors & retries
def test_error_classification():
    assert _classify(Exception("403 PERMISSION_DENIED api key not valid"))[1] is False
    assert "API key" in _classify(Exception("API key not valid"))[0]
    assert _classify(Exception("429 RESOURCE_EXHAUSTED quota"))[1] is True
    assert _classify(Exception("503 UNAVAILABLE overloaded"))[1] is True
    assert _classify(Exception("404 model not found"))[1] is False
    assert _classify(Exception("weird"))[1] is False


def test_retries_with_backoff_then_succeeds():
    client = GeminiClient("k", max_retries=3)
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("503 UNAVAILABLE")
        return "ok"

    assert client._with_retries(flaky, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0]        # exponential


def test_no_retry_for_auth_errors_and_gives_up_after_max_retries():
    client = GeminiClient("k", max_retries=2)
    n = []

    def bad_key():
        n.append(1)
        raise RuntimeError("403 PERMISSION_DENIED invalid api key")

    with pytest.raises(LLMError, match="API key"):
        client._with_retries(bad_key, sleep=lambda s: None)
    assert len(n) == 1
    n.clear()

    def always_busy():
        n.append(1)
        raise RuntimeError("429 quota")

    with pytest.raises(LLMError, match="rate limit"):
        client._with_retries(always_busy, sleep=lambda s: None)
    assert len(n) == 3                                                           # 1 try + 2 retries


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    c = GeminiClient("")
    assert not c.available
    with pytest.raises(LLMUnavailable, match="API key"):
        c.generate("hi")


# ----------------------------------------------------------------------------- offline & follow-ups
def test_extractive_answer_cites_matches():
    out = extractive_answer("q", [hit(), hit(name="other", s=20, e=30)])
    assert "Offline mode" in out and "[app/auth.py:8-13]" in out and "`verify_token`" in out
    assert "couldn't find" in extractive_answer("q", [])


def test_parse_followups_and_heuristics():
    raw = "1. What calls verify_token?\n- How is the secret key loaded?\nrandom sentence without question\n* Why?"
    assert parse_followups(raw) == ["What calls verify_token?", "How is the secret key loaded?"]
    assert heuristic_followups([hit()]) == ["What calls verify_token?"]
