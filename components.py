"""Reusable Streamlit widgets. Thin wrappers also absorb API differences between Streamlit versions."""
from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from core import config
from core.models import RepoIndex, SearchHit
from core.render import lang_color, mermaid_html
from ui.styles import CSS


# ----------------------------------------------------------------------------- version-tolerant wrappers
def df(data: Any, **kwargs) -> None:
    """st.dataframe stretched to the container (``width="stretch"`` on new Streamlit, else the old flag)."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def plot(fig, key: str | None = None) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def code(text: str, language: str | None = None, line_numbers: bool = True) -> None:
    lang = language if language not in ("text", "", None) else None
    try:
        st.code(text, language=lang, line_numbers=line_numbers)
    except TypeError:
        st.code(text, language=lang)


def mermaid(code_text: str, height: int = 520) -> None:
    """Render a Mermaid diagram (loaded from jsDelivr in the browser) with a source expander."""
    if not code_text:
        st.info("Nothing to draw for this selection.")
        return
    import streamlit.components.v1 as components

    components.html(mermaid_html(code_text, height=height), height=height, scrolling=False)
    with st.expander("Mermaid source"):
        st.code(code_text, language="text")


# ----------------------------------------------------------------------------- chrome
def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f'<span class="rl-pill {kind}">{html.escape(text)}</span>'


def hero(index: RepoIndex | None, ai_ready: bool) -> None:
    pills = []
    if index:
        pills += [pill(f"📦 {index.name}"), pill(f"🌿 {index.branch}"),
                  pill(f"{len(index.files):,} files"), pill(f"{index.loc:,} lines"),
                  pill(f"{len(index.chunks):,} chunks")]
        if index.commit:
            pills.append(pill(f"#{index.commit[:7]}"))
        pills.append(pill(f"🧠 {index.embedder}", "warn" if index.embedder == config.OFFLINE_EMBEDDER else ""))
    pills.append(pill("✨ Gemini ready" if ai_ready else "⚠ offline mode (no API key)", "ok" if ai_ready else "warn"))
    st.markdown(
        f"""<div class="rl-hero"><h1>🔎 <span class="grad">{config.APP_NAME}</span></h1>
<p>{config.TAGLINE}</p><div class="rl-meta">{''.join(pills)}</div></div>""",
        unsafe_allow_html=True)


def onboarding() -> None:
    st.markdown(
        """<div class="rl-steps">
<div class="rl-step"><div class="n">1</div><b>Add a repository</b><span>Paste a GitHub URL (or <code>owner/repo</code>) in the sidebar, or upload a ZIP.</span></div>
<div class="rl-step"><div class="n">2</div><b>Index it</b><span>RepoLens parses every function and class with real syntax trees and builds a hybrid search index.</span></div>
<div class="rl-step"><div class="n">3</div><b>Ask anything</b><span>“Where is authentication handled?” — answers cite exact files and line numbers.</span></div>
</div>""", unsafe_allow_html=True)
    st.markdown("##### Try one of these")
    cols = st.columns(len(config.EXAMPLE_REPOS))
    for col, (label, url) in zip(cols, config.EXAMPLE_REPOS):
        if col.button(f"📦 {label}", key=f"ex_{label}", help=url):
            st.session_state["repo_url"] = url
            st.session_state["flash"] = f"Loaded {label} into the sidebar — press **Index repository**."
            st.rerun()
    st.markdown(
        """<div class="rl-card" style="margin-top:14px"><h4>What makes it different</h4>
<p>• <b>Code-aware chunking</b> — one chunk per function/method/class (AST &amp; tree-sitter), not arbitrary text windows.</p>
<p>• <b>Hybrid retrieval</b> — vector search + BM25 fused with Reciprocal Rank Fusion, with a measurable benchmark in the Retrieval Lab.</p>
<p>• <b>Verified citations</b> — every <code>[file:lines]</code> in an answer is checked against what was actually retrieved.</p>
<p>• <b>Static analysis</b> — call graph, import graph, entry points, routes, complexity hotspots and a security scan, all offline.</p></div>""",
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sources
def source_card(hit: SearchHit, key: str, *, show_scores: bool = True) -> None:
    c = hit.chunk
    kind = c.symbol_type if c.symbol_type not in ("function", "method", "class") else f"{c.symbol_type} {c.qualname}"
    title = f"{c.file_path}:{c.start_line}-{c.end_line} · {kind}"
    with st.expander(title):
        meta = [f"`{c.language}`", f"{c.n_lines} lines"]
        if c.parts > 1:
            meta.append(f"part {c.part + 1}/{c.parts}")
        if c.complexity:
            meta.append(f"complexity {c.complexity}")
        st.caption(" · ".join(meta))
        if show_scores:
            bits = [f"fused **{hit.score:.4f}**"]
            if hit.vec_score is not None:
                bits.append(f"cosine **{hit.vec_score:.3f}** (rank {hit.vec_rank})")
            if hit.bm25_score is not None:
                bits.append(f"BM25 **{hit.bm25_score:.2f}** (rank {hit.bm25_rank})")
            if hit.cross_score is not None:
                bits.append(f"cross-encoder **{hit.cross_score:.2f}**")
            st.markdown(" · ".join(bits))
        if c.docstring:
            st.markdown(f"> {c.docstring.strip().splitlines()[0][:240]}")
        code(c.code, c.language)


# ----------------------------------------------------------------------------- charts
def language_donut(languages: dict[str, int]):
    import plotly.graph_objects as go

    items = list(languages.items())[:10]
    other = sum(v for _, v in list(languages.items())[10:])
    if other:
        items.append(("other", other))
    fig = go.Figure(go.Pie(labels=[k for k, _ in items], values=[v for _, v in items], hole=0.62,
                           marker=dict(colors=[lang_color(k) for k, _ in items]), sort=False,
                           textinfo="label+percent", hovertemplate="%{label}: %{value:,} lines<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def hbar(labels: list[str], values: list[float], title: str = "", color: str = "#7c6cf0", height: int = 340):
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(x=values[::-1], y=labels[::-1], orientation="h", marker_color=color,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=0, r=10, t=30 if title else 8, b=0), title=title,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis=dict(automargin=True), xaxis=dict(gridcolor="rgba(148,163,184,.15)"))
    return fig


def gauge(value: float, title: str):
    import plotly.graph_objects as go

    color = "#34d399" if value >= 80 else "#fbbf24" if value >= 50 else "#f87171"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": title},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                        "bgcolor": "rgba(148,163,184,.12)", "borderwidth": 0}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def benchmark_chart(metrics: dict[str, dict[str, float]]):
    import plotly.graph_objects as go

    names = list(metrics)
    keys = [k for k in next(iter(metrics.values())) if k.startswith("Recall") or k == "MRR"]
    palette = ["#7c6cf0", "#22d3ee", "#34d399", "#fbbf24", "#f472b6"]
    fig = go.Figure()
    for i, k in enumerate(keys):
        fig.add_bar(name=k, x=names, y=[metrics[n][k] for n in names], marker_color=palette[i % len(palette)],
                    hovertemplate="%{x}<br>" + k + ": %{y:.2%}<extra></extra>" if k != "MRR" else "%{x}<br>MRR: %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 1.05], gridcolor="rgba(148,163,184,.15)"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12))
    return fig
