"""Presentation helpers with no Streamlit dependency (so they are unit-testable)."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime

from .llm import CITATION_RE
from .models import RepoIndex
from .utils import posix

_CODE_SPAN = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.S)
_LANG_COLORS = {"python": "#3572A5", "javascript": "#f1e05a", "typescript": "#3178c6", "java": "#b07219",
                "go": "#00ADD8", "rust": "#dea584", "c": "#9aa0a6", "cpp": "#f34b7d", "csharp": "#178600",
                "markdown": "#083fa1", "json": "#cbcb41", "yaml": "#cb171e", "shell": "#89e051",
                "html": "#e34c26", "css": "#563d7c", "ruby": "#701516", "php": "#4F5D95"}


def lang_color(lang: str) -> str:
    return _LANG_COLORS.get(lang, "#8b8fa3")


def citation_badge(path: str, start: int, end: int, ok: bool = True) -> str:
    label = html.escape(f"{posix(path).rsplit('/', 1)[-1]}:{start}" + (f"-{end}" if end != start else ""))
    title = html.escape(f"{path}:{start}-{end}")
    cls = "rl-cite" if ok else "rl-cite rl-cite-bad"
    return f'<span class="{cls}" title="{title}">{label}</span>'


def safe_markdown(text: str, verified: set[tuple[str, int, int]] | None = None, badges: bool = True) -> str:
    """Prepare model output for ``st.markdown(..., unsafe_allow_html=True)``.

    * Raw ``<`` outside code spans is escaped, so HTML/script injected through repo content or the model can
      never render.
    * ``[path:12-30]`` citations outside code become small badges.
    """
    out: list[str] = []
    for part in _CODE_SPAN.split(text):
        if part.startswith(("```", "~~~", "`")) and _CODE_SPAN.fullmatch(part):
            out.append(part)
            continue
        part = part.replace("<", "&lt;")
        if badges:
            def repl(m: re.Match) -> str:
                s = int(m.group(2))
                e = int(m.group(3)) if m.group(3) else s
                ok = True if verified is None else any(
                    m.group(1) == p and s <= ve and e >= vs for p, vs, ve in verified)
                return citation_badge(m.group(1), s, e, ok)
            part = CITATION_RE.sub(repl, part)
        out.append(part)
    return "".join(out)


def tree_lines(index: RepoIndex, max_files: int = 400) -> str:
    """ASCII tree of the indexed files (sorted, folders first)."""
    tree: dict = {}
    for f in sorted(index.files, key=lambda x: x.path)[:max_files]:
        node = tree
        parts = posix(f.path).split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = f.lines
    lines: list[str] = [index.name + "/"]

    def walk(node: dict, prefix: str) -> None:
        items = sorted(node.items(), key=lambda kv: (not isinstance(kv[1], dict), kv[0].lower()))
        for i, (name, val) in enumerate(items):
            last = i == len(items) - 1
            branch = "└── " if last else "├── "
            if isinstance(val, dict):
                lines.append(f"{prefix}{branch}{name}/")
                walk(val, prefix + ("    " if last else "│   "))
            else:
                lines.append(f"{prefix}{branch}{name}  ({val})")

    walk(tree, "")
    if len(index.files) > max_files:
        lines.append(f"… and {len(index.files) - max_files} more files")
    return "\n".join(lines)


def chat_to_markdown(messages: list[dict], repo_name: str = "") -> str:
    lines = [f"# RepoLens chat — {repo_name}" if repo_name else "# RepoLens chat",
             f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"## 🧑 {m['content']}", ""]
        else:
            lines += [m["content"], ""]
            refs = m.get("sources") or []
            if refs:
                lines.append("<details><summary>Sources</summary>\n")
                lines += [f"- `{r}`" for r in refs]
                lines.append("\n</details>\n")
            lines.append("---\n")
    return "\n".join(lines)


def security_report_markdown(index: RepoIndex, findings: list[dict], score: int) -> str:
    lines = [f"# Security & quality report — {index.name}", "", f"**Score:** {score}/100  ·  "
             f"**Findings:** {len(findings)}", "", "| Severity | Issue | Location | Snippet |", "|---|---|---|---|"]
    for f in findings:
        snippet = f["snippet"].replace("|", "\\|").replace("`", "'")
        lines.append(f"| {f['severity']} | {f['issue']} | `{f['file']}:{f['line']}` | `{snippet}` |")
    lines += ["", "_Static heuristics only — review each finding manually._"]
    return "\n".join(lines)


def benchmark_summary(bench: dict) -> str:
    """One-line, CV-ready sentence from benchmark results."""
    m = bench["metrics"]
    if not m:
        return ""
    base = m.get("Vector only") or next(iter(m.values()))
    best_name, best = max(m.items(), key=lambda kv: kv[1]["MRR"])
    r5 = f"Recall@5 {best['Recall@5']:.0%}" if "Recall@5" in best else ""
    return (f"On {bench['n_cases']} auto-generated queries, '{best_name}' reached {r5}, MRR {best['MRR']:.2f} "
            f"(vector-only baseline: MRR {base['MRR']:.2f}).")


def mermaid_html(code: str, height: int = 520, theme: str = "dark") -> str:
    """Self-contained HTML that renders a Mermaid diagram (for ``st.components.v1.html``)."""
    safe = html.escape(code)
    code_js = json.dumps(code).replace("</", "<\\/").replace("<!--", "<\\!--")   # never close the <script>
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:transparent;font-family:ui-sans-serif,system-ui,sans-serif}}
#wrap{{overflow:auto;height:{height - 8}px;border:1px solid rgba(139,143,163,.3);border-radius:12px;padding:8px;box-sizing:border-box}}
.err{{color:#f87171;font:13px ui-monospace,monospace;padding:12px;white-space:pre-wrap}}
</style></head><body><div id="wrap"><pre class="mermaid">{safe}</pre></div>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
try {{
  mermaid.initialize({{startOnLoad:false, theme:"{theme}", securityLevel:"strict", flowchart:{{curve:"basis", htmlLabels:false}}}});
  await mermaid.run({{querySelector:".mermaid"}});
}} catch (e) {{
  document.getElementById("wrap").innerHTML = '<div class="err">Diagram could not be rendered (are you offline?). Copy the source below into mermaid.live.\\n\\n' + {code_js}.replace(/</g,"&lt;") + '</div>';
}}
</script></body></html>"""
