"""Custom CSS: dark developer theme with an indigo/violet accent. Works on top of .streamlit/config.toml."""

ACCENT = "#7c6cf0"

CSS = """
<style>
:root{
  --rl-accent:#7c6cf0; --rl-accent-2:#22d3ee; --rl-bg:#0d1017; --rl-card:#141925; --rl-card-2:#1a2030;
  --rl-border:rgba(148,163,184,.18); --rl-muted:#8b93a7; --rl-text:#e6e9f2;
  --rl-mono:ui-monospace,SFMono-Regular,"JetBrains Mono","Fira Code",Menlo,Consolas,monospace;
}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1280px}
section[data-testid="stSidebar"]{border-right:1px solid var(--rl-border)}
section[data-testid="stSidebar"] .block-container{padding-top:1rem}
h1,h2,h3{letter-spacing:-.01em}
code,pre,.stCode{font-family:var(--rl-mono)!important}

/* hero */
.rl-hero{position:relative;overflow:hidden;border:1px solid var(--rl-border);border-radius:18px;
  padding:22px 26px;margin-bottom:14px;
  background:radial-gradient(900px 220px at 0% 0%,rgba(124,108,240,.28),transparent 60%),
             radial-gradient(700px 200px at 100% 100%,rgba(34,211,238,.14),transparent 60%),var(--rl-card)}
.rl-hero h1{margin:0;font-size:1.9rem;font-weight:750;display:flex;align-items:center;gap:.5rem}
.rl-hero h1 span.grad{background:linear-gradient(90deg,#a99cff,#22d3ee);-webkit-background-clip:text;background-clip:text;color:transparent}
.rl-hero p{margin:.35rem 0 0;color:var(--rl-muted);font-size:1rem}
.rl-hero .rl-meta{margin-top:.9rem;display:flex;flex-wrap:wrap;gap:.45rem}

/* pills */
.rl-pill{display:inline-flex;align-items:center;gap:.35rem;padding:.18rem .6rem;border-radius:999px;font-size:.78rem;
  border:1px solid var(--rl-border);background:rgba(124,108,240,.10);color:var(--rl-text);font-family:var(--rl-mono)}
.rl-pill.ok{background:rgba(52,211,153,.12);border-color:rgba(52,211,153,.35)}
.rl-pill.warn{background:rgba(251,191,36,.12);border-color:rgba(251,191,36,.35)}
.rl-pill.bad{background:rgba(248,113,113,.12);border-color:rgba(248,113,113,.4)}
.rl-dot{width:.55rem;height:.55rem;border-radius:50%;display:inline-block}

/* citations */
.rl-cite{display:inline-block;padding:0 .45rem;margin:0 .1rem;border-radius:6px;font:600 .74rem/1.45 var(--rl-mono);
  color:#c7c0ff;background:rgba(124,108,240,.18);border:1px solid rgba(124,108,240,.45);white-space:nowrap;vertical-align:baseline}
.rl-cite-bad{color:#fca5a5;background:rgba(248,113,113,.12);border-color:rgba(248,113,113,.45);text-decoration:line-through dotted}

/* cards */
.rl-card{border:1px solid var(--rl-border);background:var(--rl-card);border-radius:14px;padding:16px 18px}
.rl-card h4{margin:0 0 .35rem 0;font-size:1rem}
.rl-card p{margin:.15rem 0;color:var(--rl-muted);font-size:.92rem}
.rl-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0}
.rl-step{border:1px solid var(--rl-border);background:var(--rl-card);border-radius:14px;padding:14px 16px}
.rl-step .n{width:1.7rem;height:1.7rem;border-radius:50%;display:inline-grid;place-items:center;font-weight:700;
  background:linear-gradient(135deg,var(--rl-accent),#4f46e5);color:white;margin-bottom:.4rem}
.rl-step b{display:block;margin-bottom:.15rem}
.rl-step span{color:var(--rl-muted);font-size:.9rem}

/* metrics */
div[data-testid="stMetric"]{background:var(--rl-card);border:1px solid var(--rl-border);border-radius:14px;padding:12px 16px}
div[data-testid="stMetricLabel"] p{color:var(--rl-muted)!important;font-size:.8rem!important;text-transform:uppercase;letter-spacing:.06em}
div[data-testid="stMetricValue"]{font-weight:700}

/* tabs */
button[data-baseweb="tab"]{font-weight:600;padding:.55rem 1rem}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--rl-accent),var(--rl-accent-2))!important;height:3px!important}

/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid var(--rl-border);transition:all .15s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--rl-accent);transform:translateY(-1px)}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#7c6cf0,#5b4de0);border:0}

/* chat */
div[data-testid="stChatMessage"]{border:1px solid var(--rl-border);background:var(--rl-card);border-radius:14px;padding:.9rem 1.1rem}
div[data-testid="stChatMessage"] p{line-height:1.62}
.rl-src-head{font-family:var(--rl-mono);font-size:.82rem;color:var(--rl-muted)}
.rl-score{font-family:var(--rl-mono);font-size:.74rem;color:var(--rl-accent-2)}
.rl-verify{font-size:.78rem;color:var(--rl-muted);margin-top:.25rem}

/* expanders / dataframes / code */
div[data-testid="stExpander"]{border:1px solid var(--rl-border)!important;border-radius:12px!important;background:var(--rl-card)}
div[data-testid="stDataFrame"]{border:1px solid var(--rl-border);border-radius:12px;overflow:hidden}
div[data-testid="stCode"] pre{border-radius:10px}
.rl-sev-critical{color:#f87171;font-weight:700}.rl-sev-high{color:#fb923c;font-weight:700}
.rl-sev-medium{color:#fbbf24;font-weight:600}.rl-sev-low{color:#94a3b8}
.rl-empty{text-align:center;padding:30px 10px;color:var(--rl-muted)}
footer{visibility:hidden}
@media (max-width:900px){.rl-hero h1{font-size:1.5rem}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
"""
