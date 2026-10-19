import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Dev only: drop cached frontier_lens modules so edits under src/ load on the
# next rerun. Sessions run in threads that share sys.modules, so purging while
# another session imports raises KeyError; keep it off outside local dev.
if os.environ.get("FRONTIER_LENS_DEV_RELOAD") == "1":
    for _name in list(sys.modules):
        if _name == "frontier_lens" or _name.startswith("frontier_lens."):
            del sys.modules[_name]

import streamlit as st

st.set_page_config(
    page_title="Frontier Lens",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="collapsed",   # sidebar hidden by default
)

# ── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide Streamlit chrome ── */
[data-testid="collapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.stDeployButton { display: none !important; }

/* Remove top padding left by hidden header */
.block-container { padding-top: 1.5rem !important; }

/* Keep sidebar usable but invisible by default */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13132A 0%, #1A1035 100%);
    border-right: 1px solid #2D2D55;
}

/* ── Headings ── */
h1 {
    background: linear-gradient(90deg, #FF6B2B 0%, #C96FFF 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 800; letter-spacing: -0.5px;
}
h2 { color: #C96FFF !important; font-weight: 600;
     border-bottom: 1px solid #2D2D55; padding-bottom: 6px; margin-top: 1.2rem; }
h3 { color: #9B97B8 !important; font-weight: 600; font-size: 0.72rem !important;
     text-transform: uppercase; letter-spacing: 0.12em; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #13132A; border: 1px solid #2D2D55; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 2px 12px rgba(201,111,255,0.08);
}
[data-testid="stMetricLabel"] { color: #9B97B8 !important; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: #FF8F5C !important; font-weight: 700; }

/* ── Primary buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #FF6B2B, #E05500);
    border: none; border-radius: 8px; color: #fff; font-weight: 600;
    padding: 0.5rem 1.4rem; transition: all 0.2s ease;
    box-shadow: 0 0 14px rgba(255,107,43,0.35);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 22px rgba(255,107,43,0.6); transform: translateY(-1px);
}
.stButton > button:not([kind="primary"]) {
    background: #1A1A3A; border: 1px solid #3D3D6B; border-radius: 8px;
    color: #C8C4E0; font-weight: 500; transition: all 0.2s ease;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #C96FFF; color: #fff; box-shadow: 0 0 10px rgba(201,111,255,0.25);
}

/* ── Never cut off button text ── */
.stButton > button, .stDownloadButton > button, [data-testid^="stBaseButton"] {
    height: auto !important; min-height: 2.5rem; white-space: normal !important; overflow: visible !important; line-height: 1.3;
}
.stButton > button p, .stDownloadButton > button p, [data-testid^="stBaseButton"] p {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: break-word;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #FF6B2B, #C96FFF) !important;
    border-radius: 4px;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #2D2D55 !important; border-radius: 10px !important;
    background: #10102A;
}

/* ── Radio pill group (role selector) ── */
[data-testid="stRadio"] > div {
    display: flex; gap: 0; background: #13132A;
    border: 1px solid #2D2D55; border-radius: 30px;
    padding: 3px; width: fit-content;
}
[data-testid="stRadio"] label {
    border-radius: 24px !important; padding: 6px 20px !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    cursor: pointer; transition: all 0.2s; color: #9B97B8 !important;
    background: transparent; white-space: nowrap;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, #FF6B2B, #C96FFF) !important;
    color: #fff !important; font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(255,107,43,0.4);
}
[data-testid="stRadio"] input { display: none; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid #2D2D55; border-radius: 10px; overflow: hidden; }

/* ── Code blocks ── */
[data-testid="stCodeBlock"] { background: #0A0A18 !important;
    border: 1px solid #2D2D55; border-radius: 8px; }

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background: #2A1A4A !important; border: 1px solid #C96FFF !important; border-radius: 6px !important;
}

/* ── Divider ── */
hr { border-color: #2D2D55; }

/* ── Caption ── */
[data-testid="stCaption"], .stCaption, small { color: #7A7898 !important; }

/* ── Top nav (Turing flat-link style) ── */
.nav-bar { display:flex;align-items:center;justify-content:space-between;
           padding:0 0 16px 0;border-bottom:1px solid #2D2D55;margin-bottom:32px; }
.nav-links { display:flex;gap:0; }
a.nav-link { display:inline-block;white-space:nowrap;padding:8px clamp(8px,1.3vw,20px);color:#EDE8DC!important;font-size:.88rem;
            font-weight:500;text-decoration:none!important;border-bottom:2px solid transparent;
            transition:all .15s;letter-spacing:.01em;outline:none; }
a.nav-link:hover { color:#FF8F5C!important;border-bottom-color:#FF6B2B; }
a.nav-link.active { color:#FF8F5C!important;font-weight:700;border-bottom-color:#FF6B2B; }
</style>
""", unsafe_allow_html=True)

# ── Top navigation strip (Turing flat-link style) ────────────────────────
st.markdown("""
<div class="nav-bar">
  <a href="/" target="_self" style="text-decoration:none">
  <svg width="150" height="22" viewBox="0 0 150 22" style="overflow:visible;">
    <defs><linearGradient id="ng" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#FF6B2B"/><stop offset="100%" stop-color="#C96FFF"/>
    </linearGradient></defs>
    <text x="0" y="17" font-family="Inter,sans-serif" font-size="15"
          font-weight="800" letter-spacing="-0.3" fill="url(#ng)">Frontier Lens</text>
  </svg></a>
  <div class="nav-links">
    <a href="/Seed"            target="_self" class="nav-link">Seed</a>
    <a href="/Evaluate"        target="_self" class="nav-link">Evaluate</a>
    <a href="/Gap_Analysis"    target="_self" class="nav-link">Gap Analysis</a>
    <a href="/Generate_Review" target="_self" class="nav-link">Generate</a>
    <a href="/Dataset"         target="_self" class="nav-link">Dataset</a>
    <a href="/Headroom"        target="_self" class="nav-link">Headroom</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Missing env-var banner ────────────────────────────────────────────────
_missing_keys = [k for k in ("OPENROUTER_API_KEY",) if not os.getenv(k)]
if _missing_keys:
    st.warning(
        f"**Missing env vars:** `{'`, `'.join(_missing_keys)}`  \n"
        "Live model calls are unavailable until the API key is configured. "
        "Viewing existing data still works.",
        icon="⚠️",
    )

# ── Hero card ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    padding: 2.4rem 2.8rem;
    background: linear-gradient(135deg, #13132A 0%, #1C1040 60%, #200D30 100%);
    border: 1px solid #2D2D55;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 40px rgba(201,111,255,0.14);
">
    <!-- Gradient title rendered as SVG text to guarantee cross-browser rendering -->
    <svg width="400" height="56" viewBox="0 0 400 56" style="display:block;margin-bottom:0.5rem;overflow:visible;">
      <defs>
        <linearGradient id="tg" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stop-color="#FF6B2B"/>
          <stop offset="100%" stop-color="#C96FFF"/>
        </linearGradient>
      </defs>
      <text x="0" y="46" font-family="Inter,sans-serif" font-size="48"
            font-weight="800" letter-spacing="-2" fill="url(#tg)">Frontier Lens</text>
    </svg>
    <p style="color:#9B97B8;font-size:1.05rem;margin:0 0 1.4rem 0;line-height:1.65;max-width:660px;">
        An RL data engine for open-weight models — generates verifiable tasks at the policy's edge,
        hardens exact-match reward, and audits quality with humans.
    </p>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <span style="background:#1E0D35;border:1px solid #C96FFF;border-radius:20px;
                     padding:5px 16px;font-size:0.75rem;color:#C96FFF;font-weight:700;letter-spacing:0.08em;">
            ✦ RL-READY
        </span>
        <span style="background:#1A0D10;border:1px solid #FF6B2B;border-radius:20px;
                     padding:5px 16px;font-size:0.75rem;color:#FF8F5C;font-weight:700;letter-spacing:0.08em;">
            OPEN-WEIGHT ONLY
        </span>
        <span style="background:#0D1A18;border:1px solid #2E9E7E;border-radius:20px;
                     padding:5px 16px;font-size:0.75rem;color:#4CC9A0;font-weight:700;letter-spacing:0.08em;">
            SANDBOX-VERIFIED
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Role selector (inline, no sidebar) ───────────────────────────────────
st.markdown("### Your role")
role = st.radio(
    "",
    ["Organizer", "SME", "Viewer"],
    horizontal=True,
    index=["Organizer", "SME", "Viewer"].index(st.session_state.get("role", "Viewer")),
    key="role_radio",
    label_visibility="collapsed",
)
st.session_state["role"] = role

ROLE_DESC = {
    "Organizer": "Full access — run calibration, generate seeds, run evaluations, trigger generation.",
    "SME":       "Subject-matter expert — approve or reject generated questions, with reasons and notes.",
    "Viewer":    "Read-only — browse runs, analysis and datasets; cannot start jobs or change data.",
}
ROLE_COLOR = {"Organizer": "#FF8F5C", "SME": "#C96FFF", "Viewer": "#4CC9A0"}
st.markdown(
    f'<p style="color:{ROLE_COLOR[role]}; font-size:0.85rem; margin-top:4px;">● {ROLE_DESC[role]}</p>',
    unsafe_allow_html=True,
)

st.divider()

# ── Pipeline steps ────────────────────────────────────────────────────────
st.markdown("""
<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.14em;color:#5A5875;margin-bottom:12px;">
    Pipeline — 6 steps
</div>
""", unsafe_allow_html=True)

cols = st.columns(6)
steps = [
    ("1", "🌱", "Seed",      "Generate procedural questions or upload your own dataset.", "/Seed"),
    ("2", "⚡", "Evaluate",  "Run the policy k=8 times on every question. Record every trace.", "/Evaluate"),
    ("3", "🔍", "Gap Analysis","Pass-rate histogram, failure-mode heatmap, and dominant error type.", "/Gap_Analysis"),
    ("4", "✨", "Generate",  "LLM-generates new items targeting weak cells. SME review queue.", "/Generate_Review"),
    ("5", "📊", "Dataset",   "Band share, audit CI, and rl_tasks.jsonl export for TRL/GRPO.", "/Dataset"),
    ("6", "📈", "Headroom",  "Before vs. after: how much the pipeline grew the RL-trainable pool.", "/Headroom"),
]

for col, (num, icon, name, desc, href) in zip(cols, steps):
    with col:
        st.markdown(f"""
<a href="{href}" target="_self" style="text-decoration:none;">
<div style="
    background:#13132A; border:1px solid #2D2D55; border-radius:16px;
    padding:1.6rem 1.2rem 1.4rem; text-align:center; cursor:pointer;
    min-height:190px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:6px;
" onmouseover="this.style.borderColor='#FF6B2B';this.style.boxShadow='0 0 24px rgba(255,107,43,0.18)'"
   onmouseout="this.style.borderColor='#2D2D55';this.style.boxShadow='none'">
    <div style="font-size:2.2rem;line-height:1;">{icon}</div>
    <div style="font-size:0.6rem;color:#FF6B2B;font-weight:700;
                letter-spacing:0.16em;text-transform:uppercase;">Step {num}</div>
    <div style="font-size:1rem;font-weight:700;color:#E8E6F0;">{name}</div>
    <div style="font-size:0.8rem;color:#7A7898;line-height:1.55;max-width:140px;">{desc}</div>
</div>
</a>
""", unsafe_allow_html=True)

# ── Getting started ───────────────────────────────────────────────────────
st.markdown("### Getting started")
st.markdown(
    "Open **Seed** to load or upload questions, run **Evaluate** to measure a model, "
    "then use **Gap Analysis** and **Generate** to find weak spots and build new questions."
)
