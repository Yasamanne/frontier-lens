import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Dev only: drop cached frontier_lens modules so edits under src/ load on the
# next rerun. Sessions run in threads that share sys.modules, so purging while
# another session imports raises KeyError; keep it off outside local dev.
if os.environ.get("FRONTIER_LENS_DEV_RELOAD") == "1":
    for _name in list(sys.modules):
        if _name == "frontier_lens" or _name.startswith("frontier_lens."):
            del sys.modules[_name]

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Frontier Lens · Gap Analysis",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Shared theme CSS + top nav ───────────────────────────────────────────
import streamlit as _st_nav
_st_nav.markdown("""
<style>.nav-bar{display:flex;align-items:center;justify-content:space-between;padding:0 0 16px 0;border-bottom:1px solid #2D2D55;margin-bottom:32px;}.nav-links{display:flex;gap:0;align-items:center;}a.nav-link{display:inline-block;white-space:nowrap;padding:8px clamp(8px,1.3vw,20px);color:#EDE8DC!important;font-size:.88rem;font-weight:500;text-decoration:none!important;border-bottom:2px solid transparent;transition:all .15s;letter-spacing:.01em;outline:none;}a.nav-link:hover{color:#FF8F5C!important;border-bottom-color:#FF6B2B;}a.nav-link.active{color:#FF8F5C!important;font-weight:700;border-bottom-color:#FF6B2B;}.stButton>button,.stDownloadButton>button,[data-testid^="stBaseButton"]{height:auto!important;min-height:2.5rem;white-space:normal!important;overflow:visible!important;line-height:1.3;}.stButton>button p,.stDownloadButton>button p,[data-testid^="stBaseButton"] p{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;overflow-wrap:break-word;}</style><div class="nav-bar"><a href="/" target="_self" style="text-decoration:none"><svg width="150" height="22" viewBox="0 0 150 22" style="overflow:visible;"><defs><linearGradient id="ng2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#FF6B2B"/><stop offset="100%" stop-color="#C96FFF"/></linearGradient></defs><text x="0" y="17" font-family="Inter,sans-serif" font-size="15" font-weight="800" letter-spacing="-0.3" fill="url(#ng2)">Frontier Lens</text></svg></a><div class="nav-links"><a href="/Seed" target="_self" class="nav-link">Seed</a><a href="/Evaluate" target="_self" class="nav-link">Evaluate</a><a href="/Gap_Analysis" target="_self" class="nav-link active">Gap Analysis</a><a href="/Generate_Review" target="_self" class="nav-link">Generate</a><a href="/Dataset" target="_self" class="nav-link">Dataset</a><a href="/Headroom" target="_self" class="nav-link">Headroom</a></div></div>
""", unsafe_allow_html=True)

# ── Role selector ────────────────────────────────────────────────────────
_current_role = st.session_state.get("role", "Viewer")
_role_cols = st.columns([4, 1])
with _role_cols[1]:
    _new_role = st.selectbox(
        "Role", ["Organizer", "SME", "Viewer"],
        index=["Organizer", "SME", "Viewer"].index(_current_role),
        key="role_select_nav", label_visibility="collapsed",
    )
    if _new_role != _current_role:
        st.session_state["role"] = _new_role
        st.rerun()
role = st.session_state.get("role", "Viewer")

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.failure_analysis import failure_modes_all_cells
from frontier_lens.heatmap import heatmap
from frontier_lens.metrics import HIST_LABELS, histogram_counts, policy_item_rates
from frontier_lens.registry import policy_spec
from frontier_lens.scope import POOLED, run_clause

st.title("Gap Analysis")
engine = get_engine()

with engine.connect() as conn:
    runs = conn.execute(text("SELECT id, created_at, status FROM eval_runs ORDER BY created_at DESC")).all()

if not runs:
    st.warning("No eval runs yet.")
    st.stop()

_STATUS_ICON = {"complete": "🟢", "failed": "🔴", "running": "🟡"}
_run_labels = [
    f"{_STATUS_ICON.get(r[2], '⚪')} {r[0]}  —  {r[1][:16].replace('T', ' ')}"
    for r in runs
]
run_id = st.selectbox(
    "Run",
    [r[0] for r in runs],
    format_func=lambda rid: next(
        (lbl for lbl, r in zip(_run_labels, runs) if r[0] == rid), rid
    ),
)
if st.checkbox(
    "Pool all completed runs",
    help="Combine the samples of every completed run, e.g. after \"Score only what's missing\" "
         "runs that each hold only part of the questions. Samples add up per question.",
):
    run_id = POOLED

# Only models that have attempts in the chosen scope.
with engine.connect() as conn:
    models = conn.execute(
        text(f"SELECT DISTINCT a.model_id FROM attempts a WHERE {run_clause('r')} ORDER BY 1"),
        {"r": run_id},
    ).scalars().all()
if not models:
    st.info("No attempts in this run yet.")
    st.stop()
policy_id = policy_spec().id
default_idx = list(models).index(policy_id) if policy_id in list(models) else 0
model_id = st.selectbox("Model", list(models), index=default_idx)

# ── Pass-rate histogram ───────────────────────────────────────────────────
st.subheader("Policy pass-rate histogram")
items = policy_item_rates(run_id, model_id, split="train")
if items:
    counts = histogram_counts([it["pass_rate"] for it in items])
    st.bar_chart(pd.DataFrame({"n": counts}, index=HIST_LABELS))
    in_band = sum(1 for i in items if i["in_band"])
    st.caption(
        "Learnable band is the open interval **0 < pass_rate < 1** "
        f"(interior bars). {in_band}/{len(items)} train items sit in the band. "
        "Bars labeled 0 and 1 are outside it. Questions an SME marked broken are left out."
    )
    with engine.connect() as conn:
        n_pending = conn.execute(
            text(
                f"""
                SELECT COUNT(DISTINCT q.id) FROM attempts a
                JOIN questions q ON q.id = a.question_id
                WHERE {run_clause('r')} AND a.model_id = :m AND q.split = 'train'
                  AND COALESCE(q.sme_flagged, 0) = 1
                """
            ),
            {"r": run_id, "m": model_id},
        ).scalar() or 0
    if n_pending:
        st.caption(
            f"{n_pending} question(s) awaiting SME review are included "
            "(their keys are not confirmed yet)."
        )
    st.caption("→ Go to **Generate** to run dead-zone rescue and RL-suitable augmentation.")
else:
    st.info("No train-split attempts for this model yet.")

# ── Heatmap ───────────────────────────────────────────────────────────────
st.subheader("Heatmap")
cells = heatmap(run_id, model_id)
st.dataframe(
    [
        {
            "topic": c.topic, "band": c.band, "n": c.n,
            "acc": round(c.accuracy, 3),
            "ci_lo": round(c.lo, 3), "ci_hi": round(c.hi, 3),
            "stability": round(c.stability, 3), "weak": c.weak,
        }
        for c in cells
    ],
    width="stretch",
)
st.caption("Grey/not weak if CI overlaps overall mean. Split=train templates.")

# ── Failure mode breakdown ────────────────────────────────────────────────
st.subheader("Failure mode breakdown")
st.caption(
    "For each (topic, band) cell where the policy mostly fails, "
    "we classify the k attempts by failure type."
)

with st.spinner("Analyzing failure modes…"):
    fa_cells = failure_modes_all_cells(run_id, model_id)

if not fa_cells:
    st.info("No failing cells found — the model passes most questions or no attempts exist yet.")
else:
    MODE_LABELS = {
        "unparsable_frac":      "Format failure\n(no FINAL:)",
        "off_by_one_frac":      "Off-by-one\n(±1 numeric)",
        "hedge_frac":           "Hedging\n(or / either)",
        "consistent_wrong_frac":"Consistent wrong\n(systematic)",
        "high_variance_frac":   "High variance\n(random noise)",
    }
    rows = []
    for cell in fa_cells:
        agg = cell.get("aggregate", {})
        row = {
            "topic": cell["topic"], "band": cell["band"],
            "failing_q": cell["n_failing_questions"],
            "dominant_mode": cell["dominant_mode"].replace("_frac", "").replace("_", " "),
        }
        for key, label in MODE_LABELS.items():
            row[label.split("\n")[0]] = f"{agg.get(key, 0):.0%}"
        rows.append(row)
    st.dataframe(rows, width="stretch")

    for cell in fa_cells:
        label = f"📍 {cell['topic']} / {cell['band']}  —  dominant: {cell['dominant_mode'].replace('_frac','').replace('_',' ')}"
        with st.expander(label):
            agg = cell.get("aggregate", {})
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Format failure",   f"{agg.get('unparsable_frac', 0):.0%}")
            col2.metric("Off-by-one",       f"{agg.get('off_by_one_frac', 0):.0%}")
            col3.metric("Hedging",          f"{agg.get('hedge_frac', 0):.0%}")
            col4.metric("Consistent wrong", f"{agg.get('consistent_wrong_frac', 0):.0%}")
            col5.metric("High variance",    f"{agg.get('high_variance_frac', 0):.0%}")

            st.caption("Per-question breakdown (failing questions only):")
            for pq in cell.get("per_question", [])[:8]:
                modes = pq["modes"]
                with st.expander(
                    f"`{pq['question_id'][:10]}` — {pq['n_correct']}/{pq['n']} correct",
                    expanded=False,
                ):
                    st.text(pq["stem_snippet"])
                    st.json({
                        "unparsable":       f"{modes.get('unparsable_frac',0):.0%}",
                        "off_by_one":       f"{modes.get('off_by_one_frac',0):.0%}",
                        "hedge":            f"{modes.get('hedge_frac',0):.0%}",
                        "consistent_wrong": bool(modes.get("consistent_wrong")),
                        "high_variance":    bool(modes.get("high_variance")),
                    })
