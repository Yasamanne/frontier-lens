import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.heatmap import weakest_cell
from frontier_lens.metrics import (
    band_share,
    candidate_band_by_mode,
    junk_by_mode,
    latest_audit,
    policy_item_rates,
)
from frontier_lens.registry import policy_spec
from frontier_lens.reward import guessable_templates

st.title("Results")
engine = get_engine()
with engine.connect() as conn:
    runs = conn.execute(text("SELECT id FROM eval_runs ORDER BY created_at DESC")).scalars().all()
    models = conn.execute(text("SELECT DISTINCT model_id FROM attempts")).scalars().all()
if not runs or not models:
    st.warning("Need a completed contest run.")
    st.stop()
run_id = st.selectbox("Run", runs)
policy_id = policy_spec().id
default_idx = models.index(policy_id) if policy_id in models else 0
model_id = st.selectbox("Model", models, index=default_idx)

items = policy_item_rates(run_id, model_id, split="train")
report = band_share(items)
st.subheader("Learnable band")
st.metric("Share of train items in (0, 1)", f"{report['share']:.1%}", help="Policy pass_rate strictly between 0 and 1")
st.metric("Mean headroom (pass@k − pass@1)", f"{report['mean_headroom']:.3f}")
st.write("By template", report["by_template"])
st.write("By generator mode (seed items listed as seed)", report["by_mode"])

st.subheader("One-shot vs agent (learnable-band hit rate)")
st.json(candidate_band_by_mode())

st.subheader("Audit defect rate")
audit = latest_audit()
if audit:
    st.write(
        f"Batch `{audit['id']}`: {audit['defects']}/{audit['n']} defects, "
        f"Wilson 95% CI [{audit['defect_lo']:.3f}, {audit['defect_hi']:.3f}], "
        f"{'accepted' if audit['accepted'] else 'rejected'} "
        f"(threshold {audit['threshold']:.0%})."
    )
else:
    st.info("No audit batch closed yet.")

st.subheader("Guessable answer spaces")
with engine.connect() as conn:
    keys = conn.execute(
        text("SELECT template, key FROM questions WHERE source='seed'")
    ).all()
by_t: dict[str, list[str]] = {}
for template, key in keys:
    by_t.setdefault(template, []).append(key)
flagged = guessable_templates(by_t)
st.write(flagged or "(none flagged)")

st.subheader("Heatmap weakest cell (supporting)")
cell = weakest_cell(run_id, model_id)
if cell:
    st.write(f"**{cell.topic} / {cell.band}** acc={cell.accuracy:.2f} n={cell.n}")

st.subheader("Junk by generator mode")
st.json(junk_by_mode())
st.caption("RL-ready, not RL-proven: no training run is implied by these numbers.")
