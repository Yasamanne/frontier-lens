import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import streamlit as st
from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.heatmap import heatmap
from frontier_lens.metrics import policy_item_rates
from frontier_lens.registry import policy_spec

st.title("Weaknesses")
engine = get_engine()
with engine.connect() as conn:
    runs = conn.execute(text("SELECT id, created_at, status FROM eval_runs ORDER BY created_at DESC")).all()
    models = conn.execute(text("SELECT DISTINCT model_id FROM attempts")).scalars().all()
if not runs:
    st.warning("No eval runs yet.")
    st.stop()
run_id = st.selectbox("Run", [r[0] for r in runs])
policy_id = policy_spec().id
default_idx = models.index(policy_id) if policy_id in models else 0
model_id = st.selectbox("Model", models, index=default_idx)

st.subheader("Policy pass-rate histogram")
items = policy_item_rates(run_id, model_id, split="train")
if items:
    labels = ["0 (out)", "0–⅛", "⅛–¼", "¼–⅜", "⅜–½", "½–⅝", "⅝–¾", "¾–⅞", "1 (out)"]
    counts = [0] * 9
    for it in items:
        r = it["pass_rate"]
        if r <= 0:
            counts[0] += 1
        elif r >= 1:
            counts[8] += 1
        else:
            counts[max(1, min(7, int(r * 8)))] += 1
    st.bar_chart(pd.DataFrame({"n": counts}, index=labels))
    in_band = sum(1 for i in items if i["in_band"])
    st.caption(
        "Learnable band is the open interval **0 < pass_rate < 1** "
        f"(interior bars). {in_band}/{len(items)} train items sit in the band. "
        "Bars labeled 0 and 1 are outside it."
    )
else:
    st.info("No train-split attempts for this model yet.")

st.subheader("Heatmap (supporting view)")
cells = heatmap(run_id, model_id)
st.dataframe(
    [
        {
            "topic": c.topic,
            "band": c.band,
            "n": c.n,
            "acc": round(c.accuracy, 3),
            "ci_lo": round(c.lo, 3),
            "ci_hi": round(c.hi, 3),
            "stability": round(c.stability, 3),
            "weak": c.weak,
        }
        for c in cells
    ],
    use_container_width=True,
)
st.caption("Grey/not weak if CI overlaps overall mean. Split=train templates.")
