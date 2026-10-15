import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.heatmap import heatmap

st.title("Weaknesses")
engine = get_engine()
with engine.connect() as conn:
    runs = conn.execute(text("SELECT id, created_at, status FROM eval_runs ORDER BY created_at DESC")).all()
    models = conn.execute(text("SELECT DISTINCT model_id FROM attempts")).scalars().all()
if not runs:
    st.warning("No eval runs yet.")
    st.stop()
run_id = st.selectbox("Run", [r[0] for r in runs])
model_id = st.selectbox("Model", models)
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
st.caption("Grey/not weak if CI overlaps overall mean. Split=selection.")
