import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.heatmap import weakest_cell
from frontier_lens.metrics import junk_by_mode, new_vs_baseline

st.title("Results")
engine = get_engine()
with engine.connect() as conn:
    runs = conn.execute(text("SELECT id FROM eval_runs ORDER BY created_at DESC")).scalars().all()
    models = conn.execute(text("SELECT DISTINCT model_id FROM attempts")).scalars().all()
if not runs or not models:
    st.warning("Need a completed contest run.")
    st.stop()
run_id = st.selectbox("Run", runs)
model_id = st.selectbox("Model", models)
cell = weakest_cell(run_id, model_id)
if cell:
    st.write(f"Weakest selection cell: **{cell.topic} / {cell.band}** acc={cell.accuracy:.2f} n={cell.n}")
    st.json(new_vs_baseline(run_id, model_id, cell.topic, cell.band))
st.subheader("Junk by generator mode")
st.json(junk_by_mode())
