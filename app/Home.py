import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

st.set_page_config(page_title="Frontier Lens", layout="wide")
st.title("Frontier Lens")
st.write(
    "Execution-verified eval for open-weight models. "
    "Keys come from running snippets. Models must predict `FINAL: <answer>` without tools."
)
st.markdown(
    """
**Pages**

1. Contest — run or replay an eval  
2. Weaknesses — heatmap with n and CIs  
3. Generate + Review — one-shot / agent + blind SME  
4. Results — new vs baseline and junk rates  

Seed first: `uv run python -m frontier_lens.seed`
"""
)
role = st.sidebar.selectbox("Role", ["Organizer", "SME", "Viewer"])
st.session_state["role"] = role
st.sidebar.caption("Role switch only. Not real access control.")
