import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

st.set_page_config(page_title="Frontier Lens", layout="wide")
st.title("Frontier Lens")
st.write(
    "A small RL data engine for open-weight models. "
    "It generates verifiable tasks at the policy's edge, hardens exact-match reward, "
    "audits quality with humans, and keeps a clean held-out eval. "
    "Claim: **RL-ready**, not RL-proven, unless a training run is on the Results page."
)
st.markdown(
    """
**Pages**

1. Contest — sample the policy (k=8) and references (k=3)
2. Weaknesses — policy pass-rate histogram plus heatmap
3. Generate + Review — calibrate to the learnable band; audit sample + suspicious queue
4. Results — band share, headroom, audit CI, one-shot vs agent

Seed first: `uv run python -m frontier_lens.seed`
"""
)
role = st.sidebar.selectbox("Role", ["Organizer", "SME", "Viewer"])
st.session_state["role"] = role
st.sidebar.caption("Role switch only. Not real access control.")
