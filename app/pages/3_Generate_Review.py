import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.review import promote

st.title("Generate + Review")
role = st.session_state.get("role", "Viewer")
engine = get_engine()
with engine.connect() as conn:
    cands = conn.execute(
        text(
            """
            SELECT id, topic, band, status, auto_reject_reason, generator_mode, stem, executed_key
            FROM candidates ORDER BY created_at DESC
            """
        )
    ).mappings().all()

st.write("Generate from CLI: `uv run python -m frontier_lens.generate --run RUN --model MODEL --mode one_shot`")
st.write("Blind SME view hides generator fields until after a decision.")

for c in cands:
    with st.expander(f"{c['id']} {c['status']} {c['topic']}/{c['band']}"):
        st.write(c["stem"][:1500] if c["stem"] else "(empty)")
        if c["status"] != "pending":
            st.write("key", c["executed_key"], "reason", c["auto_reject_reason"], "mode", c["generator_mode"])
        elif role == "SME":
            label = st.selectbox(
                "label",
                ["hard_item", "broken_item", "trivial", "rejected"],
                key=f"lab-{c['id']}",
            )
            note = st.text_input("note", key=f"note-{c['id']}")
            if st.button("Submit review", key=f"go-{c['id']}"):
                promote(
                    c["id"],
                    label,
                    fair=label == "hard_item",
                    novel=True,
                    hard=label == "hard_item",
                    note=note,
                )
                st.success("recorded")
        else:
            st.info("Switch role to SME to review (still no generator id shown).")
