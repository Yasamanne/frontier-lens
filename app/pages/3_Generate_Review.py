import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.registry import policy_spec
from frontier_lens.review import (
    close_audit_batch,
    draw_audit_sample,
    flag_suspicious,
    promote,
)

st.title("Generate + Review")
role = st.session_state.get("role", "Viewer")
engine = get_engine()

st.write(
    "Generate from CLI: `uv run python -m frontier_lens.generate --run RUN --model POLICY --mode one_shot --split train`"
)
st.write("Eval split: add `--split eval` (never samples the policy). Blind SME view hides generator fields.")

if role in {"Organizer", "SME"}:
    n = st.number_input("Audit sample size", 1, 200, 50)
    if st.button("Draw audit sample from pending train candidates"):
        try:
            batch = draw_audit_sample(n=int(n))
            st.success(f"audit batch {batch}")
        except RuntimeError as exc:
            st.error(str(exc))
    if st.button("Close audit batch (latest)"):
        with engine.connect() as conn:
            latest = conn.execute(
                text("SELECT id FROM audit_batches ORDER BY created_at DESC LIMIT 1")
            ).scalar()
        if not latest:
            st.error("no audit batch")
        else:
            try:
                st.json(close_audit_batch(latest))
            except RuntimeError as exc:
                st.error(str(exc))

with engine.connect() as conn:
    cands = conn.execute(
        text(
            """
            SELECT id, topic, band, status, auto_reject_reason, generator_mode, stem,
                   executed_key, audit_batch_id, audit_sampled, policy_pass_rate, rl_split
            FROM candidates ORDER BY created_at DESC
            """
        )
    ).mappings().all()

st.subheader("Audit sample (sampled pending items)")
sampled = [
    c for c in cands if c["audit_sampled"] and c["status"] == "pending"
]
if not sampled:
    st.info("No sampled pending items. Draw an audit sample after generating candidates.")
for c in sampled:
    with st.expander(f"{c['id']} {c['topic']}/{c['band']} split={c['rl_split']}"):
        st.write(c["stem"][:1500] if c["stem"] else "(empty)")
        if role == "SME":
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
            st.info("Switch role to SME to label. Generator model and mode stay hidden.")

st.subheader("Suspicious correct policy samples")
if role in {"Organizer", "SME"}:
    with engine.connect() as conn:
        runs = conn.execute(
            text("SELECT id FROM eval_runs WHERE status='complete' ORDER BY created_at DESC")
        ).scalars().all()
    run_id = st.selectbox("Run to scan", runs, key="sus-run") if runs else None
    if run_id and st.button("Flag suspicious correct traces"):
        n = flag_suspicious(run_id, policy_spec().id)
        st.success(f"queued {n} traces")
with engine.connect() as conn:
    sus = conn.execute(
        text(
            """
            SELECT id, question_id, sample_index, reason, status, raw_output
            FROM suspicious_samples ORDER BY id DESC LIMIT 20
            """
        )
    ).mappings().all()
if not sus:
    st.caption("Empty. Flag after a policy contest run.")
for s in sus:
    with st.expander(f"{s['question_id']} k={s['sample_index']} {s['reason']} {s['status']}"):
        st.code((s["raw_output"] or "")[:2000])
