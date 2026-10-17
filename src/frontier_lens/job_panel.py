"""Streamlit widgets for background jobs (frontier_lens.jobs): live panel, picker, resume."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import streamlit as st

from frontier_lens import jobs, runstate
from frontier_lens.config import OPENROUTER_API_KEY

STATUS_ICON = {
    "complete": "🟢", "failed": "🔴", "running": "🟡",
    "queued": "⏳", "interrupted": "🟠",
}


@st.fragment(run_every=2)
def live_panel(job_id: str, role: str, unit: str) -> None:
    """Progress, log tail and Stop for an active job; reruns the page when it ends."""
    row = jobs.load(job_id) or {}
    if not runstate.is_active(row):
        st.rerun()  # full rerun: the panel goes away and the results show
    planned, done = row.get("planned") or 0, row.get("done") or 0
    label = f"{done}/{planned} {unit}"
    if row.get("phase") and row["phase"] != "done":
        label += f" — {row['phase']}"
    st.progress(min(done / planned, 1.0) if planned else 0.0, text=label)
    st.caption(
        f"Job `{job_id}` runs in the background — refreshing or leaving this page "
        f"does not stop it. Log: `logs/op_{job_id}.stdout.log`"
    )
    log = jobs.tail(job_id)
    if log:
        st.code(log, language=None)
    if role == "Organizer" and st.button("⏹ Stop", key=f"job-stop-{job_id}"):
        jobs.stop(job_id)
        st.rerun()


def _label(row: dict) -> str:
    ts = row["created_at"][:16].replace("T", " ")
    try:
        ts = datetime.fromisoformat(row["created_at"]).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        pass
    return (
        f"{STATUS_ICON.get(row['status'], '⚪')} {row['id']} ({row['status']}) — "
        f"{row.get('done') or 0}/{row.get('planned') or 0} — {ts}"
    )


def pick_recent(kind: str, key: str, label: str = "Job") -> dict | None:
    """Select box over the last jobs of *kind*; newest first and selected by default.

    Set ``st.session_state[key] = job_id`` before a rerun to select a new job.
    """
    rows = {r["id"]: r for r in jobs.recent(kind)}
    if not rows:
        return None
    if st.session_state.get(key) not in rows:
        st.session_state.pop(key, None)
    job_id = st.selectbox(label, list(rows), format_func=lambda i: _label(rows[i]), key=key)
    return rows[job_id]


def resume_controls(row: dict, role: str, *, blocked: bool, needs_key: bool = True) -> None:
    """Explain a stopped job and offer Resume (Organizer only, one job of a kind at a time)."""
    if row["status"] not in runstate.RESUMABLE:
        return
    col_msg, col_btn = st.columns([3, 1])
    col_msg.warning(
        f"This job stopped early ({row.get('error') or row['status']}). "
        f"{row.get('done') or 0}/{row.get('planned') or 0} done; resume skips finished work."
    )
    if col_btn.button(
        "▶ Resume", key=f"job-resume-{row['id']}",
        disabled=role != "Organizer" or blocked or (needs_key and not OPENROUTER_API_KEY),
        help="Organizer only; disabled while another job of this kind is running.",
    ):
        jobs.launch(row["id"])
        st.rerun()


def job_section(
    kind: str,
    role: str,
    unit: str,
    render_results: Callable[[dict], None],
    *,
    running: dict | None,
    blocked: bool,
    needs_key: bool = True,
    label: str = "Job",
) -> None:
    """Live panel for the *running* job of *kind*, else the latest jobs with results and Resume."""
    if running:
        st.markdown("**Running job**")
        live_panel(running["id"], role, unit)
        return
    row = pick_recent(kind, key=f"{kind}_job_pick", label=label)
    if row is None:
        return
    resume_controls(row, role, blocked=blocked, needs_key=needs_key)
    render_results(row)
