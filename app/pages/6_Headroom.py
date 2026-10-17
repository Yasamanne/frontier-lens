import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Dev only: drop cached frontier_lens modules so edits under src/ load on the
# next rerun. Sessions run in threads that share sys.modules, so purging while
# another session imports raises KeyError; keep it off outside local dev.
if os.environ.get("FRONTIER_LENS_DEV_RELOAD") == "1":
    for _name in list(sys.modules):
        if _name == "frontier_lens" or _name.startswith("frontier_lens."):
            del sys.modules[_name]

import streamlit as _st_nav
import streamlit as st

_st_nav.markdown("""
<style>.nav-bar{display:flex;align-items:center;justify-content:space-between;padding:0 0 16px 0;border-bottom:1px solid #2D2D55;margin-bottom:32px;}.nav-links{display:flex;gap:0;align-items:center;}a.nav-link{display:inline-block;padding:8px 20px;color:#EDE8DC!important;font-size:.88rem;font-weight:500;text-decoration:none!important;border-bottom:2px solid transparent;transition:all .15s;letter-spacing:.01em;outline:none;}a.nav-link:hover{color:#FF8F5C!important;border-bottom-color:#FF6B2B;}a.nav-link.active{color:#FF8F5C!important;font-weight:700;border-bottom-color:#FF6B2B;}</style><div class="nav-bar"><a href="/" target="_self" style="text-decoration:none"><svg width="150" height="22" viewBox="0 0 150 22" style="overflow:visible;"><defs><linearGradient id="ng2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#FF6B2B"/><stop offset="100%" stop-color="#C96FFF"/></linearGradient></defs><text x="0" y="17" font-family="Inter,sans-serif" font-size="15" font-weight="800" letter-spacing="-0.3" fill="url(#ng2)">Frontier Lens</text></svg></a><div class="nav-links"><a href="/Seed" target="_self" class="nav-link">Seed</a><a href="/Evaluate" target="_self" class="nav-link">Evaluate</a><a href="/Weaknesses" target="_self" class="nav-link">Weaknesses</a><a href="/Generate_Review" target="_self" class="nav-link">Generate</a><a href="/Dataset" target="_self" class="nav-link">Dataset</a><a href="/Headroom" target="_self" class="nav-link active">Headroom</a></div></div>
""", unsafe_allow_html=True)

_current_role = st.session_state.get("role", "Viewer")
_role_cols = st.columns([6, 1])
with _role_cols[1]:
    _new_role = st.selectbox(
        "Role", ["Organizer", "SME", "Viewer"],
        index=["Organizer", "SME", "Viewer"].index(_current_role),
        key="role_select_nav", label_visibility="collapsed",
    )
    if _new_role != _current_role:
        st.session_state["role"] = _new_role
        st.rerun()

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.failure_analysis import rl_suitable_questions, zero_pass_questions

engine = get_engine()

st.title("📈 Headroom Growth")
st.caption(
    "Compares the original RL-trainable question pool from a baseline evaluation run "
    "against the questions added by the generation pipeline (rescue + augmentation)."
)

# ── Run / model selectors ─────────────────────────────────────────────────
with engine.connect() as conn:
    _all_runs = conn.execute(
        text("SELECT id, created_at FROM eval_runs WHERE status='complete' ORDER BY created_at DESC")
    ).all()

if not _all_runs:
    st.info("No completed evaluation runs yet. Run an evaluation first.")
    st.stop()

# ── Run selector first, then derive the models available for that run ─────
run_idx = st.selectbox(
    "Evaluation run",
    range(len(_all_runs)),
    format_func=lambda i: f"{_all_runs[i][0]}  ({_all_runs[i][1][:10]})",
    key="headroom_run",
)
run_id = _all_runs[run_idx][0]

with engine.connect() as conn:
    _run_models = conn.execute(
        text(
            "SELECT DISTINCT model_id FROM attempts "
            "WHERE run_id=:r ORDER BY model_id"
        ),
        {"r": run_id},
    ).scalars().all()

if not _run_models:
    st.info("No model attempts found for this run.")
    st.stop()

model_id = st.selectbox("Model", _run_models, key="headroom_model")

# ── Fetch baseline stats ──────────────────────────────────────────────────
rl_qs   = rl_suitable_questions(run_id, model_id)
zero_qs = zero_pass_questions(run_id, model_id)

n_rl_original = len(rl_qs)
n_zero        = len(zero_qs)

with engine.connect() as conn:
    # Fetch question IDs for this run+model (used for count and legacy-candidate matching)
    _run_qids = conn.execute(
        text(
            "SELECT DISTINCT question_id FROM attempts "
            "WHERE run_id=:r AND model_id=:m"
        ),
        {"r": run_id, "m": model_id},
    ).scalars().all()
    _run_qid_set = set(_run_qids)
    n_total_in_run = len(_run_qid_set)

    linked = conn.execute(
        text(
            """
            SELECT id, topic, band, generator_mode, status,
                   auto_reject_reason, policy_pass_rate, pass_at_k,
                   source_fail_ids, created_at, run_id
            FROM candidates
            WHERE generator_mode IN ('light_hint', 'similar_augment', 'one_shot')
              AND (
                  run_id = :run_id
                  OR (run_id IS NULL AND source_fail_ids IS NOT NULL)
              )
            """
        ),
        {"run_id": run_id},
    ).mappings().all()

    # For legacy rows (run_id IS NULL), keep only those whose source_fail_ids
    # overlap with question IDs that actually belong to this run+model.
    filtered = []
    for c in linked:
        if c["run_id"] is not None:
            filtered.append(c)
        else:
            try:
                src_ids = set(json.loads(c["source_fail_ids"] or "[]"))
            except Exception:
                src_ids = set()
            if src_ids & _run_qid_set:
                filtered.append(c)
    linked = filtered

rej_statuses = {"rejected", "auto_rejected", "review_failed"}

# Approved = SME confirmed, promoted to questions table → real new RL data
# Pending  = in SME review queue, not yet confirmed
n_approved  = sum(1 for c in linked if c["status"] == "approved")
n_pending   = sum(1 for c in linked if c["status"] == "pending")
n_total_gen = len(linked)

# "Confirmed" headroom = original + SME-approved only
# "Total potential" = original + approved + pending
n_confirmed = n_rl_original + n_approved
n_potential = n_rl_original + n_approved + n_pending
# Growth = (Total RL - Baseline RL) / Baseline RL × 100  (e.g. (7-4)/4 = 75%)
pct_growth  = round((n_confirmed - n_rl_original) / n_rl_original * 100) if n_rl_original else 0

# ── Top-line metrics ──────────────────────────────────────────────────────
st.divider()
if st.button("🔄 Refresh", help="Reload stats from the database"):
    st.rerun()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "In run", n_total_in_run,
    help="Distinct questions evaluated in this run",
)
m2.metric(
    "Baseline RL", n_rl_original,
    help="RL-suitable questions from this run (0 < pass_rate < 1)",
)
m3.metric(
    "New (approved)", n_approved,
    delta=f"+{n_approved}" if n_approved else None,
    help="Generated candidates approved by SME and promoted to the questions table",
)
m4.metric(
    "Total RL", n_confirmed,
    delta=f"+{n_approved}" if n_approved else None,
    help="Baseline + SME-approved new questions",
)
m5.metric(
    "Growth", f"{pct_growth}%" if n_rl_original else "N/A",
    delta=f"+{pct_growth}%" if pct_growth else None,
    help="(Total RL − Baseline RL) / Baseline RL × 100",
)

if n_pending:
    st.info(
        f"**{n_pending}** candidate(s) still in SME review queue — "
        f"potential total headroom if all approved: **{n_potential}**",
        icon="⏳",
    )

if n_total_gen == 0:
    st.info(
        "No generated candidates found. "
        "Run rescue or augmentation on the **Generate** page first."
    )
    st.stop()

# ── Process breakdown ─────────────────────────────────────────────────────
st.divider()
st.subheader("Process breakdown")

MODE_META = {
    "light_hint":      ("🔴 Dead-zone rescue",   "Rescued zero-pass questions by adding progressive hints"),
    "similar_augment": ("🟡 RL augmentation",    "Generated similar questions from RL-suitable seeds"),
    "one_shot":        ("🔵 One-shot train",     "Generated new train items targeting weak cells"),
}

breakdown_rows = []
for mode, (label, _desc) in MODE_META.items():
    mc = [c for c in linked if c["generator_mode"] == mode]
    if not mc:
        continue
    n_gen     = len(mc)
    n_appr_m  = sum(1 for c in mc if c["status"] == "approved")
    n_pend_m  = sum(1 for c in mc if c["status"] == "pending")
    n_rej_m   = sum(1 for c in mc if c["status"] in rej_statuses)
    rates     = [c["policy_pass_rate"] for c in mc if c["policy_pass_rate"] is not None]
    avg_pr    = f"{sum(rates)/len(rates):.2f}" if rates else "—"
    yield_    = f"{round(n_appr_m / n_gen * 100)}%" if n_gen else "—"
    breakdown_rows.append({
        "Process":        label,
        "Generated":      n_gen,
        "Approved ✅":    n_appr_m,
        "Pending ⏳":     n_pend_m,
        "Rejected ❌":    n_rej_m,
        "Yield (approved)": yield_,
        "Avg pass rate":  avg_pr,
    })

st.dataframe(breakdown_rows, width="stretch", hide_index=True)

# ── Rescue detail ─────────────────────────────────────────────────────────
rescue_cands = [c for c in linked if c["generator_mode"] == "light_hint"]
if n_zero > 0 and rescue_cands:
    st.divider()
    st.subheader("🔴 Dead-zone rescue detail")
    n_rescued = sum(1 for c in rescue_cands if c["status"] == "approved")
    n_rescue_pending = sum(1 for c in rescue_cands if c["status"] == "pending")
    rescue_rate = round(n_rescued / n_zero * 100)

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Dead-zone questions", n_zero, help="Questions the model always got wrong")
    rc2.metric("Approved rescues", n_rescued, delta=f"+{n_rescued}",
               help="Hints approved by SME — real new RL data")
    rc3.metric("Pending SME review", n_rescue_pending,
               help="Hint candidates still awaiting SME decision")
    rc4.metric("Rescue rate", f"{rescue_rate}%",
               help="Approved rescues / dead-zone questions")

# ── Topic / band breakdown ────────────────────────────────────────────────
st.divider()
st.subheader("Headroom by topic / band")

orig_cell: dict[tuple, int] = defaultdict(int)
for q in rl_qs:
    orig_cell[(q["topic"], q["band"])] += 1

new_cell: dict[tuple, int] = defaultdict(int)
for c in linked:
    if c["status"] == "approved" and c["topic"] and c["band"]:
        new_cell[(c["topic"], c["band"])] += 1

all_cells = sorted(set(orig_cell) | set(new_cell))
cell_rows = []
for topic, band in all_cells:
    before = orig_cell[(topic, band)]
    new    = new_cell[(topic, band)]
    cell_rows.append({
        "Topic":  topic,
        "Band":   band,
        "Before": before,
        "New":    new,
        "After":  before + new,
        "Δ":      f"+{new}" if new else "—",
    })

if cell_rows:
    st.dataframe(cell_rows, width="stretch", hide_index=True)

# ── Rejection breakdown ───────────────────────────────────────────────────
rej_linked = [c for c in linked if c["status"] in rej_statuses]
if rej_linked:
    st.divider()
    st.subheader("Rejection reasons")
    reasons = Counter(
        c["auto_reject_reason"] or c["status"]
        for c in rej_linked
    )
    st.dataframe(
        [{"Reason": r, "Count": n} for r, n in reasons.most_common()],
        width="stretch",
        hide_index=True,
    )

# ── Pass rate distribution ────────────────────────────────────────────────
pass_rates = [
    c["policy_pass_rate"]
    for c in linked
    if c["policy_pass_rate"] is not None and c["status"] == "approved"
]
if pass_rates:
    st.divider()
    st.subheader("Pass rate distribution — new RL-suitable candidates")
    st.caption(
        "Each candidate's estimated pass rate from calibration sampling. "
        "The learnable band is 0 < pass_rate < 1."
    )
    buckets = [0] * 10
    for r in pass_rates:
        buckets[min(int(r * 10), 9)] += 1
    dist_rows = [
        {"Pass rate range": f"{i*10}–{(i+1)*10}%", "Candidates": buckets[i]}
        for i in range(1, 10)   # skip 0–10% bucket (those would be out-of-band)
        if buckets[i] > 0
    ]
    if dist_rows:
        st.dataframe(dist_rows, width="stretch", hide_index=True)
    avg_overall = sum(pass_rates) / len(pass_rates)
    st.caption(f"Mean pass rate across {len(pass_rates)} new candidates: **{avg_overall:.2f}**")
