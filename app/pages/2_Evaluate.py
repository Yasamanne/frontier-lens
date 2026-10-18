import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Dev only: drop cached frontier_lens modules so edits under src/ load on the
# next rerun. Sessions run in threads that share sys.modules, so purging while
# another session imports raises KeyError; keep it off outside local dev.
if os.environ.get("FRONTIER_LENS_DEV_RELOAD") == "1":
    for _name in list(sys.modules):
        if _name == "frontier_lens" or _name.startswith("frontier_lens."):
            del sys.modules[_name]

from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="Frontier Lens · Evaluate",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Shared theme CSS + top nav ───────────────────────────────────────────
import streamlit as _st_nav
_st_nav.markdown("""
<style>.nav-bar{display:flex;align-items:center;justify-content:space-between;padding:0 0 16px 0;border-bottom:1px solid #2D2D55;margin-bottom:32px;}.nav-links{display:flex;gap:0;align-items:center;}a.nav-link{display:inline-block;white-space:nowrap;padding:8px clamp(8px,1.3vw,20px);color:#EDE8DC!important;font-size:.88rem;font-weight:500;text-decoration:none!important;border-bottom:2px solid transparent;transition:all .15s;letter-spacing:.01em;outline:none;}a.nav-link:hover{color:#FF8F5C!important;border-bottom-color:#FF6B2B;}a.nav-link.active{color:#FF8F5C!important;font-weight:700;border-bottom-color:#FF6B2B;}.stButton>button,.stDownloadButton>button,[data-testid^="stBaseButton"]{height:auto!important;min-height:2.5rem;white-space:normal!important;overflow:visible!important;line-height:1.3;}.stButton>button p,.stDownloadButton>button p,[data-testid^="stBaseButton"] p{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;overflow-wrap:break-word;}</style><div class="nav-bar"><a href="/" target="_self" style="text-decoration:none"><svg width="150" height="22" viewBox="0 0 150 22" style="overflow:visible;"><defs><linearGradient id="ng2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#FF6B2B"/><stop offset="100%" stop-color="#C96FFF"/></linearGradient></defs><text x="0" y="17" font-family="Inter,sans-serif" font-size="15" font-weight="800" letter-spacing="-0.3" fill="url(#ng2)">Frontier Lens</text></svg></a><div class="nav-links"><a href="/Seed" target="_self" class="nav-link">Seed</a><a href="/Evaluate" target="_self" class="nav-link active">Evaluate</a><a href="/Gap_Analysis" target="_self" class="nav-link">Gap Analysis</a><a href="/Generate_Review" target="_self" class="nav-link">Generate</a><a href="/Dataset" target="_self" class="nav-link">Dataset</a><a href="/Headroom" target="_self" class="nav-link">Headroom</a></div></div>
""", unsafe_allow_html=True)

# ── Role selector ────────────────────────────────────────────────────────
_current_role = st.session_state.get("role", "Viewer")
_role_cols = st.columns([4, 1])
with _role_cols[1]:
    _new_role = st.selectbox(
        "Role", ["Organizer", "SME", "Viewer"],
        index=["Organizer", "SME", "Viewer"].index(_current_role),
        key="role_select_nav", label_visibility="collapsed",
    )
    if _new_role != _current_role:
        st.session_state["role"] = _new_role
        st.rerun()
role = st.session_state.get("role", "Viewer")

from sqlalchemy import text

from frontier_lens import eval_job, runstate
from frontier_lens.config import OPENROUTER_API_KEY
from frontier_lens.contest import generate_results_csv, run_config, update_key_and_regrade
from frontier_lens.db import get_engine
from frontier_lens.registry import judge_spec, load_yaml
from frontier_lens.runlog import logs_for_download

st.title("Evaluate")

engine = get_engine()

# ── Provider status ───────────────────────────────────────────────────────
if OPENROUTER_API_KEY:
    st.sidebar.success("OpenRouter key set")
else:
    st.sidebar.error("OPENROUTER_API_KEY missing — live evaluation unavailable")

# ── Load model roster ─────────────────────────────────────────────────────
all_specs = load_yaml()

# Primary models: Gemma family only (policy + reference)
primary_specs = [m for m in all_specs if m.role in {"policy", "reference"} and m.family == "gemma"]
primary_ids   = [m.id for m in primary_specs]

# Verifier candidates: populated dynamically after primary selection (any non-generator, not in primary)

# ── Question counts ───────────────────────────────────────────────────────
with engine.connect() as conn:
    seed_gen_count = conn.execute(
        text("SELECT COUNT(*) FROM questions WHERE source IN ('seed','generated')")
    ).scalar() or 0
    upload_files = conn.execute(
        text("""
            SELECT COALESCE(upload_filename,'(unnamed)') as fname, COUNT(*) as n
            FROM questions WHERE source='uploaded'
            GROUP BY upload_filename ORDER BY fname
        """)
    ).all()

n_uploaded = sum(r[1] for r in upload_files)
upload_filenames = [r[0] for r in upload_files]
upload_file_counts = {r[0]: r[1] for r in upload_files}
total_q = seed_gen_count + n_uploaded

# ── Controls ──────────────────────────────────────────────────────────────
st.subheader("Run configuration")

col_primary, col_verifier = st.columns(2)
with col_primary:
    chosen = st.multiselect(
        "Primary models",
        primary_ids,
        default=primary_ids[:1],
        help="Gemma 26B (policy) and/or Gemma 31B (reference). Both are scored k times per question.",
    )
with col_verifier:
    # Any non-generator model that is not already selected as primary
    verifier_options = ["(none)"] + [
        m.id for m in all_specs
        if m.role != "generator" and m.id not in chosen
    ]
    verifier_choice = st.selectbox(
        "Verifier model (optional)",
        verifier_options,
        help=(
            "Any model except the generator and the selected primary model(s). "
            "Runs k times only on questions where the primary got 0/k correct. "
            "needs_sme_review = True only if both primary AND verifier fail."
        ),
    )
    verifier_id = None if verifier_choice == "(none)" else verifier_choice

# Judge: auto-selected from models.yaml (role: judge), toggle to enable/disable
_judge_spec = judge_spec()
if _judge_spec:
    use_judge = st.checkbox(
        f"Use LLM judge for non-exact matches  (`{_judge_spec.id}` — {_judge_spec.model})",
        value=True,
        help=(
            "When exact match fails, the judge checks semantic equivalence "
            "(e.g. 'B' == '$B$', 'Yes' == 'yes'). "
            "Adds ~1 API call per failed sample."
        ),
    )
    judge_id = _judge_spec.id if use_judge else None
else:
    judge_id = None
    st.caption("No judge model configured in `models.yaml` (add `role: judge` entry to enable).")

col_k, col_limit, col_source = st.columns(3)
with col_k:
    override_k = st.checkbox("Override k (default: policy=8, reference=3)")
    k = st.number_input("k samples", 1, 8, 4, key="k_val") if override_k else None
with col_limit:
    use_limit = st.checkbox("Limit number of items")
    limit = (
        st.number_input("Items to score", 1, max(total_q, 1), min(10, max(total_q, 1)), key="limit_val")
        if use_limit else None
    )
with col_source:
    if upload_filenames:
        file_options = ["All questions"] + [
            f"{fn} ({upload_file_counts[fn]} rows)" for fn in upload_filenames
        ]
        file_choice = st.selectbox("Source filter", file_options, index=0)
        if file_choice == "All questions":
            uploaded_only, selected_filename = False, None
        else:
            uploaded_only = True
            selected_filename = upload_filenames[file_options.index(file_choice) - 1]
    else:
        uploaded_only, selected_filename = False, None
        st.caption("No uploaded questions in DB.")

# "Train split only" is only relevant for seed + generated questions;
# uploaded questions already have their split set per-row.
if not uploaded_only:
    train_only = st.checkbox("Train split only (exclude eval items)", value=True)
else:
    train_only = False  # irrelevant for uploaded source — each row has its own split

# ── Batch source label ────────────────────────────────────────────────────
_now_str = datetime.now().strftime("%Y%m%d_%H%M")
if uploaded_only:
    source_label = f"uploaded ({selected_filename or 'all files'})"
else:
    source_label = f"generated_{_now_str}"

# ── Cost estimate ─────────────────────────────────────────────────────────
if chosen:
    k_eff = int(k) if k is not None else 8
    if uploaded_only and selected_filename:
        n_items = upload_file_counts.get(selected_filename, 0)
    elif uploaded_only:
        n_items = n_uploaded
    else:
        n_items = total_q
    if limit is not None:
        n_items = min(n_items, int(limit))
    n_calls_primary = n_items * k_eff * len(chosen)
    est_verifier = f" + up to {n_items * k_eff} verifier calls" if verifier_id else ""
    st.info(
        f"**Source:** `{source_label}` — "
        f"**{n_items} questions × k={k_eff} × {len(chosen)} model(s) = "
        f"{n_calls_primary} API calls{est_verifier}** "
        f"({'~' + str(round(n_calls_primary * 15 / 60)) + ' min est.' if n_calls_primary > 5 else 'quick'})"
    )

# ── Helpers ───────────────────────────────────────────────────────────────

def _resolve_question_ids(
    uploaded_only: bool,
    selected_filename: str | None,
    train_only: bool,
    limit: int | None,
) -> list[str]:
    """Return the ordered list of question IDs that would be scored."""
    with engine.connect() as conn:
        if uploaded_only:
            sql = "SELECT id FROM questions WHERE source='uploaded' AND COALESCE(sme_flagged, 0) != 2"
            params: dict = {}
            if selected_filename:
                sql += " AND COALESCE(upload_filename,'(unnamed)')=:fname"
                params["fname"] = selected_filename
            sql += " ORDER BY id"
            qids = conn.execute(text(sql), params).scalars().all()
        else:
            sql = (
                "SELECT id FROM questions "
                "WHERE source IN ('seed','generated','uploaded') "
                "AND COALESCE(sme_flagged, 0) != 2"
            )
            if train_only:
                sql += " AND split='train'"
            sql += " ORDER BY level, id"
            qids = conn.execute(text(sql)).scalars().all()
    qids = list(qids)
    if limit is not None:
        qids = qids[:int(limit)]
    return qids


def _already_evaluated(
    qids: list[str], model_ids: list[str], k_eff: int
) -> tuple[list[str], dict[str, int], dict[str, list[str]]]:
    """Return (all_done_qids, existing_k_by_qid, per_model_done).

    - ``all_done_qids``: questions where EVERY selected model has ≥ k_eff samples.
    - ``existing_k_by_qid``: min sample count across models for each done question.
    - ``per_model_done``: {model_id → [qids already done for that model]}.

    Uses cumulative counts across all completed runs, so a previous k=8 run
    satisfies a new k=4 request, but a previous k=4 run does NOT satisfy k=8.
    """
    if not qids or not model_ids:
        return [], {}, {mid: [] for mid in model_ids}

    q_placeholders = ",".join(f":q{i}" for i in range(len(qids)))
    m_placeholders = ",".join(f":m{i}" for i in range(len(model_ids)))
    params: dict = {}
    for i, qid in enumerate(qids):
        params[f"q{i}"] = qid
    for i, mid in enumerate(model_ids):
        params[f"m{i}"] = mid

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT a.question_id, a.model_id, COUNT(*) as n
                FROM attempts a
                JOIN eval_runs r ON r.id = a.run_id
                WHERE r.status = 'complete'
                  AND a.question_id IN ({q_placeholders})
                  AND a.model_id IN ({m_placeholders})
                GROUP BY a.question_id, a.model_id
                """
            ),
            params,
        ).mappings().all()

    from collections import defaultdict
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        counts[r["question_id"]][r["model_id"]] = r["n"]

    # Per-model: questions where this specific model has ≥ k_eff samples
    per_model_done: dict[str, list[str]] = {mid: [] for mid in model_ids}
    for qid in qids:
        model_counts = counts.get(qid, {})
        for mid in model_ids:
            if model_counts.get(mid, 0) >= k_eff:
                per_model_done[mid].append(qid)

    # All-done: questions where EVERY model is done
    needed = set(model_ids)
    all_done: list[str] = []
    existing_k: dict[str, int] = {}
    for qid in qids:
        model_counts = counts.get(qid, {})
        if needed <= set(model_counts.keys()) and all(
            model_counts[m] >= k_eff for m in needed
        ):
            all_done.append(qid)
            existing_k[qid] = min(model_counts[m] for m in needed)

    return all_done, existing_k, per_model_done


# ── Background runs ───────────────────────────────────────────────────────
# Runs execute in a detached process (frontier_lens.eval_job) and report
# through SQLite, so refreshing or leaving this page does not stop them.
runstate.reap_stale()
_active = runstate.active_runs()


def _counts_for_phase(p: dict) -> tuple[str, dict, int | None]:
    if p["phase"] == "verifier":
        return f"Verifier {p['verifier']}", p["verifier_counts"], p["planned_verifier"]
    return "Primary", p["primary"], p["planned_primary"]


@st.fragment(run_every=2)
def _run_panel(run_id: str) -> None:
    if not runstate.is_active(runstate.load_run(run_id) or {}):
        st.rerun()  # full rerun: the panel goes away and the summary shows
    p = runstate.progress(run_id)
    label, c, planned = _counts_for_phase(p)
    if planned:
        st.progress(min(c["done"] / planned, 1.0), text=f"{label}: {c['done']}/{planned} samples")
    else:
        st.progress(0.0, text="Starting…" if p["phase"] in (None, "primary") else f"{label}…")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Correct", c["correct"])
    col_b.metric("Wrong", c["wrong"])
    col_c.metric("Unparsable", c["unparsable"])
    col_d.metric("Samples", f"{c['done']} / {planned or '?'}")
    st.caption(
        f"Run `{run_id}` runs in the background — refreshing or leaving this page "
        f"does not stop it. Log: `logs/{run_id}.stdout.log`"
    )
    log = eval_job.tail(run_id)
    if log:
        st.code(log, language=None)
    if role == "Organizer" and st.button("⏹ Stop run", key=f"eval-stop-{run_id}"):
        eval_job.stop(run_id)
        st.rerun()


# The run this session started or resumed; summarise it once it stops,
# even if it finished while the user was on another page.
_watched = st.session_state.get("eval_watch_run")
_finished = None
if _watched and _watched not in {r["id"] for r in _active}:
    _finished = st.session_state.pop("eval_watch_run")
if _finished:
    _fp = runstate.progress(_finished)
    st.session_state["eval_last_result"] = _finished
    st.session_state["last_run_id"] = _finished
    if _fp["status"] == "complete":
        st.session_state["sme_current_run_qids"] = runstate.zero_correct_questions(
            _finished, _fp["models"] + ([_fp["verifier"]] if _fp["verifier"] else [])
        )
        st.session_state["sme_current_run_id"] = _finished

if _active:
    st.subheader("Run in progress")
    for _run in _active:
        _run_panel(_run["id"])
elif st.session_state.get("eval_last_result"):
    _lp = runstate.progress(st.session_state["eval_last_result"])
    _pc = _lp["primary"]
    if _lp["status"] == "complete":
        _acc = _pc["correct"] / _pc["done"] if _pc["done"] else 0
        st.success(
            f"Run `{_lp['run_id']}` — **{_pc['correct']}/{_pc['done']}** correct ({_acc:.0%}), "
            f"{_pc['unparsable']} unparsable"
        )
        if _lp["verifier"]:
            _vc = _lp["verifier_counts"]
            if _vc["done"]:
                st.success(
                    f"Verifier `{_lp['verifier']}`: {_vc['correct']}/{_vc['done']} correct "
                    "on questions the primary model(s) got 0/k."
                )
            else:
                st.success("Primary model answered all questions correctly at least once — no verifier needed.")
        _n_sme = len(st.session_state.get("sme_current_run_qids") or [])
        if _n_sme:
            st.warning(
                f"**{_n_sme} question(s)** had 0 correct across all models "
                f"→ flagged for SME review (visible in the SME section below)."
            )
        else:
            st.success("No questions need SME review — every question had at least one correct answer.")
    elif _lp["status"] == "interrupted":
        st.warning(
            f"Run `{_lp['run_id']}` stopped early ({_lp['error'] or 'process ended'}). "
            f"{_pc['done']} sample(s) are saved; resume it from **Download results** below."
        )
    elif _lp["status"] == "failed":
        st.error(
            f"Run `{_lp['run_id']}` failed: {_lp['error'] or 'see log'}. "
            f"{_pc['done']} sample(s) are saved; resume it from **Download results** below."
        )

# ── Run button ────────────────────────────────────────────────────────────
if role == "Viewer":
    st.warning("Viewer role — evaluation is read-only. Switch to Organizer to run live evaluation.")

can_run = bool(chosen) and role == "Organizer" and bool(OPENROUTER_API_KEY) and not _active

if _active:
    # One run at a time (single SQLite writer, shared API rate limits), whatever the dataset.
    _blk = _active[0]
    _bp = runstate.progress(_blk["id"])
    _bplanned = _bp["planned_primary"]
    st.info(
        f"🔒 Run `{_blk['id']}` (batch: {_blk['notes'] or 'live'}) is in progress"
        + (f", {_bp['primary']['done']}/{_bplanned} samples" if _bplanned else "")
        + ". Only one evaluation runs at a time, so Evaluate unlocks when it finishes"
        + (" or you stop it above." if role == "Organizer" else ".")
    )

# ── Overlap detection state machine ──────────────────────────────────────
# State: None → detect → waiting_choice → running
_eval_state = st.session_state.get("eval_state", None)

if st.button("▶ Evaluate", type="primary", disabled=not can_run, key="eval_btn_main"):
    # Clear SME state from any previous run immediately
    st.session_state["sme_current_run_qids"] = None
    st.session_state["sme_current_run_id"] = None
    # Step 1: resolve question IDs and check for overlap
    all_qids = _resolve_question_ids(uploaded_only, selected_filename, train_only, limit)
    k_eff_check = int(k) if k is not None else 8
    already_done, existing_k_map, per_model_done = _already_evaluated(all_qids, chosen, k_eff_check)
    # Union of new questions: questions that at least one model hasn't evaluated
    new_qids = [q for q in all_qids if any(
        q not in set(per_model_done[m]) for m in chosen
    )]

    if already_done or any(per_model_done[m] for m in chosen):
        st.session_state["eval_state"] = "waiting_choice"
        st.session_state["eval_all_qids"] = all_qids
        st.session_state["eval_new_qids"] = new_qids
        st.session_state["eval_already_done"] = already_done
        st.session_state["eval_existing_k_map"] = existing_k_map
        st.session_state["eval_per_model_done"] = per_model_done
        st.session_state["eval_k_requested"] = k_eff_check
        st.session_state["eval_chosen"] = chosen
        st.session_state["eval_verifier_id"] = verifier_id
        st.session_state["eval_judge_id"] = judge_id
        st.session_state["eval_k"] = k
        st.session_state["eval_source_label"] = source_label
        st.rerun()
    else:
        # No overlap — go straight to run
        st.session_state["eval_state"] = "running"
        st.session_state["eval_all_qids"] = all_qids
        st.session_state["eval_new_qids"] = all_qids
        st.session_state["eval_already_done"] = []
        st.session_state["eval_chosen"] = chosen
        st.session_state["eval_verifier_id"] = verifier_id
        st.session_state["eval_judge_id"] = judge_id
        st.session_state["eval_k"] = k
        st.session_state["eval_source_label"] = source_label
        st.rerun()

# ── Overlap choice UI ─────────────────────────────────────────────────────
if st.session_state.get("eval_state") == "waiting_choice":
    already_done    = st.session_state["eval_already_done"]
    new_qids        = st.session_state["eval_new_qids"]
    all_qids        = st.session_state["eval_all_qids"]
    existing_k_map  = st.session_state.get("eval_existing_k_map", {})
    per_model_done  = st.session_state.get("eval_per_model_done", {})
    k_requested     = st.session_state.get("eval_k_requested", 8)
    _chosen         = st.session_state.get("eval_chosen", [])

    # Summarise previous k values
    prev_k_values = sorted(set(existing_k_map.values()))
    prev_k_str = (
        f"k={prev_k_values[0]}" if len(prev_k_values) == 1
        else f"k={prev_k_values[0]}–{prev_k_values[-1]}"
    ) if prev_k_values else f"k={k_requested}"

    if k_requested > max(prev_k_values, default=0) and prev_k_values:
        k_note = (
            f" ⚠ Previous results used {prev_k_str} < k={k_requested} requested — "
            f"reusing means those questions have fewer samples than requested."
        )
    else:
        k_note = ""

    # Per-model breakdown table
    model_rows = []
    for mid in _chosen:
        done_count = len(per_model_done.get(mid, []))
        new_count  = len(all_qids) - done_count
        model_rows.append({
            "Model": mid,
            "Already evaluated": done_count,
            "New (needs scoring)": new_count,
            f"k (requested)": k_requested,
        })
    st.dataframe(model_rows, width="stretch")

    if len(new_qids) == 0:
        st.success(
            f"✅ All **{len(all_qids)} question(s)** are fully evaluated for every selected model "
            f"(≥k={k_requested} samples).{k_note}  Nothing new to score."
        )
    else:
        st.warning(
            f"Some question–model combinations have existing results.{k_note}  "
            f"**{len(new_qids)} question(s)** need scoring for at least one model."
        )

    col_skip, col_rerun = st.columns(2)
    if col_skip.button(
        f"▶ Score only what's missing per model  ({len(new_qids)} question(s) in scope)",
        type="primary", key="eval_skip_existing",
        disabled=len(new_qids) == 0,
        help="Each model will skip questions it has already fully evaluated. "
             "Questions new to at least one model will be scored.",
    ):
        st.session_state["eval_state"] = "running"
        st.session_state["eval_reuse_prev"] = True
        st.rerun()

    if col_rerun.button(
        f"🔁 Re-evaluate everything at k={k_requested}",
        key="eval_rerun_all",
        help="Ignores all previous results. Runs all questions for all models from scratch.",
    ):
        st.session_state["eval_state"] = "running"
        st.session_state["eval_new_qids"] = all_qids
        st.session_state["eval_reuse_prev"] = False
        st.rerun()

    st.stop()

# ── Actual run ────────────────────────────────────────────────────────────
if st.session_state.get("eval_state") == "running":
    chosen       = st.session_state.pop("eval_chosen")
    verifier_id  = st.session_state.pop("eval_verifier_id")
    judge_id     = st.session_state.pop("eval_judge_id", None)
    k            = st.session_state.pop("eval_k")
    source_label = st.session_state.pop("eval_source_label")
    qids_to_run  = st.session_state.pop("eval_new_qids")
    reuse_prev   = st.session_state.pop("eval_reuse_prev", False)
    st.session_state.pop("eval_all_qids", None)
    st.session_state.pop("eval_already_done", None)
    st.session_state.pop("eval_per_model_done", None)
    st.session_state["eval_state"] = None

    if not qids_to_run:
        st.success("All selected questions already have complete results. Nothing to run.")
        st.stop()

    # Primary contest, verifier, and SME flagging all run in the background job.
    try:
        run_id = eval_job.start(
            run_config(
                chosen, k, qids_to_run,
                notes=source_label,
                reuse_prev_runs=reuse_prev,
                judge_model_id=judge_id,
                verifier_model=verifier_id,
            )
        )
    except eval_job.RunInProgress:
        # Someone else (another tab, the CLI) started a run after this page rendered.
        st.warning("Another evaluation run just started. Wait for it to finish, then try again.")
        st.stop()
    st.session_state.pop("eval_last_result", None)
    st.session_state["eval_watch_run"] = run_id
    st.session_state["last_run_id"] = run_id
    st.rerun()

# ── CSV download ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Download results")

with engine.connect() as conn:
    all_runs = conn.execute(
        text("""
            SELECT r.id, r.created_at, r.status, r.k, r.notes, r.config_json, r.error,
                   GROUP_CONCAT(DISTINCT a.model_id) as models,
                   COUNT(DISTINCT a.question_id) as n_items,
                   COUNT(*) as n_samples
            FROM eval_runs r
            LEFT JOIN attempts a ON a.run_id = r.id
            GROUP BY r.id ORDER BY r.created_at DESC LIMIT 20
        """)
    ).mappings().all()

if not all_runs:
    st.info("No evaluation runs yet.")
else:
    STATUS_ICON = {
        "complete": "🟢", "failed": "🔴", "running": "🟡",
        "queued": "⏳", "interrupted": "🟠",
    }
    last_run = st.session_state.get("last_run_id")
    run_ids = [r["id"] for r in all_runs]
    default_idx = run_ids.index(last_run) if last_run in run_ids else 0

    run_labels = [
        f"{STATUS_ICON.get(r['status'], '⚪')} {r['id']}  "
        f"({r['status']}) — {(r['models'] or '').replace(',', ', ')} — "
        f"{r['n_items'] or 0} items × {r['n_samples'] or 0} samples — "
        f"batch: {r['notes'] or 'live'} — "
        f"{r['created_at'][:16]}"
        for r in all_runs
    ]
    selected_idx = st.selectbox(
        "Select run", range(len(all_runs)), index=default_idx,
        format_func=lambda i: run_labels[i], key="dl_run_select",
    )
    selected_run = all_runs[selected_idx]
    sel_id = selected_run["id"]
    sel_cfg = runstate.run_config(dict(selected_run))

    if selected_run["status"] in runstate.RESUMABLE:
        if sel_cfg:
            col_msg, col_resume = st.columns([3, 1])
            col_msg.warning(
                f"This run stopped early ({selected_run['error'] or selected_run['status']}). "
                "Resume samples only what is missing from it."
            )
            needs_key = not sel_cfg.get("replay")
            if col_resume.button(
                "▶ Resume run", key=f"eval-resume-{sel_id}",
                disabled=role != "Organizer" or bool(_active) or (needs_key and not OPENROUTER_API_KEY),
                help="Organizer only; disabled while another run is in progress.",
            ):
                try:
                    eval_job.launch(sel_id)
                except eval_job.RunInProgress:
                    st.warning("Another evaluation run just started. Resume once it finishes.")
                    st.stop()
                st.session_state.pop("eval_last_result", None)
                st.session_state["eval_watch_run"] = sel_id
                st.session_state["last_run_id"] = sel_id
                st.rerun()
        else:
            st.caption("This run predates resumable runs, so it cannot be resumed. Start a new evaluation instead.")

    col_info, col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 1, 1])
    with col_info:
        st.caption(
            f"**{selected_run['status']}**  •  "
            f"models: {(selected_run['models'] or '—').replace(',', ', ')}  •  "
            f"k={selected_run['k']}  •  "
            f"{selected_run['n_items'] or 0} questions  •  "
            f"{selected_run['n_samples'] or 0} total samples"
        )
    try:
        files = logs_for_download(sel_id)
        with col_dl1:
            st.download_button(
                "⬇ .txt log", data=files["txt"][1],
                file_name=files["txt"][0], mime="text/plain",
                key=f"dl-txt-{sel_id}",
            )
        with col_dl2:
            st.download_button(
                "⬇ .jsonl log", data=files["jsonl"][1],
                file_name=files["jsonl"][0], mime="application/jsonl",
                key=f"dl-jsonl-{sel_id}",
            )
        with col_dl3:
            # Wide CSV with per-response columns
            sel_verifier = sel_cfg.get("verifier_model") if sel_cfg else None
            with engine.connect() as conn:
                run_model_ids = conn.execute(
                    text("SELECT DISTINCT model_id FROM attempts WHERE run_id=:r"),
                    {"r": sel_id},
                ).scalars().all()
                run_ts = conn.execute(
                    text("SELECT created_at FROM eval_runs WHERE id=:r"),
                    {"r": sel_id},
                ).scalar() or ""
            # The verifier gets its own columns, so keep it out of the primary list.
            primary_models_for_run = [m for m in run_model_ids if m != sel_verifier]
            # Format timestamp for filename: 20261016_1139
            try:
                from datetime import datetime as _dt
                _ts_fmt = _dt.fromisoformat(run_ts.replace("Z", "+00:00")).strftime("%Y%m%d_%H%M")
            except Exception:
                _ts_fmt = run_ts[:16].replace(":", "").replace("-", "").replace("T", "_").replace(" ", "_")
            csv_data = generate_results_csv(
                sel_id,
                primary_models_for_run,
                verifier_model_id=sel_verifier,
            )
            st.download_button(
                "⬇ results.csv", data=csv_data,
                file_name=f"results_{sel_id}_{_ts_fmt}.csv", mime="text/csv",
                key=f"dl-csv-{sel_id}",
            )
    except Exception as exc:
        st.error(f"Could not build download: {exc}")

    st.caption("→ Go to **Gap Analysis** to see the pass-rate histogram, dead-zone rescue, and RL-suitable augmentation.")

# ── SME review section ────────────────────────────────────────────────────
st.divider()
st.subheader("📋 SME Review — questions all models failed")
st.caption(
    "Questions that got 0 correct across all primary + verifier attempts. "
    "Review each one: the gold key may be wrong, the stem may be ambiguous, "
    "or the question may be genuinely beyond the model's capability."
)

_sme_run_qids  = st.session_state.get("sme_current_run_qids")   # None = no run yet this session
_sme_run_id    = st.session_state.get("sme_current_run_id")

if _active:
    # Reviewers can keep working on earlier flags while a run is in progress.
    st.info("⏳ Evaluation in progress — this run's failures are added when it completes.")
    _sme_filter_ids = None
elif _sme_run_qids is None:
    # No run has happened this session — show historical queue
    st.caption("Showing all historically flagged questions. Run an evaluation to see current-run results.")
    _sme_filter_ids = None   # None = show all
else:
    # A run just completed
    if _sme_run_qids:
        st.caption(
            f"Showing **{len(_sme_run_qids)} question(s)** flagged in run `{_sme_run_id}`. "
            "Use the toggle below to also see previously flagged questions."
        )
    else:
        st.success("✅ No questions failed in this run — SME queue is empty for this run.")
    _sme_filter_ids = _sme_run_qids   # show only current run's items

    show_all_sme = st.checkbox("Also show questions flagged in previous runs", value=False, key="sme_show_all")
    if show_all_sme:
        _sme_filter_ids = None   # show all

# Build query
import json as _json

def _sme_query(filter_ids: list | None):
    with engine.connect() as conn:
        if filter_ids is not None and len(filter_ids) == 0:
            return []
        if filter_ids:
            ph = ",".join(f":sq{i}" for i in range(len(filter_ids)))
            params = {f"sq{i}": qid for i, qid in enumerate(filter_ids)}
            params["flag"] = 1
            rows = conn.execute(
                text(
                    f"""
                    SELECT id, stem, key, topic, band, source, params_json,
                           COALESCE(sme_source, 'evaluate') as sme_source
                    FROM questions
                    WHERE COALESCE(sme_flagged, 0) = 1 AND id IN ({ph})
                    ORDER BY topic, band
                    """
                ),
                params,
            ).mappings().all()
        else:
            rows = conn.execute(
                text(
                    """
                    SELECT id, stem, key, topic, band, source, params_json,
                           COALESCE(sme_source, 'evaluate') as sme_source
                    FROM questions
                    WHERE COALESCE(sme_flagged, 0) = 1
                    ORDER BY topic, band
                    """
                )
            ).mappings().all()
    return rows

sme_qs = _sme_query(_sme_filter_ids)

if not sme_qs and _sme_run_qids is None:
    st.info("No questions flagged for SME review. Run an evaluation to populate this queue.")
elif sme_qs:
    st.caption(f"**{len(sme_qs)}** question(s) awaiting review.")
    for q in sme_qs:
        params = _json.loads(q["params_json"] or "{}")
        sol = params.get("solution_steps", "")
        phase_label = "📊 Evaluate" if q["sme_source"] == "evaluate" else "🔁 Generate"
        with st.expander(
            f"{phase_label} | `{q['id'][:12]}` — {q['topic']}/{q['band']} | key: `{q['key']}`",
            expanded=True,
        ):
            st.text((q["stem"] or "")[:800])
            if sol:
                st.caption(f"**Solution steps:** {sol}")

            if role in {"Organizer", "SME"}:
                col_valid, col_broken, col_key = st.columns(3)

                if col_valid.button("✅ Mark valid", key=f"sme-valid-{q['id']}"):
                    with engine.begin() as conn2:
                        conn2.execute(
                            text(
                                "UPDATE questions SET sme_flagged=0, sme_validated_key=key "
                                "WHERE id=:id"
                            ),
                            {"id": q["id"]},
                        )
                    # Remove from current run's list too
                    if _sme_run_qids and q["id"] in _sme_run_qids:
                        st.session_state["sme_current_run_qids"] = [
                            x for x in _sme_run_qids if x != q["id"]
                        ]
                    st.success("Marked as valid — removed from SME queue.")
                    st.rerun()

                if col_broken.button("🗑 Mark broken (remove)", key=f"sme-broken-{q['id']}"):
                    with engine.begin() as conn2:
                        conn2.execute(
                            text("UPDATE questions SET sme_flagged=2 WHERE id=:id"),
                            {"id": q["id"]},
                        )
                    if _sme_run_qids and q["id"] in _sme_run_qids:
                        st.session_state["sme_current_run_qids"] = [
                            x for x in _sme_run_qids if x != q["id"]
                        ]
                    st.warning("Marked as broken — will be excluded from future runs.")
                    st.rerun()

                new_key = col_key.text_input(
                    "Edit gold key", value=q["key"], key=f"sme-key-{q['id']}"
                )
                if new_key != q["key"] and col_key.button("💾 Save key", key=f"sme-save-{q['id']}"):
                    n_regraded = update_key_and_regrade(q["id"], new_key)
                    if _sme_run_qids and q["id"] in _sme_run_qids:
                        st.session_state["sme_current_run_qids"] = [
                            x for x in _sme_run_qids if x != q["id"]
                        ]
                    st.success(
                        f"Key updated to `{new_key}`. Re-graded stored attempts "
                        f"({n_regraded} changed). Removed from SME queue."
                    )
                    st.rerun()
            else:
                st.info("Switch Role to **Organizer** or **SME** to take action.")
