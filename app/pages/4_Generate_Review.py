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

# ── Shared theme CSS + top nav (injected) ────────────────────────────────
import streamlit as _st_nav
import streamlit as st

st.set_page_config(
    page_title="Frontier Lens · Generate",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_st_nav.markdown("""
<style>.nav-bar{display:flex;align-items:center;justify-content:space-between;padding:0 0 16px 0;border-bottom:1px solid #2D2D55;margin-bottom:32px;}.nav-links{display:flex;gap:0;align-items:center;}a.nav-link{display:inline-block;white-space:nowrap;padding:8px clamp(8px,1.3vw,20px);color:#EDE8DC!important;font-size:.88rem;font-weight:500;text-decoration:none!important;border-bottom:2px solid transparent;transition:all .15s;letter-spacing:.01em;outline:none;}a.nav-link:hover{color:#FF8F5C!important;border-bottom-color:#FF6B2B;}a.nav-link.active{color:#FF8F5C!important;font-weight:700;border-bottom-color:#FF6B2B;}.stButton>button,.stDownloadButton>button,[data-testid^="stBaseButton"]{height:auto!important;min-height:2.5rem;white-space:normal!important;overflow:visible!important;line-height:1.3;}.stButton>button p,.stDownloadButton>button p,[data-testid^="stBaseButton"] p{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;overflow-wrap:break-word;}</style><div class="nav-bar"><a href="/" target="_self" style="text-decoration:none"><svg width="150" height="22" viewBox="0 0 150 22" style="overflow:visible;"><defs><linearGradient id="ng2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#FF6B2B"/><stop offset="100%" stop-color="#C96FFF"/></linearGradient></defs><text x="0" y="17" font-family="Inter,sans-serif" font-size="15" font-weight="800" letter-spacing="-0.3" fill="url(#ng2)">Frontier Lens</text></svg></a><div class="nav-links"><a href="/Seed" target="_self" class="nav-link">Seed</a><a href="/Evaluate" target="_self" class="nav-link">Evaluate</a><a href="/Gap_Analysis" target="_self" class="nav-link">Gap Analysis</a><a href="/Generate_Review" target="_self" class="nav-link active">Generate</a><a href="/Dataset" target="_self" class="nav-link">Dataset</a><a href="/Headroom" target="_self" class="nav-link">Headroom</a></div></div>
""", unsafe_allow_html=True)
# ── Persistent role selector (shown on every page) ───────────────────────
_current_role = st.session_state.get("role", "Viewer")
_role_cols = st.columns([4, 1])
with _role_cols[1]:
    _new_role = st.selectbox(
        "Role",
        ["Organizer", "SME", "Viewer"],
        index=["Organizer", "SME", "Viewer"].index(_current_role),
        key="role_select_nav",
        label_visibility="collapsed",
    )
    if _new_role != _current_role:
        st.session_state["role"] = _new_role
        st.rerun()
role = st.session_state.get("role", "Viewer")


from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.registry import policy_spec
from frontier_lens.review import (
    flag_suspicious,
    promote,
)

st.title("Generate")
engine = get_engine()

from frontier_lens import job_panel, jobs
from frontier_lens.failure_analysis import rl_suitable_questions, zero_pass_questions
from frontier_lens.feedback import REASONS
from frontier_lens.generate.utils import reason_label
from frontier_lens.feedback import context_for as sme_feedback_for

# ── Run / model selectors ─────────────────────────────────────────────────
with engine.connect() as conn:
    _all_runs = conn.execute(text("SELECT id, created_at FROM eval_runs WHERE status='complete' ORDER BY created_at DESC")).all()

if not _all_runs:
    st.info("No completed eval runs yet. Run an evaluation first.")
    st.stop()

_run_labels = [f"{r[0]}  —  {r[1][:16].replace('T',' ')}" for r in _all_runs]
_run_id = st.selectbox(
    "Eval run",
    [r[0] for r in _all_runs],
    format_func=lambda rid: next(
        (lbl for lbl, r in zip(_run_labels, _all_runs) if r[0] == rid), rid
    ),
    key="gen_run_sel2",
)
# Only models that have samples in the chosen run.
with engine.connect() as conn:
    _all_models = conn.execute(
        text("SELECT DISTINCT model_id FROM attempts WHERE run_id = :r ORDER BY 1"),
        {"r": _run_id},
    ).scalars().all()
if not _all_models:
    st.info("This run has no samples yet.")
    st.stop()
_pol_id   = policy_spec().id
_def_mi   = list(_all_models).index(_pol_id) if _pol_id in list(_all_models) else 0
_model_id = st.selectbox("Model", list(_all_models), index=_def_mi, key="gen_model_sel2")

st.divider()
st.subheader("⚙ Suspicious correct policy samples")
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

# ── Background jobs (rescue / augmentation) ───────────────────────────────
# Both run in a detached process (frontier_lens.jobs) and save each finished
# item to the database, so refreshing or leaving this page does not stop them.
_active_jobs = {r["kind"]: r for r in jobs.active(("rescue", "augment"))}
_AUG_MODES = {"🎯 One-shot": "one_shot", "🤖 ReAct agent + Review": "react_review"}


def _job_section(kind: str, unit: str, render_results) -> None:
    job_panel.job_section(
        kind, role, unit, render_results,
        running=_active_jobs.get(kind), blocked=False,
        label=f"Recent {'rescue' if kind == 'rescue' else 'augmentation'} jobs",
    )


def _render_rescue_results(row: dict) -> None:
    res = jobs.results(row)
    results, skipped = res["items"], res["skipped"]
    if not results and not skipped:
        st.caption("No questions finished yet.")
        return
    rescued = sum(1 for r in results if r["rl_suitable"])
    msg = f"**{rescued}/{len(results)}** rescued.  Log → `logs/op_{row['id']}.txt`"
    (st.success if row["status"] == "complete" else st.info)(msg)
    st.dataframe(
        [{"question_id": r["question_id"][:14], "rescued": r["rl_suitable"],
          "rescue_level": r.get("rescue_level") or "—",
          "pass_rate": round(r["pass_rate"], 3),
          "candidate_id": r.get("candidate_id") or "—",
          "hint_L1": (r["hint_rounds"][0]["hint"] if len(r.get("hint_rounds", [])) > 0 else "")[:60],
          "hint_L2": (r["hint_rounds"][1]["hint"] if len(r.get("hint_rounds", [])) > 1 else "")[:60],
          "hint_L3": (r["hint_rounds"][2]["hint"] if len(r.get("hint_rounds", [])) > 2 else "")[:60]}
         for r in results],
        width="stretch",
    )
    if skipped:
        st.caption(f"Skipped {len(skipped)} question(s): " + "; ".join(
            f"`{s['question_id'][:12]}` ({s.get('reason')})" for s in skipped
        ))
    unrescued = len(results) - rescued
    if unrescued > 0:
        from frontier_lens.generate import generate_rescue_sft_csv

        st.warning(
            f"**{unrescued}** question(s) unrescuable after "
            f"{jobs.config(row)['max_hints']} levels → SFT CSV below."
        )
        st.download_button(
            f"⬇ SFT CSV ({unrescued} unrescued)", data=generate_rescue_sft_csv(results),
            file_name=f"sft_unrescued_{row['id']}.csv", mime="text/csv",
            key=f"sft-dl-{row['id']}",
        )


def _render_augment_results(row: dict) -> None:
    results = jobs.results(row)["items"]
    if not results:
        st.caption("No candidates finished yet.")
        return
    rl_ok = sum(1 for r in results if r["status"] == "pending")
    msg = f"**{rl_ok}/{len(results)}** RL-suitable.  Log → `logs/op_{row['id']}.txt`"
    (st.success if row["status"] == "complete" else st.info)(msg)
    st.dataframe(
        [{"source_id": r["source_id"][:14], "candidate_id": r["candidate_id"],
          "status": r["status"],
          "reason": reason_label(r.get("reason"), r.get("pass_rate")),
          "pass_rate": round(r["pass_rate"], 3) if r["pass_rate"] is not None else None}
         for r in results],
        width="stretch",
    )


# ── Dead-zone rescue (pass@k = 0) ─────────────────────────────────────────
st.divider()
st.subheader("🔴 Dead-zone rescue  (pass@k = 0)")
st.caption(
    "Questions the model **never** answers correctly across all k attempts. "
    "The generator adds progressive hints (L1→L2→L3), calibrates after each level, "
    "and stops as soon as the question becomes RL-suitable. "
    "Unrescuable questions are exported as an SFT CSV."
)

dead_qs = zero_pass_questions(_run_id, _model_id)
if not dead_qs:
    st.info("No pass@k=0 questions found for this run/model.")
else:
    st.caption(f"**{len(dead_qs)}** question(s) with 0/{dead_qs[0]['n']} correct.")
    st.dataframe(
        [{"id": q["id"][:12], "topic": q["topic"], "band": q["band"],
          "k": q["n"], "stem (preview)": (q["stem"] or "")[:120]}
         for q in dead_qs],
        width="stretch",
    )

    if role == "Organizer":
        _AVG = 15
        col_rlim2, col_rk2, col_rh = st.columns(3)
        rescue_limit = col_rlim2.number_input(
            "Rescue at most N questions", min_value=1,
            max_value=len(dead_qs), value=min(5, len(dead_qs)), key="rescue_limit",
        )
        rescue_k = col_rk2.slider(
            "Calibration k per hint level", min_value=1, max_value=8, value=4,
            help="Policy samples checked after each hint level.", key="rescue_k",
        )
        rescue_max_hints = col_rh.slider(
            "Max hint levels (1–3)", min_value=1, max_value=3, value=3,
            help="L1=domain framing | L2=technique | L3=common mistake. Generator knows gold key.",
            key="rescue_max_hints",
        )
        _est = int(rescue_limit) * int(rescue_max_hints) * (1 + int(rescue_k))
        st.caption(
            f"~{_est} API calls ({int(rescue_limit)}q × {int(rescue_max_hints)} levels × "
            f"(1 gen + {int(rescue_k)} samples)) ≈ **{round(_est*_AVG/60,1)} min**"
        )

        if st.button(
            "💡 Progressive hint rescue", key="rescue_btn", disabled="rescue" in _active_jobs,
            help="Runs in the background; one rescue job at a time.",
        ):
            ids_to_rescue = [q["id"] for q in dead_qs[: int(rescue_limit)]]
            st.session_state["rescue_job_pick"] = jobs.start(
                "rescue",
                {
                    "run_id": _run_id, "model_id": _model_id, "question_ids": ids_to_rescue,
                    "calibration_k": int(rescue_k), "max_hints": int(rescue_max_hints),
                },
                planned=len(ids_to_rescue),
            )
            st.rerun()
    else:
        st.info("Switch Role to **Organizer** to run the rescue.")

_job_section("rescue", "questions", _render_rescue_results)

# ── RL-suitable augmentation ──────────────────────────────────────────────
st.divider()
st.subheader("🟡 RL-suitable augmentation  (0 < correct < k)")
st.caption(
    "Questions already in the learnable band. The generator produces similar-but-distinct "
    "questions. Each is calibrated on the policy: candidates the policy always or never "
    "solves are rejected automatically (`out_of_band`); only in-band ones land in the SME Queue below."
)

rl_qs = rl_suitable_questions(_run_id, _model_id)
if not rl_qs:
    st.info("No RL-suitable questions found for this run/model.")
else:
    st.caption(f"**{len(rl_qs)}** RL-suitable question(s) available as seeds.")
    st.dataframe(
        [{"id": q["id"][:12], "topic": q["topic"], "band": q["band"],
          "correct": int(q["n_correct"]), "k": int(q["n"]),
          "pass_rate": f"{int(q['n_correct'])/int(q['n']):.0%}",
          "stem (preview)": (q["stem"] or "")[:120]}
         for q in rl_qs],
        width="stretch",
    )

    if role == "Organizer":
        _AVG = 15
        col_aug_n, col_aug_each, col_aug_k = st.columns(3)
        aug_limit = col_aug_n.number_input(
            "Use at most N seeds", min_value=1, max_value=len(rl_qs),
            value=min(5, len(rl_qs)), key="aug_limit",
        )
        aug_each = col_aug_each.number_input(
            "Similar questions per seed", min_value=1, max_value=5, value=1, key="aug_each",
        )
        aug_k = col_aug_k.slider(
            "Calibration k", min_value=1, max_value=8, value=4,
            help="Policy samples per candidate to verify RL suitability.", key="aug_k",
        )
        aug_mode = st.radio(
            "Generation mode",
            ["🎯 One-shot", "🤖 ReAct agent + Review"],
            index=1,
            horizontal=True, key="aug_mode",
            help=(
                "**One-shot**: a single LLM call per candidate.  "
                "**ReAct agent + Review**: the generator works as an agent (it tests its code "
                "with run_snippet and checks similarity), then a reviewer scores the draft on "
                "5 quality criteria (difficulty is not scored: it is measured on the policy); "
                "below 75/100 the feedback goes back into the same conversation "
                "and the agent revises. Then the policy is sampled on the draft: if it solves "
                "it every time or never, that result goes back to the agent to tune the "
                "difficulty. The generator knows the seed's gold answer, how the policy "
                "behaves on the seed, the earlier variants of the seed, and what SMEs "
                "rejected and approved on other samples."
            ),
        )
        aug_loops = None
        aug_fix_iters = None
        aug_tune = None
        if aug_mode == "🤖 ReAct agent + Review":
            col_loops, col_fix, col_tune = st.columns(3)
            aug_loops = col_loops.slider(
                "Max agent loops per draft", 1, 6, 3, key="aug_loops",
                help="Tool-call rounds the agent may use before it must submit a draft.",
            )
            aug_fix_iters = col_fix.slider(
                "Max fix iterations", 1, 4, 3, key="aug_fix_iters",
                help="How many times the agent can revise after reviewer feedback.",
            )
            aug_tune = col_tune.slider(
                "Max difficulty-tuning rounds", 0, 3, 2, key="aug_tune_rounds",
                help="After the reviewer passes a draft, the policy is sampled on it. If it solves "
                     "it every time (or never), the result goes back to the agent, which writes a "
                     "harder (or corrected/easier) version. 0 = no tuning, a candidate outside "
                     "the learnable band is simply rejected.",
            )

        _total_cands = int(aug_limit) * int(aug_each)
        if aug_mode == "🎯 One-shot":
            _est = _total_cands * (1 + int(aug_k))
        else:  # ReAct agent + Review, worst case: (tune + 1) rounds, each up to (fix + 1) drafts
            # of loops + 1 review calls, plus k policy samples to calibrate the round's draft
            _est = _total_cands * (int(aug_tune if aug_tune is not None else 2) + 1) * (
                (int(aug_fix_iters or 3) + 1) * (int(aug_loops or 3) + 1) + int(aug_k)
            )
        _worst = "" if aug_mode == "🎯 One-shot" else "up to "
        st.caption(
            f"{_worst}~{_est} API calls ({_total_cands} candidate(s)) ≈ "
            f"**{round(_est*_AVG/60,1)} min**"
            + ("" if aug_mode == "🎯 One-shot" else " worst case; most candidates need far fewer")
        )

        if st.button(
            "🔁 Generate similar questions", key="augment_btn", disabled="augment" in _active_jobs,
            help="Runs in the background; one augmentation job at a time.",
        ):
            seeds = rl_qs[: int(aug_limit)]
            st.session_state["augment_job_pick"] = jobs.start(
                "augment",
                {
                    "run_id": _run_id,
                    "mode": _AUG_MODES[aug_mode],
                    "seeds": [dict(q) for q in seeds],
                    "n_each": int(aug_each),
                    "calibration_k": int(aug_k),
                    "max_loops": int(aug_loops or 3),
                    "max_fix_iterations": int(aug_fix_iters or 3),
                    "max_tune_rounds": int(aug_tune if aug_tune is not None else 2),
                },
                planned=len(seeds) * int(aug_each),
            )
            st.rerun()
    else:
        st.info("Switch Role to **Organizer** to generate similar questions.")

_job_section("augment", "candidates", _render_augment_results)

# ── SME Queue ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🟢 SME Queue — RL-suitable generated questions")
st.caption(
    "Candidates from rescue and augmentation that are in the learnable band. "
    "An **SME** approves or rejects each one. Approved items are promoted to the questions table."
)

with engine.connect() as conn:
    sme_cands = conn.execute(
        text(
            """
            SELECT id, topic, band, stem, code, lang, executed_key, policy_pass_rate, pass_at_k,
                   generator_mode, source_fail_ids, created_at
            FROM candidates
            WHERE status = 'pending'
              AND generator_mode IN ('similar_augment', 'light_hint')
            ORDER BY created_at DESC
            """
        )
    ).mappings().all()

with st.expander("🧠 Feedback the generator sees (built from SME reviews)"):
    with engine.connect() as conn:
        _fb_topics = conn.execute(
            text("SELECT DISTINCT topic FROM candidates WHERE topic IS NOT NULL ORDER BY topic")
        ).scalars().all()
    if _fb_topics:
        _fb_topic = st.selectbox("Topic", _fb_topics, key="fb_preview_topic")
        _fb_q, _fb_h = st.tabs(["Question generation", "Rescue hints"])
        _fb_q.code(sme_feedback_for(_fb_topic), language=None)
        _fb_h.code(sme_feedback_for(_fb_topic, hints=True), language=None)
        st.caption("Not used for held-out eval generation (reviewers see policy pass rates).")
    else:
        st.caption("No generated candidates yet.")

if not sme_cands:
    st.info("Queue is empty. Run rescue or augmentation above to populate it.")
else:
    st.caption(f"**{len(sme_cands)}** candidate(s) awaiting review.")
    _PHASE_LABELS = {
        "light_hint":      "🔴 Dead-zone rescue",
        "similar_augment": "🟡 RL augmentation",
    }
    for c in sme_cands:
        phase_label = _PHASE_LABELS.get(c["generator_mode"], f"🔁 {c['generator_mode']}")
        rate_str = f"{c['policy_pass_rate']:.2f}" if c["policy_pass_rate"] is not None else "?"
        pak_str  = f"{c['pass_at_k']:.2f}" if c["pass_at_k"] is not None else "?"
        with st.expander(
            f"{phase_label} | `{c['id']}` — {c['topic']}/{c['band']} "
            f"| pass_rate={rate_str} | pass@k={pak_str}",
            expanded=False,
        ):
            st.caption(f"Gold key: `{c['executed_key']}` | lang: `{c['lang']}`")
            if c["source_fail_ids"]:
                import json as _json2
                src = _json2.loads(c["source_fail_ids"] or "[]")
                st.caption(f"Source question(s): {src}")

            if role == "SME":
                # Editable stem and code — pre-filled with current values
                edited_stem = st.text_area(
                    "Stem (edit if needed)",
                    value=c["stem"] or "",
                    height=120,
                    key=f"sme-stem-{c['id']}",
                )
                with st.expander("Edit verifier code (optional)", expanded=False):
                    edited_code = st.text_area(
                        "Verifier code",
                        value=c["code"] or "",
                        height=200,
                        key=f"sme-code-{c['id']}",
                    )
                    if edited_code != (c["code"] or ""):
                        st.warning(
                            "⚠️ Code changed — the sandbox will re-run on your version "
                            "and the gold key will be recomputed."
                        )

                reasons = st.multiselect(
                    "What's wrong? (to reject, pick a reason or write a note)",
                    list(REASONS), format_func=REASONS.get, key=f"sme-reasons-{c['id']}",
                )
                sme_note = st.text_area(
                    "Feedback for the generator (optional)",
                    placeholder="What should it do differently, or what makes this one good?",
                    height=68, key=f"sme-note-{c['id']}",
                )
                st.caption("Reasons, notes and your edits are shown to the generator next time.")
                btn_approve, btn_reject = st.columns(2)
                with btn_approve:
                    if st.button("✅ Approve", key=f"sme-approve-{c['id']}", width="stretch"):
                        try:
                            promote(
                                c["id"], "hard_item",
                                fair=True, novel=True, hard=True,
                                note=sme_note, kind="sme_queue",
                                edited_stem=edited_stem,
                                edited_code=edited_code,
                                reasons=reasons,
                            )
                            st.success(f"Candidate `{c['id']}` approved and promoted ✅")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with btn_reject:
                    if st.button("❌ Reject", key=f"sme-reject-{c['id']}", width="stretch"):
                        if not reasons and not sme_note.strip():
                            st.error("Pick a reason or write a note so the generator can learn from it.")
                        else:
                            try:
                                # Edits are not applied: the generated stem is what was wrong.
                                promote(
                                    c["id"], "rejected",
                                    fair=False, novel=True, hard=False,
                                    note=sme_note, kind="sme_queue", reasons=reasons,
                                )
                                st.warning(f"Candidate `{c['id']}` rejected ❌")
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))
            else:
                st.text(c["stem"][:1000] if c["stem"] else "(empty)")
                st.info("Switch Role to **SME** to approve or reject.")
