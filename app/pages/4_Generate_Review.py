import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
for _name in list(sys.modules):
    if _name == "frontier_lens" or _name.startswith("frontier_lens."):
        del sys.modules[_name]

import streamlit as st

# ── Shared theme CSS + top nav (injected) ────────────────────────────────
import streamlit as _st_nav
_st_nav.markdown("""
<style>.nav-bar{display:flex;align-items:center;justify-content:space-between;padding:0 0 16px 0;border-bottom:1px solid #2D2D55;margin-bottom:32px;}.nav-links{display:flex;gap:0;align-items:center;}a.nav-link{display:inline-block;padding:8px 20px;color:#EDE8DC!important;font-size:.88rem;font-weight:500;text-decoration:none!important;border-bottom:2px solid transparent;transition:all .15s;letter-spacing:.01em;outline:none;}a.nav-link:hover{color:#FF8F5C!important;border-bottom-color:#FF6B2B;}a.nav-link.active{color:#FF8F5C!important;font-weight:700;border-bottom-color:#FF6B2B;}</style><div class="nav-bar"><a href="/" target="_self" style="text-decoration:none"><svg width="150" height="22" viewBox="0 0 150 22" style="overflow:visible;"><defs><linearGradient id="ng2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#FF6B2B"/><stop offset="100%" stop-color="#C96FFF"/></linearGradient></defs><text x="0" y="17" font-family="Inter,sans-serif" font-size="15" font-weight="800" letter-spacing="-0.3" fill="url(#ng2)">Frontier Lens</text></svg></a><div class="nav-links"><a href="/Seed" target="_self" class="nav-link">Seed</a><a href="/Evaluate" target="_self" class="nav-link">Evaluate</a><a href="/Weaknesses" target="_self" class="nav-link">Weaknesses</a><a href="/Generate_Review" target="_self" class="nav-link active">Generate</a><a href="/Dataset" target="_self" class="nav-link">Dataset</a></div></div>
""", unsafe_allow_html=True)
# ── Persistent role selector (shown on every page) ───────────────────────
_current_role = st.session_state.get("role", "Viewer")
_role_cols = st.columns([6, 1])
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
from frontier_lens.registry import policy_spec, sync_models
from frontier_lens.review import (
    flag_suspicious,
    promote,
)

st.title("Generate")
engine = get_engine()

import uuid as _uuid
from frontier_lens.failure_analysis import rl_suitable_questions, zero_pass_questions
from frontier_lens.runlog import write_op_event

# ── Run / model selectors ─────────────────────────────────────────────────
with engine.connect() as conn:
    _all_runs   = conn.execute(text("SELECT id, created_at FROM eval_runs WHERE status='complete' ORDER BY created_at DESC")).all()
    _all_models = conn.execute(text("SELECT DISTINCT model_id FROM attempts")).scalars().all()

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
        use_container_width=True,
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

        if st.button("💡 Progressive hint rescue", key="rescue_btn"):
            from frontier_lens.generate import generate_rescue_sft_csv, rescue_with_hints

            ids_to_rescue = [q["id"] for q in dead_qs[: int(rescue_limit)]]
            op_id = f"rescue_{_uuid.uuid4().hex[:8]}"
            k_val, h_val = int(rescue_k), int(rescue_max_hints)
            n_q = len(ids_to_rescue)
            _total_ops = n_q * h_val * (1 + k_val)
            _done_ops = [0]

            st.caption(f"Op log: `logs/op_{op_id}.txt`")
            prog = st.progress(0.0, text=f"0 / {_total_ops} steps")
            log_lines: list[str] = []
            log_box = st.empty()

            def _fmt_rescue(event, **f):
                if event == "rescue_start":
                    return f"[START] {f.get('n_questions')}q | k={f.get('calibration_k')} | max_hints={f.get('max_hints')}"
                if event == "rescue_question":
                    return f"\n[Q {f.get('index')}/{f.get('total')}] {str(f.get('question_id',''))[:12]}  {f.get('topic')}/{f.get('band')}  key={str(f.get('gold_key',''))[:20]}"
                if event == "hint_attempt_start":
                    return f"\n  ── Hint round {f.get('level','?')}/{f.get('max_hints','?')} ──"
                if event == "hint_generated":
                    if f.get("parse_ok"):
                        return f"  💡 L{f.get('level','?')} (✓): {str(f.get('hint',''))[:100]}"
                    return f"  ⚠ L{f.get('level','?')} parse FAILED: {f.get('parse_error','')} | raw: {str(f.get('raw_response_preview',''))[:150]}"
                if event == "calibration_sample":
                    mark = "✓" if f.get("is_correct") else "✗"
                    return f"    sample {f.get('sample_index',0)+1}: {mark}"
                if event == "hint_attempt_result":
                    rl = f.get("rl_suitable")
                    badge = "✅ rescued!" if rl else f"❌ still failing ({f.get('n_correct')}/{f.get('n')})"
                    return f"  L{f.get('level','?')}: {badge}  pass_rate={f.get('pass_rate',0):.2f}"
                if event == "hint_result":
                    if f.get("rl_suitable"):
                        return f"  ✅ RESCUED L{f.get('rescue_level')} | cand={f.get('candidate_id')} pass_rate={f.get('pass_rate',0):.2f}"
                    return f"  ❌ UNRESCUABLE after {f.get('n_hints_tried')} hint(s) → SFT export"
                if event == "rescue_end":
                    return f"\n[DONE] rescued={f.get('n_rescued')}/{f.get('n_total')}  rejected={f.get('n_rejected')}"
                return f"[{event}]"

            def rescue_log_fn(event, **fields):
                write_op_event(op_id, event, **fields)
                if event in ("hint_generated", "calibration_sample"):
                    _done_ops[0] += 1
                    pct = min(_done_ops[0] / max(_total_ops, 1), 1.0)
                    lvl = fields.get("level", "?")
                    lbl = (f"L{lvl} hint…" if event == "hint_generated"
                           else f"L{lvl} sample {fields.get('sample_index',0)+1}/{k_val}")
                    prog.progress(pct, text=f"Step {_done_ops[0]}/{_total_ops} — {lbl}")
                log_lines.append(_fmt_rescue(event, **fields))
                log_box.code("\n".join(log_lines[-30:]), language=None)

            rescue_results = rescue_with_hints(
                _run_id, _model_id, ids_to_rescue,
                log_fn=rescue_log_fn, calibration_k=k_val, max_hints=h_val,
            )
            prog.progress(1.0, text="✅ Complete")
            rescued = sum(1 for r in rescue_results if r["rl_suitable"])
            unrescued = len(rescue_results) - rescued
            st.success(f"**{rescued}/{len(rescue_results)}** rescued.  Log → `logs/op_{op_id}.txt`")
            st.dataframe(
                [{"question_id": r["question_id"][:14], "rescued": r["rl_suitable"],
                  "rescue_level": r.get("rescue_level") or "—",
                  "pass_rate": round(r["pass_rate"], 3),
                  "candidate_id": r.get("candidate_id") or "—",
                  "hint_L1": (r["hint_rounds"][0]["hint"] if len(r.get("hint_rounds",[])) > 0 else "")[:60],
                  "hint_L2": (r["hint_rounds"][1]["hint"] if len(r.get("hint_rounds",[])) > 1 else "")[:60],
                  "hint_L3": (r["hint_rounds"][2]["hint"] if len(r.get("hint_rounds",[])) > 2 else "")[:60]}
                 for r in rescue_results],
                use_container_width=True,
            )
            if unrescued > 0:
                from frontier_lens.generate import generate_rescue_sft_csv
                sft_csv = generate_rescue_sft_csv(rescue_results)
                st.warning(f"**{unrescued}** question(s) unrescuable after {h_val} levels → SFT CSV below.")
                st.download_button(
                    f"⬇ SFT CSV ({unrescued} unrescued)", data=sft_csv,
                    file_name=f"sft_unrescued_{op_id}.csv", mime="text/csv",
                    key=f"sft-dl-{op_id}",
                )
    else:
        st.info("Switch Role to **Organizer** to run the rescue.")

# ── RL-suitable augmentation ──────────────────────────────────────────────
st.divider()
st.subheader("🟡 RL-suitable augmentation  (0 < correct < k)")
st.caption(
    "Questions already in the learnable band. The generator produces similar-but-distinct "
    "questions. RL-suitable ones land in the SME Queue below."
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
        use_container_width=True,
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
            ["🎯 One-shot", "🤖 ReAct agent", "🔄 Generate + Review (mk0-style)"],
            index=2,
            horizontal=True, key="aug_mode",
            help=(
                "**One-shot**: single LLM call.  "
                "**ReAct agent**: iterative with run_snippet + check_similarity.  "
                "**Generate + Review**: generator → reviewer → fix loop (up to 3 fix cycles). "
                "Reviewer scores on 5 criteria; must reach 75/100 to pass. "
                "Generator knows the seed's gold answer and prior variants."
            ),
        )
        aug_loops = None
        aug_fix_iters = None
        if aug_mode == "🤖 ReAct agent":
            aug_loops = st.slider("Max agent loops", 1, 6, 3, key="aug_loops")
        elif aug_mode == "🔄 Generate + Review (mk0-style)":
            aug_fix_iters = st.slider(
                "Max fix iterations", 1, 4, 3, key="aug_fix_iters",
                help="How many times the generator can revise after reviewer feedback."
            )

        _total_cands = int(aug_limit) * int(aug_each)
        if aug_mode == "🎯 One-shot":
            _est = _total_cands * (1 + int(aug_k))
        elif aug_mode == "🤖 ReAct agent":
            _est = _total_cands * (int(aug_loops or 3) + int(aug_k))
        else:  # Generate + Review
            # (1 gen + 1 review) × fix_iters + k calibration
            _est = _total_cands * ((int(aug_fix_iters or 3) + 1) * 2 + int(aug_k))
        st.caption(
            f"~{_est} API calls ({_total_cands} candidate(s)) ≈ **{round(_est*_AVG/60,1)} min**"
        )

        if st.button("🔁 Generate similar questions", key="augment_btn"):
            use_review = (aug_mode == "🔄 Generate + Review (mk0-style)")
            use_agent  = (aug_mode == "🤖 ReAct agent")
            if use_review:
                from frontier_lens.generate import generate_similar_with_review
            elif use_agent:
                from frontier_lens.generate import generate_similar_react
            else:
                from frontier_lens.generate import generate_similar_questions

            seeds = rl_qs[: int(aug_limit)]
            op_id = f"augment_{_uuid.uuid4().hex[:8]}"
            k_val = int(aug_k)
            _total_ops = _total_cands * (1 + k_val)
            _done_ops = [0]

            st.caption(f"Op log: `logs/op_{op_id}.txt`")
            prog = st.progress(0.0, text=f"0 / {_total_ops} steps")
            log_lines2: list[str] = []
            log_box2 = st.empty()

            def _fmt_aug(event, **f):
                if event == "similar_start":
                    return (f"[START] seeds={f.get('n_seeds')} n_each={f.get('n_each')} "
                            f"total={f.get('total_candidates')} k={f.get('calibration_k')} "
                            f"mode={f.get('mode','')} max_fix={f.get('max_fix_iterations','')}")
                if event == "similar_generate":
                    return f"\n[{f.get('index')}/{f.get('total')}] source={str(f.get('source_id',''))[:12]}  {f.get('topic')}/{f.get('band')}"
                if event == "sandbox_result" or event == "review_sandbox":
                    return f"  🔧 sandbox: {'OK ✓' if f.get('ok') else 'FAIL ✗ (' + str(f.get('reason','')) + ')'}"
                if event == "calibration_sample":
                    return f"  sample {f.get('sample_index',0)+1}: {'✓' if f.get('is_correct') else '✗'}"
                if event == "similar_result":
                    rate = f.get("pass_rate"); rate_str = f"{rate:.2f}" if rate is not None else "n/a"
                    icon = "✅" if f.get("status") == "pending" else "❌"
                    rev = f" [review={f.get('review_score')}]" if f.get("review_score") else ""
                    return f"  {icon} cand={f.get('candidate_id','?')} status={f.get('status')} pass_rate={rate_str}{rev}"
                if event == "similar_end":
                    return f"\n[DONE] pending={f.get('n_pending')}/{f.get('n_total')}"
                # ── Generate+Review events ────────────────────────────────
                if event == "review_generate":
                    return f"  📝 generating (iter {f.get('fix_iter',0)})…"
                if event == "review_start":
                    return f"  🔍 reviewing (iter {f.get('fix_iter',0)})…"
                if event == "review_result":
                    score = f.get("score", 0)
                    passed = f.get("passed", False)
                    badge = "✅ PASSED" if passed else f"❌ FAILED"
                    issues = "; ".join(f.get("issues", []))
                    cs = f.get("criterion_scores", {})
                    scores_str = " | ".join(f"{k.replace('_',' ')[:8]}={v}" for k,v in cs.items())
                    return (f"  {badge} score={score}/100  {scores_str}\n"
                            + (f"    issues: {issues}" if issues else ""))
                if event == "review_fix":
                    return (f"\n  🔧 Fix iter {f.get('fix_iter')}: score was {f.get('score')} — "
                            f"issues: {'; '.join(str(i) for i in f.get('issues', []))[:120]}")
                # ── ReAct agent events ────────────────────────────────────
                if event == "agent_tool_call":
                    return f"  ↳ step {f.get('step')} tool={f.get('tool')}"
                if event == "agent_end":
                    cid = f.get("candidate_id")
                    return f"  🏁 done  steps={f.get('steps_used')}  {'cand='+cid if cid else 'no submit'}"
                return f"[{event}]"

            def augment_log_fn(event, **fields):
                write_op_event(op_id, event, **fields)
                if event in ("sandbox_result", "calibration_sample",
                             "handle_calibration_sample", "agent_tool_call"):
                    _done_ops[0] += 1
                    pct = min(_done_ops[0] / max(_total_ops, 1), 1.0)
                    prog.progress(pct, text=f"Step {_done_ops[0]}/{_total_ops}")
                log_lines2.append(_fmt_aug(event, **fields))
                log_box2.code("\n".join(log_lines2[-30:]), language=None)

            if use_review:
                aug_results = generate_similar_with_review(
                    seeds, n_each=int(aug_each),
                    max_fix_iterations=int(aug_fix_iters or 3),
                    log_fn=augment_log_fn, calibration_k=k_val,
                )
            elif use_agent:
                aug_results = generate_similar_react(
                    seeds, n_each=int(aug_each), max_loops=int(aug_loops or 3),
                    log_fn=augment_log_fn, calibration_k=k_val,
                )
            else:
                aug_results = generate_similar_questions(
                    seeds, n_each=int(aug_each),
                    log_fn=augment_log_fn, calibration_k=k_val,
                )

            prog.progress(1.0, text="✅ Complete")
            rl_ok = sum(1 for r in aug_results if r["status"] == "pending")
            st.success(f"**{rl_ok}/{len(aug_results)}** RL-suitable.  Log → `logs/op_{op_id}.txt`")
            st.dataframe(
                [{"source_id": r["source_id"][:14], "candidate_id": r["candidate_id"],
                  "status": r["status"],
                  "pass_rate": round(r["pass_rate"], 3) if r["pass_rate"] is not None else None}
                 for r in aug_results],
                use_container_width=True,
            )
    else:
        st.info("Switch Role to **Organizer** to generate similar questions.")

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
            SELECT id, topic, band, stem, executed_key, policy_pass_rate, pass_at_k,
                   generator_mode, source_fail_ids, created_at
            FROM candidates
            WHERE status = 'pending'
              AND generator_mode IN ('similar_augment', 'light_hint')
            ORDER BY created_at DESC
            """
        )
    ).mappings().all()

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
            st.text(c["stem"][:1000] if c["stem"] else "(empty)")
            st.caption(f"Gold key: `{c['executed_key']}`")
            if c["source_fail_ids"]:
                import json as _json2
                src = _json2.loads(c["source_fail_ids"] or "[]")
                st.caption(f"Source question(s): {src}")

            if role == "SME":
                sme_label = st.selectbox(
                    "Label", ["hard_item", "trivial", "broken_item", "rejected"],
                    key=f"sme-lab-{c['id']}",
                )
                sme_note = st.text_input("Note (optional)", key=f"sme-note-{c['id']}")
                if st.button("✔ Submit review", key=f"sme-go-{c['id']}"):
                    try:
                        promote(
                            c["id"], sme_label,
                            fair=sme_label == "hard_item", novel=True,
                            hard=sme_label == "hard_item", note=sme_note,
                            kind="sme_queue",
                        )
                        st.success(
                            f"Candidate `{c['id']}` "
                            + ("promoted ✅" if sme_label == "hard_item" else "rejected ❌")
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            else:
                st.info("Switch Role to **SME** to approve or reject.")
