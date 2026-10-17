import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
for _name in list(sys.modules):
    if _name == "frontier_lens" or _name.startswith("frontier_lens."):
        del sys.modules[_name]

import json

import streamlit as st

# ── Shared theme CSS + top nav ───────────────────────────────────────────
import streamlit as _st_nav
_st_nav.markdown("""
<style>.nav-bar{display:flex;align-items:center;justify-content:space-between;padding:0 0 16px 0;border-bottom:1px solid #2D2D55;margin-bottom:32px;}.nav-links{display:flex;gap:0;align-items:center;}a.nav-link{display:inline-block;padding:8px 20px;color:#EDE8DC!important;font-size:.88rem;font-weight:500;text-decoration:none!important;border-bottom:2px solid transparent;transition:all .15s;letter-spacing:.01em;outline:none;}a.nav-link:hover{color:#FF8F5C!important;border-bottom-color:#FF6B2B;}a.nav-link.active{color:#FF8F5C!important;font-weight:700;border-bottom-color:#FF6B2B;}</style><div class="nav-bar"><a href="/" target="_self" style="text-decoration:none"><svg width="150" height="22" viewBox="0 0 150 22" style="overflow:visible;"><defs><linearGradient id="ng2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#FF6B2B"/><stop offset="100%" stop-color="#C96FFF"/></linearGradient></defs><text x="0" y="17" font-family="Inter,sans-serif" font-size="15" font-weight="800" letter-spacing="-0.3" fill="url(#ng2)">Frontier Lens</text></svg></a><div class="nav-links"><a href="/Seed" target="_self" class="nav-link">Seed</a><a href="/Evaluate" target="_self" class="nav-link">Evaluate</a><a href="/Weaknesses" target="_self" class="nav-link">Weaknesses</a><a href="/Generate_Review" target="_self" class="nav-link">Generate</a><a href="/Dataset" target="_self" class="nav-link active">Dataset</a><a href="/Headroom" target="_self" class="nav-link">Headroom</a></div></div>
""", unsafe_allow_html=True)

# ── Role selector ────────────────────────────────────────────────────────
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
role = st.session_state.get("role", "Viewer")

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.failure_analysis import rl_suitable_questions, zero_pass_questions
from frontier_lens.registry import policy_spec
from frontier_lens.review import promote

st.title("Dataset")

engine = get_engine()

with engine.connect() as conn:
    runs   = conn.execute(text(
        "SELECT id, created_at FROM eval_runs WHERE status='complete' ORDER BY created_at DESC"
    )).all()
    models = conn.execute(text("SELECT DISTINCT model_id FROM attempts")).scalars().all()

if not runs:
    st.warning("No completed eval runs yet.")
    st.stop()

_run_labels = [f"{r[0]}  —  {r[1][:16].replace('T',' ')}" for r in runs]
run_id = st.selectbox(
    "Eval run",
    [r[0] for r in runs],
    format_func=lambda rid: next(
        (lbl for lbl, r in zip(_run_labels, runs) if r[0] == rid), rid
    ),
)
_pol_id = policy_spec().id
_models = list(models)
model_id = st.selectbox("Model", _models,
                        index=_models.index(_pol_id) if _pol_id in _models else 0)

# ── Summary metrics (computed after run_id + model_id are known) ──────────
n_rl_seeds = len(rl_suitable_questions(run_id, model_id))

with engine.connect() as conn:
    n_approved = conn.execute(
        text("SELECT COUNT(*) FROM candidates WHERE status='approved'")
    ).scalar() or 0

    dead_zone_rows = conn.execute(
        text("""
            SELECT COUNT(DISTINCT a.question_id)
            FROM attempts a
            JOIN questions q ON q.id = a.question_id
            WHERE a.run_id = :r AND a.model_id = :m
              AND q.code IS NOT NULL AND TRIM(q.code) != ''
              AND COALESCE(q.sme_flagged, 0) = 0
            GROUP BY a.question_id
            HAVING SUM(a.is_correct) = 0
              AND (COUNT(*) - SUM(a.unparsable)) > 0
        """),
        {"r": run_id, "m": model_id},
    ).fetchall()
    n_sft = len(dead_zone_rows)

    rescued_ids = conn.execute(
        text("SELECT DISTINCT source_fail_ids FROM candidates WHERE generator_mode='light_hint' AND status IN ('pending','approved')")
    ).scalars().all()
    n_rescued = sum(1 for r in rescued_ids if r)
    n_sft_unrescued = max(0, n_sft - n_rescued)

    n_sme_queue = conn.execute(
        text("SELECT COUNT(*) FROM candidates WHERE status='pending' AND generator_mode IN ('similar_augment','light_hint')")
    ).scalar() or 0

c1, c2, c3 = st.columns(3)
c1.metric("✅ RL-ready", f"{n_rl_seeds + n_approved}",
          help="RL-suitable seeds + approved generated questions")
c2.metric("📚 SFT pool", f"{n_sft_unrescued}",
          help="Dead-zone questions that resisted all hint levels")
c3.metric("🔍 Pending SME", f"{n_sme_queue}",
          help="Generated candidates awaiting quality review")

st.divider()

tab_rl, tab_sft, tab_sme = st.tabs(["✅ RL-ready", "📚 SFT", "🔍 SME Review"])

# ═══════════════════════════════════════════════════════════════════════════
with tab_rl:
# ═══════════════════════════════════════════════════════════════════════════
    st.subheader("RL-ready questions")
    st.caption(
        "Questions in the learnable band (0 < pass_rate < 1) — suitable for RL training "
        "without further review. Includes RL-suitable seeds from this run **and** "
        "SME-approved generated questions."
    )

    # ── RL-suitable seeds ─────────────────────────────────────────────────
    rl_qs = rl_suitable_questions(run_id, model_id)
    st.markdown(f"**{len(rl_qs)} RL-suitable seed / uploaded questions** (this run)")
    if rl_qs:
        st.dataframe(
            [{
                "id": q["id"][:14],
                "topic": q["topic"], "band": q["band"],
                "correct": int(q["n_correct"]), "k": int(q["n"]),
                "content_pass_rate": q.get("content_pass_rate",
                    round(int(q["n_correct"]) / max(int(q["n"]) - int(q.get("n_unparsable", 0)), 1), 3)),
                "stem": (q["stem"] or "")[:100],
            } for q in rl_qs],
            use_container_width=True,
        )
        # CSV download
        import csv as _csv, io as _io
        out = _io.StringIO()
        writer = _csv.DictWriter(out, fieldnames=["id","topic","band","stem","key","n_correct","n","content_pass_rate"])
        writer.writeheader()
        for q in rl_qs:
            np_ = max(int(q["n"]) - int(q.get("n_unparsable", 0)), 1)
            writer.writerow({
                "id": q["id"], "topic": q["topic"], "band": q["band"],
                "stem": q["stem"], "key": q["key"],
                "n_correct": int(q["n_correct"]), "n": int(q["n"]),
                "content_pass_rate": round(int(q["n_correct"]) / np_, 3),
            })
        st.download_button(
            "⬇ Download RL seeds CSV",
            data=out.getvalue(),
            file_name=f"rl_seeds_{run_id}.csv", mime="text/csv",
            key="dl-rl-seeds",
        )

    # ── Approved generated questions ─────────────────────────────────────
    st.divider()
    with engine.connect() as conn:
        approved = conn.execute(
            text("""
                SELECT id, topic, band, stem, executed_key, policy_pass_rate,
                       generator_mode, created_at
                FROM candidates WHERE status='approved'
                ORDER BY created_at DESC
            """)
        ).mappings().all()

    st.markdown(f"**{len(approved)} SME-approved generated questions**")
    if approved:
        st.dataframe(
            [{
                "id": c["id"][:14], "topic": c["topic"], "band": c["band"],
                "mode": c["generator_mode"],
                "pass_rate": round(c["policy_pass_rate"] or 0, 3),
                "stem": (c["stem"] or "")[:100],
            } for c in approved],
            use_container_width=True,
        )
        out2 = _io.StringIO()
        w2 = _csv.DictWriter(out2, fieldnames=["id","topic","band","stem","key","pass_rate","generator_mode"])
        w2.writeheader()
        for c in approved:
            w2.writerow({
                "id": c["id"], "topic": c["topic"], "band": c["band"],
                "stem": c["stem"], "key": c["executed_key"],
                "pass_rate": c["policy_pass_rate"], "generator_mode": c["generator_mode"],
            })
        st.download_button(
            "⬇ Download approved generated CSV",
            data=out2.getvalue(),
            file_name="rl_approved_generated.csv", mime="text/csv",
            key="dl-rl-approved",
        )
    else:
        st.info("No approved generated questions yet. SME-approve candidates in the Generate tab.")

# ═══════════════════════════════════════════════════════════════════════════
with tab_sft:
# ═══════════════════════════════════════════════════════════════════════════
    st.subheader("SFT pool — unrescuable dead-zone questions")
    st.caption(
        "Questions where the model **never** answered correctly (pass@k = 0) AND "
        "all hint levels failed to make it RL-suitable. "
        "These are good candidates for **supervised fine-tuning (SFT)**: "
        "the model clearly needs direct exposure to similar solved examples."
    )

    # Dead-zone questions with verifier code
    dead_qs = zero_pass_questions(run_id, model_id)

    if not dead_qs:
        st.info("No dead-zone questions for this run/model.")
    else:
        # Find which ones have been rescued (light_hint pending/approved candidate)
        with engine.connect() as conn:
            rescued_raw = conn.execute(
                text("SELECT source_fail_ids FROM candidates WHERE generator_mode='light_hint' AND status IN ('pending','approved')")
            ).scalars().all()
        rescued_set: set[str] = set()
        for raw in rescued_raw:
            try:
                ids = json.loads(raw or "[]")
                rescued_set.update(ids)
            except Exception:
                pass

        # Also find auto_rejected light_hint candidates for these questions
        dz_ids = [q["id"] for q in dead_qs]
        with engine.connect() as conn:
            hint_cands = conn.execute(
                text("""
                    SELECT source_fail_ids, stem, executed_key, auto_reject_reason
                    FROM candidates
                    WHERE generator_mode='light_hint' AND status='auto_rejected'
                """)
            ).mappings().all()

        # Build map: source_qid → best hinted stem attempted
        hint_map: dict[str, str] = {}
        for hc in hint_cands:
            try:
                src_ids = json.loads(hc["source_fail_ids"] or "[]")
                for sid in src_ids:
                    if sid not in hint_map:
                        hint_map[sid] = hc["stem"] or ""
            except Exception:
                pass

        unrescued = [q for q in dead_qs if q["id"] not in rescued_set]
        st.caption(
            f"**{len(dead_qs)}** dead-zone questions total → "
            f"**{len(rescued_set) & set(dz_ids).__len__()}** rescued → "
            f"**{len(unrescued)}** unrescuable (SFT pool)"
        )

        if unrescued:
            st.dataframe(
                [{
                    "id": q["id"][:14], "topic": q["topic"], "band": q["band"],
                    "k": int(q["n"]), "gold_key": q["key"],
                    "best_hint_tried": hint_map.get(q["id"], "(no hint attempted)")[:100],
                    "stem": (q["stem"] or "")[:100],
                } for q in unrescued],
                use_container_width=True,
            )

            # Build SFT CSV: original responses from attempts table + best hint stem
            with engine.connect() as conn:
                orig_attempts = conn.execute(
                    text("""
                        SELECT question_id, sample_index, raw_output, parsed_answer, is_correct
                        FROM attempts WHERE run_id=:r AND model_id=:m
                        ORDER BY question_id, sample_index
                    """),
                    {"r": run_id, "m": model_id},
                ).mappings().all()

            from collections import defaultdict
            by_q: dict[str, list] = defaultdict(list)
            for a in orig_attempts:
                by_q[a["question_id"]].append(dict(a))

            max_k = max((len(v) for v in by_q.values()), default=1)
            sft_cols = ["question_id", "topic", "band", "stem_original", "gold_key",
                        "best_hint_stem"] + \
                       [f"response_{i+1}" for i in range(max_k)] + \
                       [f"correct_{i+1}" for i in range(max_k)]

            sft_out = _io.StringIO()
            sft_writer = _csv.DictWriter(sft_out, fieldnames=sft_cols, extrasaction="ignore")
            sft_writer.writeheader()
            for q in unrescued:
                row: dict = {
                    "question_id": q["id"], "topic": q["topic"], "band": q["band"],
                    "stem_original": q["stem"], "gold_key": q["key"],
                    "best_hint_stem": hint_map.get(q["id"], ""),
                }
                for i, att in enumerate(by_q.get(q["id"], [])[:max_k]):
                    row[f"response_{i+1}"] = att.get("raw_output", "")
                    row[f"correct_{i+1}"] = att.get("is_correct", False)
                sft_writer.writerow(row)

            st.download_button(
                f"⬇ Download SFT CSV ({len(unrescued)} questions)",
                data=sft_out.getvalue(),
                file_name=f"sft_pool_{run_id}.csv", mime="text/csv",
                key="dl-sft",
            )
        else:
            st.success("All dead-zone questions have been rescued! Nothing left in the SFT pool.")

# ═══════════════════════════════════════════════════════════════════════════
with tab_sme:
# ═══════════════════════════════════════════════════════════════════════════
    st.subheader("SME Review queue")
    st.caption(
        "Generated questions awaiting quality review. An **SME** approves or rejects each one. "
        "Approved items move to the RL-ready pool."
    )

    with engine.connect() as conn:
        sme_cands = conn.execute(
            text("""
                SELECT id, topic, band, stem, executed_key, policy_pass_rate, pass_at_k,
                       generator_mode, source_fail_ids, created_at
                FROM candidates
                WHERE status = 'pending'
                  AND generator_mode IN ('similar_augment', 'light_hint')
                ORDER BY created_at DESC
            """)
        ).mappings().all()

    if not sme_cands:
        st.info("Queue is empty. Run rescue or augmentation in Generate to populate it.")
    else:
        st.caption(f"**{len(sme_cands)}** candidate(s) awaiting review.")
        _PHASE = {
            "light_hint":      "🔴 Dead-zone rescue",
            "similar_augment": "🟡 RL augmentation",
        }
        for c in sme_cands:
            phase = _PHASE.get(c["generator_mode"], f"🔁 {c['generator_mode']}")
            rate_str = f"{c['policy_pass_rate']:.2f}" if c["policy_pass_rate"] is not None else "?"
            with st.expander(
                f"{phase} | `{c['id']}` — {c['topic']}/{c['band']} | pass_rate={rate_str}",
                expanded=False,
            ):
                st.text((c["stem"] or "")[:800])
                st.caption(f"Gold key: `{c['executed_key']}`")
                if c["source_fail_ids"]:
                    src = json.loads(c["source_fail_ids"] or "[]")
                    st.caption(f"Source question(s): {src}")

                if role == "SME":
                    sme_label = st.selectbox(
                        "Label", ["hard_item", "trivial", "broken_item", "rejected"],
                        key=f"ds-lab-{c['id']}",
                    )
                    sme_note = st.text_input("Note (optional)", key=f"ds-note-{c['id']}")
                    if st.button("✔ Submit review", key=f"ds-go-{c['id']}"):
                        try:
                            promote(
                                c["id"], sme_label,
                                fair=sme_label == "hard_item", novel=True,
                                hard=sme_label == "hard_item", note=sme_note,
                                kind="sme_queue",
                            )
                            st.success(
                                f"Candidate `{c['id']}` "
                                + ("→ RL-ready ✅" if sme_label == "hard_item" else "rejected ❌")
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                else:
                    st.info("Switch Role to **SME** to approve or reject.")
