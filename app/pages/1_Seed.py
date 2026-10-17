import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
for _name in list(sys.modules):
    if _name == "frontier_lens" or _name.startswith("frontier_lens."):
        del sys.modules[_name]

import json
import os
import subprocess

import streamlit as st

# ── Shared theme CSS + top nav (injected) ────────────────────────────────
import streamlit as _st_nav
_st_nav.markdown("""
<style>.nav-bar{display:flex;align-items:center;justify-content:space-between;padding:0 0 16px 0;border-bottom:1px solid #2D2D55;margin-bottom:32px;}.nav-links{display:flex;gap:0;align-items:center;}a.nav-link{display:inline-block;padding:8px 20px;color:#EDE8DC!important;font-size:.88rem;font-weight:500;text-decoration:none!important;border-bottom:2px solid transparent;transition:all .15s;letter-spacing:.01em;outline:none;}a.nav-link:hover{color:#FF8F5C!important;border-bottom-color:#FF6B2B;}a.nav-link.active{color:#FF8F5C!important;font-weight:700;border-bottom-color:#FF6B2B;}</style><div class="nav-bar"><a href="/" target="_self" style="text-decoration:none"><svg width="150" height="22" viewBox="0 0 150 22" style="overflow:visible;"><defs><linearGradient id="ng2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#FF6B2B"/><stop offset="100%" stop-color="#C96FFF"/></linearGradient></defs><text x="0" y="17" font-family="Inter,sans-serif" font-size="15" font-weight="800" letter-spacing="-0.3" fill="url(#ng2)">Frontier Lens</text></svg></a><div class="nav-links"><a href="/Seed" target="_self" class="nav-link active">Seed</a><a href="/Evaluate" target="_self" class="nav-link">Evaluate</a><a href="/Weaknesses" target="_self" class="nav-link">Weaknesses</a><a href="/Generate_Review" target="_self" class="nav-link">Generate</a><a href="/Dataset" target="_self" class="nav-link">Dataset</a></div></div>
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

from frontier_lens.calibrate import GRIDS, KNOBS_PATH, PRIMARY, load_manifest
from frontier_lens.config import ROOT
from frontier_lens.db import get_engine

st.title("Seed")
st.caption("Step 1 of the pipeline — build the question bank the model will be evaluated on.")

engine = get_engine()
tab_auto, tab_upload = st.tabs(["🌱 Auto-generate", "📂 Upload CSV"])

# ═══════════════════════════════════════════════════════════════════════════
with tab_auto:
# ═══════════════════════════════════════════════════════════════════════════

    # ── Current seed bank status ──────────────────────────────────────────
    st.subheader("Current seed bank")

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT topic, level, split, COUNT(*) as n
                FROM questions
                WHERE source = 'seed'
                GROUP BY topic, level, split
                ORDER BY topic, level, split
                """
            )
        ).all()

    if rows:
        by_topic: dict[str, dict] = {}
        total = 0
        for topic, level, split, n in rows:
            by_topic.setdefault(topic, {"L1": 0, "L2 train": 0, "L2 eval": 0, "L3 train": 0, "L3 eval": 0})
            key = f"L{level}" if level == 1 else f"L{level} {split}"
            by_topic[topic][key] = n
            total += n

        table_rows = [{"topic": t, **counts} for t, counts in sorted(by_topic.items())]
        st.dataframe(table_rows, use_container_width=True)
        st.caption(f"**{total} seed questions** across {len(by_topic)} topics.")
    else:
        st.warning("No seed questions in the database. Run **Generate Seeds** below.")

    # ── Calibration status ────────────────────────────────────────────────
    st.subheader("Difficulty calibration")

    manifest = load_manifest()
    source = manifest.get("source", "probe")
    policy_id = manifest.get("policy_id")
    created_at = manifest.get("created_at", "—")

    if source == "policy":
        st.success(
            f"Calibrated to **{policy_id}** on {created_at[:10]}  |  k={manifest.get('k')}"
        )
    else:
        st.warning(
            "Using **synthetic probe curve** (not measured on your model). "
            "L2 ≈ 50% pass rate and L3 ≈ 15% are assumed, not measured. "
            "Run calibration below for accurate difficulty."
        )

    with st.expander("Knob details per topic"):
        rows_cal = []
        for topic in GRIDS:
            block = manifest.get(topic, {})
            rows_cal.append({
                "topic": topic,
                "primary_knob": PRIMARY.get(topic, "—"),
                "L2": str(block.get("2", "—")),
                "L3": str(block.get("3", "—")),
                "measured": block.get("measured", False),
            })
        st.dataframe(rows_cal, use_container_width=True)

    # ── Step 1: Calibrate ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Step 1 — Calibrate difficulty (optional but recommended)")
    st.markdown(
        "Measures the policy's actual pass rate at each grid point. "
        "Writes `data/level_knobs.json` so L2/L3 are tuned to *this* model. "
        "\n\n**Cost:** 7 topics × 6 grid points × k samples API calls (~28 min for k=8)."
    )

    col_topics, col_k_cal = st.columns(2)
    with col_topics:
        all_topics = list(GRIDS.keys())
        cal_topics = st.multiselect(
            "Topics to calibrate",
            all_topics,
            default=all_topics,
            help="Calibrate only new topics to save time.",
        )
    with col_k_cal:
        cal_k = st.number_input("k per grid point", 1, 8, 4,
                                 help="4 is a good balance of accuracy and cost.")

    n_cal_calls = len(cal_topics) * 6 * cal_k
    st.caption(f"≈ {n_cal_calls} API calls for the selected configuration.")

    if st.button("Run calibration", disabled=(role != "Organizer" or not cal_topics)):
        if role != "Organizer":
            st.error("Set Role to Organizer on Home to run calibration.")
        else:
            cmd = [
                sys.executable, "-m", "frontier_lens.calibrate",
                "--live", "--k", str(cal_k),
                "--topics", *cal_topics,
            ]
            st.write("Running:", " ".join(cmd))
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            log_box = st.empty()
            chunks: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                chunks.append(line if line.endswith("\n") else line + "\n")
                log_box.code("".join(chunks)[-2000:])
            code = proc.wait()
            if code == 0:
                st.success("Calibration complete. Knobs written to `data/level_knobs.json`. Now regenerate seeds.")
                st.rerun()
            else:
                st.error(f"Calibration failed (exit {code}).")

    if role != "Organizer":
        st.caption("Switch Role to **Organizer** on Home to run calibration.")

    # ── Step 2: Generate seeds ────────────────────────────────────────────
    st.divider()
    st.subheader("Step 2 — Generate seed questions")
    st.markdown(
        "Builds the full question bank (7 topics × 26 items = **182 seed questions**). "
        "Uses the calibrated knobs if available, otherwise the synthetic probe curve. "
        "\n\n`--force` deletes existing seed rows and rebuilds from scratch. "
        "Generated questions are left in place."
    )

    force = st.checkbox("Force rebuild (delete existing seeds)", value=bool(rows))

    if st.button("Generate seeds", type="primary", disabled=(role != "Organizer")):
        if role != "Organizer":
            st.error("Set Role to Organizer on Home to generate seeds.")
        else:
            cmd = [sys.executable, "-m", "frontier_lens.seed"]
            if force:
                cmd.append("--force")
            st.write("Running:", " ".join(cmd))

            prog = st.progress(0.0, text="Seeding…")
            log_box = st.empty()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            chunks: list[str] = []
            n_seeded = 0
            TARGET = 182
            assert proc.stdout is not None
            for line in proc.stdout:
                chunks.append(line if line.endswith("\n") else line + "\n")
                if "seeded" in line:
                    try:
                        n_seeded = int(line.strip().split()[1])
                        prog.progress(min(n_seeded / TARGET, 1.0), text=f"Seeded {n_seeded}/{TARGET} items")
                    except (IndexError, ValueError):
                        pass
                log_box.code("".join(chunks)[-2000:])
            code = proc.wait()
            prog.progress(1.0, text="Done")
            if code == 0:
                st.success(f"✅ Seeded {n_seeded} questions. Go to **Evaluate** (page 2) to run the model.")
                st.rerun()
            else:
                st.error(f"Seeding failed (exit {code}).")

    if role != "Organizer":
        st.caption("Switch Role to **Organizer** on Home to generate seeds.")

    # ── Topic previews ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Topic preview — sample seed question per family")

    EXAMPLE_STEMS = {
        "python_state": "Shows a Python program with aliased lists/dicts/sets. Asks: *what number does this print?* (rolling checksum, 0–999)",
        "shell_pipelines": "Shows a CSV file + a multi-stage shell pipeline (sort | filter | awk). Asks: *what number does this print?*",
        "constraint_counting": "Describes counting constraints on strings (adjacency, parity, order). Asks: *how many such strings are there?*",
        "multi_step_units": "Describes a chain of financial operations (shipping fee, tax, rounding). Asks: *what is the final dollar amount?*",
        "fsm_queries": "Describes a finite state machine + input string + accumulator procedure. Asks: *what four-digit number does the simulation produce?*",
        "minimax": "Describes a game tree with leaf values (MAX/MIN players alternate). Asks: *minimax value + sum of all leaves?*",
        "logic_grid": "Describes N named items with position clues. Asks: *what four-digit checksum does the arrangement produce?*",
    }

    for topic, description in EXAMPLE_STEMS.items():
        with st.expander(f"`{topic}`"):
            st.markdown(description)
            with engine.connect() as conn:
                ex = conn.execute(
                    text("SELECT stem FROM questions WHERE topic=:t AND source='seed' AND level=2 LIMIT 1"),
                    {"t": topic},
                ).scalar()
            if ex:
                st.text(ex[:600] + ("…" if len(ex) > 600 else ""))
            else:
                st.caption("(no seed questions yet — generate seeds first)")


# ═══════════════════════════════════════════════════════════════════════════
with tab_upload:
# ═══════════════════════════════════════════════════════════════════════════

    st.subheader("Upload your own questions (CSV)")
    st.caption(
        "Uploaded questions survive `seed --force`, appear in Evaluate alongside seed questions, "
        "and are eligible for the Weaknesses workflows."
    )

    # ── Format guide ──────────────────────────────────────────────────────
    with st.expander("📋 Format guide & constraints", expanded=True):
        st.markdown(
            """
**Supported file types:** `.csv` or `.jsonl` &nbsp;|&nbsp; **Max file size:** 10 MB &nbsp;|&nbsp; **Max rows:** 10,000

---

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `stem` | string | The question text shown to the model. |
| `key` | string | The gold answer (exact match). |

### Optional columns

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `topic` | string | `custom` | Label used for heatmap grouping. |
| `difficulty` | string or int | `medium` | `easy` / `medium` / `hard`  or  `1` / `2` / `3` |
| `solution_steps` | string | *(empty)* | Reference solution — shown to SME, **never** sent to the policy model. |
| `numeric` | bool | `true` | Whether to normalise the answer as a number before matching. |
| `split` | string | `train` | `train`, `eval`, or `harness`. |

> `difficulty` is the friendly alias for the internal `level` field.  
> If you include both `difficulty` and `level`, `difficulty` wins.

---

### Minimal CSV example
```csv
stem,key,topic,difficulty,solution_steps
"What is 3! + 2! ?",8,combinatorics,easy,"3!=6, 2!=2, 6+2=8"
"How many edges in K₅?",10,graph_theory,medium,"K_n has n(n-1)/2 edges; K5 → 5×4/2=10"
"Rank of [[1,2],[2,4]]",1,linear_algebra,hard,"Rows are linearly dependent → rank 1"
```

### JSONL example
```jsonl
{"stem": "What is 3! + 2!?", "key": "8", "topic": "combinatorics", "difficulty": "easy", "solution_steps": "3!=6, 2!=2"}
{"stem": "How many edges in K5?", "key": "10", "difficulty": "medium"}
```
"""
        )

    # ── File uploader ─────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload questions (.csv or .jsonl)",
        type=["csv", "jsonl"],
        help="Required: stem, key. Optional: topic, difficulty, solution_steps, numeric, split",
    )

    if uploaded_file is not None:
        from frontier_lens.upload import delete_uploaded, import_rows, validate

        data = uploaded_file.read()
        result = validate(data, uploaded_file.name)

        if result.errors:
            for err in result.errors:
                st.error(err)
        else:
            if result.warnings:
                for w in result.warnings:
                    st.warning(w)

            n_new = len(result.rows) - result.n_already_in_db
            st.success(
                f"**{len(result.rows)} valid rows** found  "
                f"({result.n_duplicates_skipped} exact duplicates removed, "
                f"{result.n_already_in_db} already in DB, "
                f"**{n_new} new**)"
            )

            # ── Fuzzy near-duplicate review ───────────────────────────────
            if result.fuzzy_pairs:
                with st.expander(
                    f"⚠ {len(result.fuzzy_pairs)} near-duplicate pair(s) detected — review before importing",
                    expanded=True,
                ):
                    st.caption(
                        "These questions look identical after normalising case, "
                        "whitespace, and LaTeX. Both are included in the import by default. "
                        "Use the button below to skip the second occurrence of each pair."
                    )
                    pair_rows = []
                    for i, p in enumerate(result.fuzzy_pairs, 1):
                        pair_rows.append({
                            "#": i,
                            "Stem A": p.row_a["stem"][:80] + ("…" if len(p.row_a["stem"]) > 80 else ""),
                            "Stem B": p.row_b["stem"][:80] + ("…" if len(p.row_b["stem"]) > 80 else ""),
                            "Key A": p.row_a["key"],
                            "Key B": p.row_b["key"],
                            "Same key?": "✅" if p.same_key else "⚠ different",
                        })
                    st.dataframe(pair_rows, use_container_width=True)

                    # Store the choice in session state so it persists across reruns
                    dedup_choice = st.session_state.get("fuzzy_dedup_choice", "keep_all")
                    col_keep, col_skip = st.columns(2)
                    if col_keep.button("Import all rows (keep both variants)", key="fuzzy_keep"):
                        st.session_state["fuzzy_dedup_choice"] = "keep_all"
                        dedup_choice = "keep_all"
                    if col_skip.button("Skip second occurrence of each near-duplicate", key="fuzzy_skip"):
                        st.session_state["fuzzy_dedup_choice"] = "skip_b"
                        dedup_choice = "skip_b"

                    if dedup_choice == "skip_b":
                        st.info(
                            f"Will skip {len(result.fuzzy_pairs)} near-duplicate(s). "
                            f"{len(result.rows) - len(result.fuzzy_pairs)} rows will be imported."
                        )
                    else:
                        st.info(f"Will import all {len(result.rows)} rows (both variants of each pair).")
            else:
                dedup_choice = "keep_all"

            # Preview
            has_steps = any(r.get("solution_steps") for r in result.rows[:10])
            preview = []
            for r in result.rows[:10]:
                row_preview = {
                    "stem": r["stem"][:70] + ("…" if len(r["stem"]) > 70 else ""),
                    "key": r["key"],
                    "topic": r["topic"],
                    "difficulty": {1: "easy", 2: "medium", 3: "hard"}.get(r["level"], str(r["level"])),
                }
                if has_steps:
                    row_preview["solution_steps"] = (r.get("solution_steps") or "")[:60]
                preview.append(row_preview)
            st.dataframe(preview, use_container_width=True)
            if len(result.rows) > 10:
                st.caption(f"Showing 10 of {len(result.rows)} rows.")

            # Apply fuzzy dedup choice
            if dedup_choice == "skip_b":
                skip_ids = {p.row_b["id"] for p in result.fuzzy_pairs}
                base_rows = [r for r in result.rows if r["id"] not in skip_ids]
            else:
                base_rows = result.rows

            # Optional subsampling
            use_sample = st.checkbox(
                "Import only a random sample",
                value=False,
                help="Useful for quick experiments without importing the full dataset.",
            )
            if use_sample:
                sample_n = st.slider(
                    "Number of rows to import",
                    min_value=1,
                    max_value=len(base_rows),
                    value=min(20, len(base_rows)),
                )
                import random as _rng
                _seed_rows = _rng.Random(42).sample(base_rows, sample_n)
                st.caption(f"Will import **{sample_n}** of {len(base_rows)} rows.")
            else:
                _seed_rows = base_rows

            replace = st.checkbox(
                "Replace existing uploaded questions with same id",
                value=False,
                help="If unchecked, already-existing rows are silently skipped.",
            )

            if st.button("⬆ Import into database", type="primary", disabled=(role != "Organizer")):
                n = import_rows(_seed_rows, replace=replace, filename=uploaded_file.name)
                st.success(
                    f"Imported **{n}** new question(s) from `{uploaded_file.name}`. "
                    "They appear in the Evaluate tab immediately."
                )
                st.rerun()

            if role != "Organizer":
                st.caption("Switch Role to **Organizer** to import.")

    # ── Uploaded files in DB ──────────────────────────────────────────────
    st.divider()
    st.subheader("Questions currently in DB")

    with engine.connect() as conn:
        n_uploaded = conn.execute(
            text("SELECT COUNT(*) FROM questions WHERE source='uploaded'")
        ).scalar()

    if n_uploaded:
        with engine.connect() as conn:
            fname_rows = conn.execute(
                text("""
                    SELECT COALESCE(upload_filename,'(unnamed)') as fname, COUNT(*) as n
                    FROM questions WHERE source='uploaded'
                    GROUP BY upload_filename ORDER BY fname
                """)
            ).all()
        st.dataframe(
            [{"file": r[0], "questions": r[1]} for r in fname_rows],
            use_container_width=True,
        )
        if st.button(f"🗑 Delete all {n_uploaded} uploaded questions", disabled=(role != "Organizer")):
            from frontier_lens.upload import delete_uploaded
            n = delete_uploaded()
            st.success(f"Deleted {n} uploaded questions.")
            st.rerun()
    else:
        st.info("No uploaded questions in the database yet.")
