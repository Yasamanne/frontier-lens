from sqlalchemy import text

from frontier_lens import runstate
from frontier_lens.db import get_engine
from frontier_lens.failure_analysis import failure_modes
from frontier_lens.heatmap import heatmap, weakest_cell
from frontier_lens.metrics import HIST_LABELS, histogram_counts, policy_item_rates
from frontier_lens.scope import POOLED

M = "gemma4-26b-a4b"


def _question(conn, qid, topic="t", band="standard", key="4", flagged=0):
    conn.execute(
        text(
            """
            INSERT INTO questions (id, topic, level, band, split, template, params_json, stem,
                code, lang, input_files_json, key, numeric, source, sme_flagged)
            VALUES (:id, :topic, 2, :band, 'train', 't', '{}', 'stem', '', 'python', '{}',
                    :key, 1, 'seed', :flagged)
            """
        ),
        {"id": qid, "topic": topic, "band": band, "key": key, "flagged": flagged},
    )


def _attempts(conn, run, qid, answers, start=1):
    """answers: list of parsed answers; the key is '4' so '4' is correct."""
    for i, ans in enumerate(answers, start=start):
        conn.execute(
            text(
                "INSERT INTO attempts (run_id, model_id, question_id, sample_index, raw_output, "
                "parsed_answer, is_correct, unparsable) VALUES (:r, :m, :q, :i, :raw, :a, :ok, 0)"
            ),
            {"r": run, "m": M, "q": qid, "i": i, "raw": f"FINAL: {ans}", "a": ans,
             "ok": int(ans == "4")},
        )


def _run(run_id, status="complete"):
    runstate.insert_run(run_id, ["q"], 8, None, {}, status=status)


def test_histogram_places_rates_in_the_right_bar_for_any_k():
    labels = HIST_LABELS
    counts = histogram_counts([0, 1 / 8, 0.2, 1 / 3, 0.4, 7 / 8, 0.95, 1])
    by_label = dict(zip(labels, counts))
    assert by_label["0 (out)"] == 1
    assert by_label["0–⅛"] == 1        # 1/8, upper edge inclusive
    assert by_label["⅛–¼"] == 1        # 0.2
    assert by_label["¼–⅜"] == 1        # 1/3 (k=3), was misplaced one bar left
    assert by_label["⅜–½"] == 1        # 0.4
    assert by_label["¾–⅞"] == 1        # 7/8
    assert by_label["⅞–1"] == 1        # 0.95, had no bar before
    assert by_label["1 (out)"] == 1
    assert sum(counts) == 8


def test_broken_questions_are_left_out_and_pending_ones_stay():
    engine = get_engine()
    _run("r")
    with engine.begin() as conn:
        _question(conn, "ok")
        _question(conn, "pending", flagged=1)
        _question(conn, "broken", flagged=2)
        for q in ("ok", "pending", "broken"):
            _attempts(conn, "r", q, ["4", "5"])
    assert sorted(i["id"] for i in policy_item_rates("r", M)) == ["ok", "pending"]
    assert heatmap("r", M)[0].n == 2
    assert failure_modes("r", M, "t", "standard")["n_failing_questions"] == 0


def test_pooling_adds_samples_of_all_completed_runs():
    engine = get_engine()
    _run("r1")
    _run("r2")
    _run("r3", status="interrupted")
    with engine.begin() as conn:
        _question(conn, "q1")
        _attempts(conn, "r1", "q1", ["4", "5"])
        _attempts(conn, "r2", "q1", ["5", "5"])
        _attempts(conn, "r3", "q1", ["4", "4"])  # not complete: ignored
    (single,) = policy_item_rates("r1", M)
    (pooled,) = policy_item_rates(POOLED, M)
    assert (single["correct"], single["n"]) == (1, 2)
    assert (pooled["correct"], pooled["n"]) == (1, 4)


def test_dominant_mode_is_none_when_no_mode_stands_out():
    engine = get_engine()
    _run("r")
    with engine.begin() as conn:
        _question(conn, "q1", key="10")
        # Two distinct wrong answers, not off by one: no mode applies.
        for i, ans in enumerate(["7", "7", "20", "20"], start=1):
            conn.execute(
                text(
                    "INSERT INTO attempts (run_id, model_id, question_id, sample_index, "
                    "raw_output, parsed_answer, is_correct, unparsable) "
                    "VALUES ('r', :m, 'q1', :i, :raw, :a, 0, 0)"
                ),
                {"m": M, "i": i, "raw": f"FINAL: {ans}", "a": ans},
            )
    result = failure_modes("r", M, "t", "standard")
    assert result["n_failing_questions"] == 1
    assert result["dominant_mode"] == "none"


def test_weakest_cell_prefers_confirmed_weak_cells_and_skips_tiny_ones():
    engine = get_engine()
    _run("r")
    with engine.begin() as conn:
        # tiny: one failing question, accuracy 0 but n=1
        _question(conn, "tiny", topic="tiny")
        _attempts(conn, "r", "tiny", ["5"] * 4)
        # big weak: 8 failing questions
        for i in range(8):
            _question(conn, f"w{i}", topic="weak")
            _attempts(conn, "r", f"w{i}", ["5"] * 4)
        # big strong: 8 solved questions
        for i in range(8):
            _question(conn, f"s{i}", topic="strong")
            _attempts(conn, "r", f"s{i}", ["4"] * 4)
    cell = weakest_cell("r", M)
    assert (cell.topic, cell.n, cell.weak) == ("weak", 8, True)


# ── pooling scope: selected run + completed runs, one dataset ───────────

def _dq(conn, qid, source="uploaded", filename="a.csv"):
    conn.execute(
        text(
            """
            INSERT INTO questions (id, topic, level, band, split, template, params_json, stem,
                code, lang, input_files_json, key, numeric, source, upload_filename)
            VALUES (:id, 't', 2, 'standard', 'train', 't', '{}', 'stem', '', 'python', '{}',
                    '4', 1, :src, :fn)
            """
        ),
        {"id": qid, "src": source, "fn": filename},
    )


def _scoped_ids(scope, dataset=None):
    return sorted(i["id"] for i in policy_item_rates(scope, M, dataset=dataset))


def test_pooling_always_includes_the_selected_run_even_if_it_is_not_complete():
    from frontier_lens.scope import pool_with

    engine = get_engine()
    _run("done_a")                          # complete, file a.csv
    _run("partial", status="interrupted")   # the run the user is looking at, file a.csv
    _run("other_partial", status="failed")  # another incomplete run: must stay out
    with engine.begin() as conn:
        for qid in ("a1", "a2", "p1", "o1"):
            _dq(conn, qid)
        _attempts(conn, "done_a", "a1", ["4"])
        _attempts(conn, "done_a", "a2", ["4"])
        _attempts(conn, "partial", "p1", ["4"])
        _attempts(conn, "other_partial", "o1", ["4"])
    assert _scoped_ids("partial") == ["p1"]
    # Pooled is never smaller than the run itself: it adds the completed run, not the other partial one.
    assert _scoped_ids(pool_with("partial")) == ["a1", "a2", "p1"]


def test_pooling_one_dataset_leaves_other_datasets_out():
    from frontier_lens.scope import DATASET_BUILTIN, pool_with

    engine = get_engine()
    _run("ra")
    _run("rb")
    _run("rs")
    _run("sel", status="interrupted")
    with engine.begin() as conn:
        _dq(conn, "qa", filename="a.csv")
        _dq(conn, "qb", filename="b.csv")
        _dq(conn, "qs", source="seed", filename="")
        _dq(conn, "qsel", filename="a.csv")
        _attempts(conn, "ra", "qa", ["4"])
        _attempts(conn, "rb", "qb", ["4"])
        _attempts(conn, "rs", "qs", ["4"])
        _attempts(conn, "sel", "qsel", ["4"])
    scope = pool_with("sel")
    assert _scoped_ids(scope) == ["qa", "qb", "qs", "qsel"]
    assert _scoped_ids(scope, dataset="a.csv") == ["qa", "qsel"]
    assert _scoped_ids(scope, dataset="b.csv") == ["qb"]
    assert _scoped_ids(scope, dataset=DATASET_BUILTIN) == ["qs"]
    # Heatmap and failure analysis take the same scope and dataset.
    assert heatmap(scope, M, dataset="b.csv")[0].n == 1
    assert failure_modes(scope, M, "t", "standard", dataset="a.csv")["n_failing_questions"] == 0
