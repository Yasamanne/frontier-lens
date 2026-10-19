import json
from pathlib import Path

from sqlalchemy import text
from streamlit.testing.v1 import AppTest

from frontier_lens import runstate
from frontier_lens.db import get_engine
from frontier_lens.generate import calibration
from frontier_lens.generate.calibration import estimate_pass_rate
from frontier_lens.generate.candidates import _store_candidate
from frontier_lens.providers.base import Completion
from frontier_lens.registry import sync_models

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "4_Generate_Review.py")

RAW = json.dumps(
    {"stem": "what number?", "lang": "python", "code": "print(4)\n", "input_files": {},
     "numeric": True}
)


def _candidate(mode, answers, split="train"):
    replies = iter(answers)

    def complete(_prompt):
        return next(replies)

    cid = _store_candidate(
        RAW, "python_state", "hard", "gen", mode, ["src"], rl_split=split,
        calibrate=split == "train", complete_fn=complete, calibration_k=len(answers),
    )
    with get_engine().connect() as conn:
        return conn.execute(
            text("SELECT status, auto_reject_reason, policy_pass_rate FROM candidates WHERE id=:i"),
            {"i": cid},
        ).mappings().one()


# ── augmentation candidates must land in the learnable band ────────────

def test_augmentation_candidate_the_policy_always_solves_is_rejected():
    row = _candidate("similar_augment", ["FINAL: 4"] * 4)
    assert (row["status"], row["auto_reject_reason"], row["policy_pass_rate"]) == (
        "auto_rejected", "out_of_band", 1.0)


def test_augmentation_candidate_the_policy_never_solves_is_rejected():
    row = _candidate("similar_augment", ["FINAL: 9"] * 4)
    assert (row["status"], row["auto_reject_reason"], row["policy_pass_rate"]) == (
        "auto_rejected", "out_of_band", 0.0)


def test_augmentation_candidate_in_the_band_reaches_the_sme_queue():
    row = _candidate("similar_augment", ["FINAL: 4", "FINAL: 9", "FINAL: 4", "FINAL: 9"])
    assert (row["status"], row["auto_reject_reason"], row["policy_pass_rate"]) == (
        "pending", None, 0.5)


def test_eval_split_candidates_are_not_calibrated_or_filtered():
    row = _candidate("similar_augment", [], split="eval")
    assert (row["status"], row["policy_pass_rate"]) == ("pending", None)


# ── calibration samples the policy the way Evaluate does ───────────────

def test_calibration_uses_the_policys_thinking_flag_and_output_cap(monkeypatch):
    seen: list[dict] = []

    class FakeProvider:
        def __init__(self, *a, **k):
            pass

        def complete(self, prompt, model, **kw):
            seen.append(kw)
            return Completion(text="FINAL: 4", latency_ms=1, tokens_in=1, tokens_out=1)

    monkeypatch.setattr(calibration, "OpenAICompatibleProvider", FakeProvider)
    estimate_pass_rate("stem", "4", k=2)
    calibration._calibrate_collect("stem", "4", True, 2)
    assert len(seen) == 4
    assert all(kw["max_tokens"] == 16000 and kw["thinking"] is False for kw in seen)


# ── the Model picker only lists models that have samples in the run ────

def _completed_run(run_id, model_id, created):
    runstate.insert_run(run_id, ["q"], 1, None, {}, status="complete")
    with get_engine().begin() as conn:
        conn.execute(text("UPDATE eval_runs SET created_at=:c WHERE id=:i"), {"c": created, "i": run_id})
        conn.execute(
            text(
                "INSERT INTO attempts (run_id, model_id, question_id, sample_index, raw_output, "
                "parsed_answer, is_correct, unparsable) VALUES (:r, :m, 'q1', 1, 'FINAL: 4', '4', 1, 0)"
            ),
            {"r": run_id, "m": model_id},
        )


def test_generate_page_lists_only_the_models_of_the_selected_run():
    sync_models()
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO questions (id, topic, level, band, split, template, params_json, stem,
                    code, lang, input_files_json, key, numeric, source)
                VALUES ('q1', 't', 2, 'standard', 'train', 't', '{}', 'stem', 'print(4)', 'python',
                        '{}', '4', 1, 'seed')
                """
            )
        )
    _completed_run("run_old", "gemma4-31b", "2026-10-01T00:00:00+00:00")
    _completed_run("run_new", "gemma4-26b-a4b", "2026-10-02T00:00:00+00:00")

    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state["role"] = "Viewer"
    at.run()
    assert not at.exception
    assert list(at.selectbox(key="gen_model_sel2").options) == ["gemma4-26b-a4b"]  # newest run

    at.selectbox(key="gen_run_sel2").select("run_old").run()
    assert not at.exception
    assert list(at.selectbox(key="gen_model_sel2").options) == ["gemma4-31b"]


# ── the gold key must be one non-empty line ─────────────────────────────

def _raw(code):
    return json.dumps(
        {"stem": "what number?", "lang": "python", "code": code, "input_files": {}, "numeric": True}
    )


def _store(code):
    replies = iter(["FINAL: 4", "FINAL: 9"])  # half right, so a valid candidate is in the band
    cid = _store_candidate(
        _raw(code), "python_state", "hard", "gen", "similar_augment", ["src"],
        calibrate=True, complete_fn=lambda _p: next(replies), calibration_k=2,
    )
    with get_engine().connect() as conn:
        return cid, conn.execute(
            text("SELECT status, auto_reject_reason, executed_key FROM candidates WHERE id=:i"),
            {"i": cid},
        ).mappings().one()


def test_candidate_whose_code_prints_nothing_is_auto_rejected():
    _, row = _store("x = 1 + 1\n")
    assert (row["status"], row["auto_reject_reason"], row["executed_key"]) == (
        "auto_rejected", "empty_output", None)


def test_candidate_whose_code_prints_several_lines_is_auto_rejected():
    _, row = _store("print(1)\nprint(2)\n")
    assert (row["status"], row["auto_reject_reason"]) == ("auto_rejected", "multi_line_output")


def test_agent_run_snippet_reports_a_bad_output_to_the_model():
    from frontier_lens.agent.tools import handle

    out = json.loads(
        handle("run_snippet", {"lang": "python", "code": "print(1)\nprint(2)\n"},
               gen_id="g", fail_model="")
    )
    assert out["ok"] is False and out["reason"] == "multi_line_output"


def test_approving_a_candidate_with_bad_output_does_not_promote_it():
    from frontier_lens.review import promote

    cid, _ = _store("print(4)\n")  # a good pending candidate...
    with get_engine().begin() as conn:  # ...whose code is then edited to print two lines
        conn.execute(text("UPDATE candidates SET code='print(1)\nprint(2)\n' WHERE id=:i"), {"i": cid})
    promote(cid, "hard_item", fair=True, novel=True, hard=True)
    with get_engine().connect() as conn:
        status, reason = conn.execute(
            text("SELECT status, auto_reject_reason FROM candidates WHERE id=:i"), {"i": cid}
        ).one()
        n_questions = conn.execute(
            text("SELECT COUNT(*) FROM questions WHERE source='generated'")
        ).scalar()
    assert (status, reason, n_questions) == ("auto_rejected", "multi_line_output", 0)
