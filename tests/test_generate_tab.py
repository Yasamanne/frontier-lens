import json
from pathlib import Path
from types import SimpleNamespace as NS

from sqlalchemy import text
from streamlit.testing.v1 import AppTest

from frontier_lens import runstate
from frontier_lens.db import get_engine
from frontier_lens.generate import calibration, similar
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


# ── ReAct agent + Review (the merged augmentation mode) ─────────────────

SEED = {"id": "seed1", "stem": "seed stem", "key": "7", "numeric": 1,
        "topic": "python_state", "band": "hard"}


def _draft(stem="new stem", code="print(4)\n"):
    return {"stem": stem, "lang": "python", "code": code, "topic": "python_state",
            "band": "hard", "numeric": True}


def _call(call_id, name, args):
    return NS(id=call_id, function=NS(name=name, arguments=json.dumps(args)))


class FakeAgentClient:
    """Each chat call answers with the next scripted list of tool calls (None = no tool call)."""

    def __init__(self, rounds):
        self.rounds = iter(rounds)
        self.seen: list[list] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.seen.append(list(kw["messages"]))
        calls = next(self.rounds, None)
        return NS(choices=[NS(message=NS(tool_calls=calls, content=""))])


class FakeJudge:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.prompts: list[str] = []

    def complete(self, prompt, model, **kw):
        self.prompts.append(prompt)
        score = next(self.scores)
        body = {"score": score, "passed": score >= 75, "criterion_scores": {},
                "issues": [] if score >= 75 else ["too close to the seed"],
                "fix_instructions": "make it more different"}
        return Completion(text=json.dumps(body), latency_ms=1, tokens_in=1, tokens_out=1)


def _run_react_review(rounds, scores, monkeypatch, **kw):
    sync_models()
    monkeypatch.setattr(similar, "sme_feedback_for", lambda *a, **k: "SME-FEEDBACK-MARKER")
    client, judge = FakeAgentClient(rounds), FakeJudge(scores)
    replies = iter(["FINAL: 4", "FINAL: 9"] * 4)  # half right: in the band
    results = similar.generate_similar_react_review(
        [SEED], max_loops=3, calibration_k=2, complete_fn=lambda _p: next(replies),
        _client=client, _judge_provider=judge, **kw,
    )
    return results, client, judge


def _candidates():
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(
            text("SELECT stem, code, status, generator_mode, source_fail_ids FROM candidates")
        ).mappings()]


def test_react_review_agent_tests_its_code_then_a_passing_draft_is_stored(monkeypatch):
    sync_models()
    with get_engine().begin() as conn:  # an earlier variant of the same seed
        conn.execute(text(
            "INSERT INTO candidates (id, stem, status, generator_mode, source_fail_ids, topic, band) "
            "VALUES ('old', 'OLD VARIANT STEM', 'pending', 'similar_augment', '[\"seed1\"]', "
            "'python_state', 'hard')"))
    rounds = [
        [_call("1", "run_snippet", {"lang": "python", "code": "print(4)\n"})],
        [_call("2", "submit_draft", _draft())],
    ]
    results, client, judge = _run_react_review(rounds, [85], monkeypatch)

    assert [(r["status"], r["reason"]) for r in results] == [("pending", None)]
    stored = [c for c in _candidates() if c["stem"] == "new stem"]
    assert len(stored) == 1
    assert (stored[0]["generator_mode"], json.loads(stored[0]["source_fail_ids"])) == (
        "similar_augment", ["seed1"])
    first_prompt = client.seen[0][0]["content"]
    assert "SME-FEEDBACK-MARKER" in first_prompt          # SME feedback from other samples
    assert "OLD VARIANT STEM" in first_prompt             # earlier variants of this seed
    assert "Gold answer of the seed" in first_prompt and "7" in first_prompt
    assert "SME-FEEDBACK-MARKER" in judge.prompts[0]      # the reviewer sees it too
    tool_msgs = [m for m in client.seen[1] if isinstance(m, dict) and m.get("role") == "tool"]
    assert json.loads(tool_msgs[0]["content"])["ok"] is True   # run_snippet really ran


def test_react_review_failed_review_goes_back_into_the_same_conversation(monkeypatch):
    rounds = [[_call("1", "submit_draft", _draft("first stem"))],
              [_call("2", "submit_draft", _draft("second stem"))]]
    results, client, judge = _run_react_review(rounds, [60, 90], monkeypatch)

    assert [r["status"] for r in results] == ["pending"]
    assert [c["stem"] for c in _candidates()] == ["second stem"]
    feedback = [m for m in client.seen[1] if isinstance(m, dict) and m.get("role") == "user"][-1]
    assert "REVIEWER FEEDBACK" in feedback["content"] and "too close to the seed" in feedback["content"]
    assert len(judge.prompts) == 2


def test_react_review_stores_nothing_when_no_draft_reaches_the_pass_mark(monkeypatch):
    rounds = [[_call("1", "submit_draft", _draft())], [_call("2", "submit_draft", _draft())]]
    results, _, _ = _run_react_review(rounds, [50, 55], monkeypatch, max_fix_iterations=1)
    assert [r["status"] for r in results] == ["review_failed"]
    assert "2 drafts" in results[0]["reason"]
    assert _candidates() == []


def test_react_review_agent_that_never_submits_is_handled(monkeypatch):
    results, _, judge = _run_react_review([None], [], monkeypatch, max_fix_iterations=0)
    assert [r["status"] for r in results] == ["review_failed"]
    assert judge.prompts == [] and _candidates() == []


def test_react_review_bad_sandbox_output_never_reaches_the_reviewer(monkeypatch):
    rounds = [[_call("1", "submit_draft", _draft(code="print(1)\nprint(2)\n"))],
              [_call("2", "submit_draft", _draft("fixed stem"))]]
    results, client, judge = _run_react_review(rounds, [88], monkeypatch)
    assert [r["status"] for r in results] == ["pending"]
    assert len(judge.prompts) == 1                      # only the runnable draft was reviewed
    fix = [m for m in client.seen[1] if isinstance(m, dict) and m.get("role") == "user"][-1]
    assert "multi_line_output" in fix["content"]


# ── the augment job: dispatch and the saved reject reason ───────────────

def test_augment_job_dispatches_react_review_and_keeps_the_reject_reason(monkeypatch):
    from frontier_lens import jobs

    got: dict = {}

    def fake(seeds, *, max_loops, max_fix_iterations, log_fn, **common):
        got.update(seeds=seeds, max_loops=max_loops, max_fix_iterations=max_fix_iterations)
        log_fn("similar_result", source_id="seed1", candidate_id="c1", status="auto_rejected",
               pass_rate=1.0, auto_reject_reason="out_of_band")

    monkeypatch.setattr("frontier_lens.generate.generate_similar_react_review", fake)
    monkeypatch.setattr(jobs, "_log_fn", lambda job_id, fmt, on_event: (
        lambda event, **fields: on_event(event, fields)))
    res = NS(items=[], skipped=[], add=lambda item, skipped=False: res.items.append(item))
    cfg = {"run_id": "r", "mode": "react_review", "seeds": [SEED], "n_each": 1,
           "calibration_k": 2, "max_loops": 4, "max_fix_iterations": 2}
    jobs._run_augment("job1", cfg, res)

    assert (got["max_loops"], got["max_fix_iterations"]) == (4, 2)
    assert [(i["status"], i["reason"]) for i in res.items] == [("auto_rejected", "out_of_band")]


def test_augmentation_controls_offer_one_shot_and_the_merged_mode():
    sync_models()
    with get_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO questions (id, topic, level, band, split, template, params_json, stem, "
            "code, lang, input_files_json, key, numeric, source) VALUES "
            "('q1', 't', 2, 'standard', 'train', 't', '{}', 'stem', 'print(4)', 'python', "
            "'{}', '4', 1, 'seed')"))
    _completed_run("run1", "gemma4-26b-a4b", "2026-10-02T00:00:00+00:00")
    with get_engine().begin() as conn:  # one right and one wrong sample: in the band
        conn.execute(text(
            "INSERT INTO attempts (run_id, model_id, question_id, sample_index, raw_output, "
            "parsed_answer, is_correct, unparsable) VALUES "
            "('run1', 'gemma4-26b-a4b', 'q1', 2, 'FINAL: 9', '9', 0, 0)"))

    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state["role"] = "Organizer"
    at.run()
    assert not at.exception
    assert list(at.radio(key="aug_mode").options) == ["🎯 One-shot", "🤖 ReAct agent + Review"]
    assert at.radio(key="aug_mode").value == "🤖 ReAct agent + Review"
    assert at.slider(key="aug_loops").value == 3 and at.slider(key="aug_fix_iters").value == 3
    # 1 candidate: (3 fix + 1 drafts) x (3 loops + 1 review) + k=4 calibration = 20 calls
    assert any("~20 API calls" in c.value for c in at.caption)

    at.radio(key="aug_mode").set_value("🎯 One-shot").run()
    assert not at.exception
    assert any("~5 API calls" in c.value for c in at.caption)  # 1 + k
