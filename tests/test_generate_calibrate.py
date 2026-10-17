import itertools

from frontier_lens.agent.tools import handle, tools_for
from frontier_lens.generate import _store_candidate, estimate_pass_rate
from frontier_lens.seed import seed


def test_estimate_pass_rate_learnable_band():
    def always_right(_prompt: str) -> str:
        return "FINAL: 3"

    stats = estimate_pass_rate("x", "3", complete_fn=always_right, k=8)
    assert stats["pass_rate"] == 1.0
    assert not stats["in_learnable_band"]

    def always_wrong(_prompt: str) -> str:
        return "FINAL: 0"

    stats = estimate_pass_rate("x", "3", complete_fn=always_wrong, k=8)
    assert stats["pass_rate"] == 0.0
    assert not stats["in_learnable_band"]

    calls = itertools.count(1)  # atomic under concurrent sampling

    def mixed(_prompt: str) -> str:
        return "FINAL: 3" if next(calls) % 2 == 0 else "FINAL: 0"

    stats = estimate_pass_rate("x", "3", complete_fn=mixed, k=8)
    assert stats["pass_rate"] == 0.5
    assert stats["in_learnable_band"]


def test_one_shot_post_filters_out_of_band():
    seed(force=False)
    raw = (
        '{"stem":"what number?","lang":"python","code":"print(4)\\n",'
        '"input_files":{},"numeric":true}'
    )

    def always_right(_prompt: str) -> str:
        return "FINAL: 4"

    cid = _store_candidate(
        raw,
        "python_state",
        "hard",
        "glm-5.3",
        "one_shot",
        ["gemma4-e4b"],
        rl_split="train",
        calibrate=True,
        complete_fn=always_right,
    )
    from sqlalchemy import text

    from frontier_lens.db import get_engine

    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT status, auto_reject_reason, policy_pass_rate FROM candidates WHERE id=:id"),
            {"id": cid},
        ).mappings().one()
    assert row["status"] == "auto_rejected"
    assert row["auto_reject_reason"] == "out_of_band"
    assert row["policy_pass_rate"] == 1.0


def test_eval_split_skips_policy_and_keeps_pending():
    seed(force=False)
    raw = (
        '{"stem":"what number?","lang":"python","code":"print(4)\\n",'
        '"input_files":{},"numeric":true}'
    )

    def boom(_prompt: str) -> str:
        raise AssertionError("eval must not sample the policy")

    cid = _store_candidate(
        raw,
        "python_state",
        "hard",
        "glm-5.3",
        "one_shot",
        [],
        rl_split="eval",
        calibrate=False,
        complete_fn=boom,
    )
    from sqlalchemy import text

    from frontier_lens.db import get_engine

    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT status, policy_pass_rate, rl_split FROM candidates WHERE id=:id"),
            {"id": cid},
        ).mappings().one()
    assert row["status"] == "pending"
    assert row["policy_pass_rate"] is None
    assert row["rl_split"] == "eval"


def test_eval_tools_exclude_policy_calls():
    names = {t["function"]["name"] for t in tools_for("eval")}
    assert "estimate_pass_rate" not in names
    assert "get_cell_failures" not in names
    blocked = handle(
        "estimate_pass_rate",
        {"stem": "x", "lang": "python", "code": "print(1)"},
        gen_id="glm-5.3",
        fail_model="gemma4-e4b",
        rl_split="eval",
    )
    assert "must not call the policy" in blocked
