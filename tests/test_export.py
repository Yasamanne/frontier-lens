import json

import pytest
from pydantic import ValidationError

from frontier_lens.export import EvalTaskRow, SftRow, TaskRow, dump_jsonl, load_task_rows
from frontier_lens.reward import REWARD_FN
from frontier_lens.seed import seed


def test_train_row_schema():
    row = TaskRow(
        id="abc",
        prompt="FINAL",
        ground_truth="1",
        reward_fn=REWARD_FN,
        template="state_l2",
        level=2,
        params={"idx": 0},
        policy_model="gemma4-e4b",
        policy_pass_rate=0.375,
        pass_at_k=1.0,
        generator_model=None,
        generator_mode=None,
        audit_batch_id=None,
        split="train",
    )
    dumped = json.loads(row.model_dump_json())
    assert dumped["reward_fn"] == "exact_match_v1"
    assert dumped["split"] == "train"


def test_eval_row_rejects_policy_feedback():
    with pytest.raises(ValidationError):
        EvalTaskRow(
            id="abc",
            prompt="FINAL",
            ground_truth="1",
            template="state_l3",
            level=3,
            params={},
            policy_model="gemma4-e4b",
            policy_pass_rate=0.5,
            pass_at_k=0.9,
            split="eval",
        )


def test_eval_jsonl_omits_policy_feedback(tmp_path):
    from sqlalchemy import text

    from frontier_lens.db import get_engine

    seed(force=False)
    engine = get_engine()
    with engine.connect() as conn:
        n_eval = conn.execute(
            text("SELECT COUNT(*) FROM questions WHERE split='eval'")
        ).scalar()
    if not n_eval:
        seed(force=True)
    rows = load_task_rows(split="eval", in_band_only=False)
    assert rows, "seed eval templates should export"
    path = dump_jsonl(tmp_path / "eval_tasks.jsonl", rows)
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        assert "policy_pass_rate" not in payload
        assert "pass_at_k" not in payload
        assert payload["split"] == "eval"
        EvalTaskRow.model_validate(payload)


def test_sft_row_schema():
    row = SftRow(prompt="p", completion="FINAL: 1", model="gemma4-e4b", item_id="x")
    assert row.model_dump()["item_id"] == "x"
