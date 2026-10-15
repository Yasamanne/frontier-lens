"""Export JSONL for open RL trainers (e.g. TRL GRPO) and SFT rejection sampling.

The files are RL-ready: each train row names reward_fn=exact_match_v1, a sandbox
key, and the policy pass-rate used to keep the item in the learnable band.
They are not RL-proven until a training run is measured on eval_tasks.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text
from typing import Literal

from frontier_lens.config import DATA_DIR
from frontier_lens.db import get_engine
from frontier_lens.prompt import contest_prompt
from frontier_lens.registry import policy_spec
from frontier_lens.reward import REWARD_FN
from frontier_lens.stats import in_learnable_band, pass_at_k, pass_rate

EXPORT_DIR = DATA_DIR / "exports"


class TaskRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    ground_truth: str
    reward_fn: Literal["exact_match_v1"] = REWARD_FN
    template: str
    level: int
    params: dict
    policy_model: str | None = None
    policy_pass_rate: float | None = None
    pass_at_k: float | None = Field(default=None, alias="pass_at_k")
    generator_model: str | None = None
    generator_mode: str | None = None
    audit_batch_id: str | None = None
    split: Literal["train", "eval"]

    @model_validator(mode="after")
    def eval_has_no_policy_feedback(self) -> "TaskRow":
        if self.split == "eval" and (
            self.policy_pass_rate is not None or self.pass_at_k is not None
        ):
            raise ValueError("eval rows must not carry policy-feedback fields")
        return self


class EvalTaskRow(TaskRow):
    """Eval rows use the same columns except policy-feedback rates are omitted."""

    split: Literal["eval"] = "eval"
    policy_pass_rate: None = None
    pass_at_k: None = None


class SftRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    completion: str
    model: str
    item_id: str


def _policy_rates(conn, run_id: str | None, question_id: str, policy_id: str) -> tuple[float | None, float | None]:
    if not run_id:
        return None, None
    rows = conn.execute(
        text(
            """
            SELECT is_correct FROM attempts
            WHERE run_id=:r AND model_id=:m AND question_id=:q
            ORDER BY sample_index
            """
        ),
        {"r": run_id, "m": policy_id, "q": question_id},
    ).scalars().all()
    if not rows:
        return None, None
    n = len(rows)
    c = int(sum(rows))
    return pass_rate(c, n), pass_at_k(n, c, n)


def _latest_run(conn) -> str | None:
    return conn.execute(
        text("SELECT id FROM eval_runs WHERE status='complete' ORDER BY created_at DESC LIMIT 1")
    ).scalar()


def load_task_rows(*, split: str, in_band_only: bool = True) -> list[TaskRow]:
    policy = policy_spec()
    engine = get_engine()
    rows: list[TaskRow] = []
    with engine.connect() as conn:
        run_id = _latest_run(conn)
        qrows = conn.execute(
            text(
                """
                SELECT id, stem, key, template, level, params_json, generator_model,
                       generator_mode, split, audit_batch_id
                FROM questions
                WHERE split=:split AND source IN ('seed', 'generated')
                """
            ),
            {"split": split},
        ).mappings().all()
        for q in qrows:
            rate = pak = None
            if split == "train":
                rate, pak = _policy_rates(conn, run_id, q["id"], policy.id)
                if rate is None:
                    cand_rate = conn.execute(
                        text(
                            """
                            SELECT policy_pass_rate, pass_at_k FROM candidates
                            WHERE id = json_extract(:params, '$.candidate')
                            """
                        ),
                        {"params": q["params_json"]},
                    ).mappings().first()
                    if cand_rate:
                        rate = cand_rate["policy_pass_rate"]
                        pak = cand_rate["pass_at_k"]
                if in_band_only and rate is not None and not in_learnable_band(rate):
                    continue
            payload = {
                "id": q["id"],
                "prompt": contest_prompt(q["stem"]),
                "ground_truth": q["key"],
                "reward_fn": REWARD_FN,
                "template": q["template"],
                "level": q["level"],
                "params": json.loads(q["params_json"] or "{}"),
                "policy_model": policy.id,
                "generator_model": q["generator_model"],
                "generator_mode": q["generator_mode"],
                "audit_batch_id": q["audit_batch_id"],
                "split": split,
            }
            if split == "train":
                payload["policy_pass_rate"] = rate
                payload["pass_at_k"] = pak
                rows.append(TaskRow.model_validate(payload))
            else:
                rows.append(EvalTaskRow.model_validate(payload))
    return rows


def load_sft_rows(run_id: str | None = None) -> list[SftRow]:
    engine = get_engine()
    out: list[SftRow] = []
    with engine.connect() as conn:
        if run_id is None:
            run_id = _latest_run(conn)
        if not run_id:
            return []
        rows = conn.execute(
            text(
                """
                SELECT a.raw_output, a.model_id, a.question_id, q.stem
                FROM attempts a JOIN questions q ON q.id=a.question_id
                JOIN models m ON m.id=a.model_id
                WHERE a.run_id=:r AND a.is_correct=1 AND m.role IN ('policy', 'reference')
                """
            ),
            {"r": run_id},
        ).mappings().all()
        for r in rows:
            if not r["raw_output"]:
                continue
            out.append(
                SftRow(
                    prompt=contest_prompt(r["stem"]),
                    completion=r["raw_output"],
                    model=r["model_id"],
                    item_id=r["question_id"],
                )
            )
    return out


def dump_jsonl(path: Path, models: list[BaseModel]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in models:
            data = row.model_dump(exclude_none=False)
            if isinstance(row, EvalTaskRow) or data.get("split") == "eval":
                data.pop("policy_pass_rate", None)
                data.pop("pass_at_k", None)
            fh.write(json.dumps(data) + "\n")
    return path


def export_all(out_dir: Path | None = None, *, in_band_only: bool = True) -> dict[str, Path]:
    out_dir = out_dir or EXPORT_DIR
    train = load_task_rows(split="train", in_band_only=in_band_only)
    eval_rows = load_task_rows(split="eval", in_band_only=False)
    sft = load_sft_rows()
    paths = {
        "rl_tasks": dump_jsonl(out_dir / "rl_tasks.jsonl", train),
        "eval_tasks": dump_jsonl(out_dir / "eval_tasks.jsonl", eval_rows),
        "sft": dump_jsonl(out_dir / "sft_rejection_sampled.jsonl", sft),
    }
    print(
        f"wrote {len(train)} train, {len(eval_rows)} eval, {len(sft)} sft rows under {out_dir}"
    )
    return paths


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(EXPORT_DIR))
    p.add_argument("--all-train", action="store_true", help="include items outside the learnable band")
    args = p.parse_args()
    export_all(Path(args.out), in_band_only=not args.all_train)


if __name__ == "__main__":
    main()
