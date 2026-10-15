from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from frontier_lens.config import POLICY_K
from frontier_lens.db import get_engine
from frontier_lens.heatmap import weakest_cell
from frontier_lens.prompt import contest_prompt
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import policy_spec, sync_models
from frontier_lens.reward import grade
from frontier_lens.sandbox import execute
from frontier_lens.split import EVAL_TEMPLATES
from frontier_lens.stats import in_learnable_band, pass_at_k, pass_rate
from frontier_lens.templates import all_specs

ONE_SHOT_TRAIN = """You write a NEW original train item for open-weight models.
Topic: {topic}
Band: {band}
These items were hard (failed often). Do not clone them:
{failures}

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf;
python may import math, itertools, collections, decimal only. No network, no files outside input_files.
The policy should neither always fail nor always pass (0 < pass_rate < 1).
"""

ONE_SHOT_EVAL = """You write a NEW original held-out eval item for open-weight models.
Topic: {topic}
Band: {band}
Do not use any policy-model results. This item must not be calibrated to a policy.

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf;
python may import math, itertools, collections, decimal only. No network, no files outside input_files.
"""


def _eval_cells() -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for spec in all_specs():
        if spec.template in EVAL_TEMPLATES:
            cell = (spec.topic, spec.band)
            if cell not in seen:
                seen.append(cell)
    return seen or [("python_state", "hard")]


def _failures(run_id: str, model_id: str, topic: str, band: str, limit: int = 6) -> str:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT q.stem, q.id, AVG(a.is_correct) as acc
                FROM attempts a JOIN questions q ON q.id=a.question_id
                WHERE a.run_id=:r AND a.model_id=:m AND q.topic=:t AND q.band=:b AND q.split='train'
                GROUP BY q.id
                HAVING acc < 0.5
                LIMIT :n
                """
            ),
            {"r": run_id, "m": model_id, "t": topic, "b": band, "n": limit},
        ).mappings().all()
    if not rows:
        return "(no listed failures)"
    return "\n---\n".join(r["stem"][:800] for r in rows)


def estimate_pass_rate(
    stem: str,
    key: str,
    numeric: bool = True,
    *,
    complete_fn=None,
    k: int | None = None,
) -> dict:
    """Sample the policy k times. Train-split calibration only — never call on eval."""
    policy = policy_spec()
    n = k if k is not None else POLICY_K
    if complete_fn is None:
        provider = OpenAICompatibleProvider(policy.base_url or "", policy.api_key_env)

        def complete_fn(prompt: str) -> str:  # noqa: F811
            return provider.complete(prompt, policy.model).text

    correct = 0
    for _ in range(n):
        text_out = complete_fn(contest_prompt(stem))
        reward, _, _ = grade(text_out, key, numeric)
        correct += reward
    rate = pass_rate(correct, n)
    return {
        "n": n,
        "correct": correct,
        "pass_rate": rate,
        "pass_at_k": pass_at_k(n, correct, n),
        "in_learnable_band": in_learnable_band(rate),
        "policy_model": policy.id,
    }


def propose_one_shot(
    run_id: str,
    model_id: str,
    n: int = 10,
    *,
    rl_split: str = "train",
) -> list[str]:
    if rl_split not in {"train", "eval"}:
        raise ValueError(f"bad split {rl_split}")
    specs = {m.id: m for m in sync_models()}
    gen = next(m for m in specs.values() if m.role == "generator")
    provider = OpenAICompatibleProvider(gen.base_url or "", gen.api_key_env)
    ids = []
    if rl_split == "eval":
        cells = _eval_cells()
        for i in range(n):
            topic, band = cells[i % len(cells)]
            prompt = ONE_SHOT_EVAL.format(topic=topic, band=band)
            completion = provider.complete(prompt, gen.model)
            cid = _store_candidate(
                completion.text,
                topic,
                band,
                gen.id,
                "one_shot",
                [],
                rl_split="eval",
                calibrate=False,
            )
            ids.append(cid)
        return ids

    cell = weakest_cell(run_id, model_id)
    if not cell:
        raise RuntimeError("no heatmap cell")
    fails = _failures(run_id, model_id, cell.topic, cell.band)
    for _ in range(n):
        prompt = ONE_SHOT_TRAIN.format(topic=cell.topic, band=cell.band, failures=fails)
        completion = provider.complete(prompt, gen.model)
        cid = _store_candidate(
            completion.text,
            cell.topic,
            cell.band,
            gen.id,
            "one_shot",
            [model_id],
            rl_split="train",
            calibrate=True,
        )
        ids.append(cid)
    return ids


def _store_candidate(
    raw: str,
    topic: str,
    band: str,
    gen_id: str,
    mode: str,
    fail_ids: list[str],
    *,
    rl_split: str = "train",
    calibrate: bool = True,
    complete_fn=None,
) -> str:
    cid = uuid.uuid4().hex[:12]
    status = "pending"
    reason = None
    key = None
    payload = None
    rate = None
    pak = None
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        payload = json.loads(raw[start : end + 1])
    except Exception:
        status, reason = "auto_rejected", "unsafe"
    stem = code = lang = ""
    files = {}
    numeric = True
    if payload:
        stem = payload.get("stem") or ""
        code = payload.get("code") or ""
        lang = payload.get("lang") or "python"
        files = payload.get("input_files") or {}
        numeric = bool(payload.get("numeric", True))
        result = execute(lang, code, files, twice=True)
        if not result.ok:
            status, reason = "auto_rejected", result.reason or "exec_error"
        else:
            key = result.stdout
            if rl_split == "train" and calibrate and status == "pending":
                stats = estimate_pass_rate(
                    stem, key, numeric, complete_fn=complete_fn
                )
                rate = stats["pass_rate"]
                pak = stats["pass_at_k"]
                if mode == "one_shot" and not stats["in_learnable_band"]:
                    status, reason = "auto_rejected", "out_of_band"
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO candidates (
                    id, topic, band, stem, code, lang, input_files_json, executed_key, numeric,
                    status, auto_reject_reason, generator_model, generator_mode, source_fail_ids,
                    created_at, policy_pass_rate, pass_at_k, rl_split
                ) VALUES (
                    :id, :topic, :band, :stem, :code, :lang, :files, :key, :numeric,
                    :status, :reason, :gm, :mode, :fails, :ts, :rate, :pak, :split
                )
                """
            ),
            {
                "id": cid,
                "topic": topic,
                "band": band,
                "stem": stem,
                "code": code,
                "lang": lang,
                "files": json.dumps(files),
                "key": key,
                "numeric": int(numeric),
                "status": status,
                "reason": reason,
                "gm": gen_id,
                "mode": mode,
                "fails": json.dumps(fail_ids),
                "ts": datetime.now(timezone.utc).isoformat(),
                "rate": rate,
                "pak": pak,
                "split": rl_split,
            },
        )
    print(f"candidate {cid} {status} {reason} split={rl_split} pass_rate={rate}")
    return cid


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--mode", choices=["one_shot", "agent"], default="one_shot")
    p.add_argument("--split", choices=["train", "eval"], default="train")
    args = p.parse_args()
    if args.mode == "agent":
        from frontier_lens.agent.loop import propose_agent

        ids = propose_agent(args.run, args.model, args.n, rl_split=args.split)
    else:
        ids = propose_one_shot(args.run, args.model, args.n, rl_split=args.split)
    print("candidates", ids)


if __name__ == "__main__":
    main()
