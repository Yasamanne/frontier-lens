from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.heatmap import weakest_cell
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import sync_models
from frontier_lens.sandbox import execute

ONE_SHOT = """You write a NEW original eval item for open-weight models.
Topic: {topic}
Band: {band}
These items were hard (failed often). Do not clone them:
{failures}

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf;
python may import math, itertools, collections, decimal only. No network, no files outside input_files.
"""


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


def propose_one_shot(run_id: str, model_id: str, n: int = 10) -> list[str]:
    specs = {m.id: m for m in sync_models()}
    gen = next(m for m in specs.values() if m.role == "generator")
    cell = weakest_cell(run_id, model_id)
    if not cell:
        raise RuntimeError("no heatmap cell")
    provider = OpenAICompatibleProvider(gen.base_url or "", gen.api_key_env)
    fails = _failures(run_id, model_id, cell.topic, cell.band)
    ids = []
    for _ in range(n):
        prompt = ONE_SHOT.format(topic=cell.topic, band=cell.band, failures=fails)
        completion = provider.complete(prompt, gen.model)
        cid = _store_candidate(completion.text, cell.topic, cell.band, gen.id, "one_shot", [model_id])
        ids.append(cid)
    return ids


def _store_candidate(raw: str, topic: str, band: str, gen_id: str, mode: str, fail_ids: list[str]) -> str:
    cid = uuid.uuid4().hex[:12]
    status = "pending"
    reason = None
    key = None
    payload = None
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
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO candidates (
                    id, topic, band, stem, code, lang, input_files_json, executed_key, numeric,
                    status, auto_reject_reason, generator_model, generator_mode, source_fail_ids, created_at
                ) VALUES (
                    :id, :topic, :band, :stem, :code, :lang, :files, :key, :numeric,
                    :status, :reason, :gm, :mode, :fails, :ts
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
            },
        )
    print(f"candidate {cid} {status} {reason}")
    return cid


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--mode", choices=["one_shot", "agent"], default="one_shot")
    args = p.parse_args()
    if args.mode == "agent":
        from frontier_lens.agent.loop import propose_agent

        ids = propose_agent(args.run, args.model, args.n)
    else:
        ids = propose_one_shot(args.run, args.model, args.n)
    print("candidates", ids)


if __name__ == "__main__":
    main()
