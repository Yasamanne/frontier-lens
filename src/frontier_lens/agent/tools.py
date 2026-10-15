from __future__ import annotations

import json

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.generate import _store_candidate, estimate_pass_rate
from frontier_lens.sandbox import execute

BASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cell_failures",
            "description": "Failed train items in a cell. Do not call on the eval split.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "model_id": {"type": "string"},
                    "topic": {"type": "string"},
                    "band": {"type": "string"},
                },
                "required": ["run_id", "model_id", "topic", "band"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_snippet",
            "description": "Execute a snippet twice in the sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "lang": {"type": "string"},
                    "code": {"type": "string"},
                    "input_files": {"type": "object"},
                },
                "required": ["lang", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_similarity",
            "description": "Nearest seed stems by shared token overlap",
            "parameters": {
                "type": "object",
                "properties": {"stem": {"type": "string"}},
                "required": ["stem"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_candidate",
            "description": "Submit a candidate for execution and SME queue",
            "parameters": {
                "type": "object",
                "properties": {
                    "stem": {"type": "string"},
                    "lang": {"type": "string"},
                    "code": {"type": "string"},
                    "input_files": {"type": "object"},
                    "numeric": {"type": "boolean"},
                    "topic": {"type": "string"},
                    "band": {"type": "string"},
                },
                "required": ["stem", "lang", "code", "topic", "band"],
            },
        },
    },
]

ESTIMATE_TOOL = {
    "type": "function",
    "function": {
        "name": "estimate_pass_rate",
        "description": (
            "Sample the policy k=8 on this candidate and return pass_rate. "
            "Aim for the learnable band 0 < pass_rate < 1. Training split only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "stem": {"type": "string"},
                "lang": {"type": "string"},
                "code": {"type": "string"},
                "input_files": {"type": "object"},
                "numeric": {"type": "boolean"},
            },
            "required": ["stem", "lang", "code"],
        },
    },
}

# Backward-compatible default is the train toolset.
TOOLS = BASE_TOOLS + [ESTIMATE_TOOL]


def tools_for(rl_split: str) -> list[dict]:
    if rl_split == "eval":
        blocked = {"get_cell_failures", "estimate_pass_rate"}
        return [t for t in BASE_TOOLS if t["function"]["name"] not in blocked]
    return BASE_TOOLS + [ESTIMATE_TOOL]


def handle(
    name: str,
    args: dict,
    *,
    gen_id: str,
    fail_model: str,
    rl_split: str = "train",
) -> str:
    if name == "estimate_pass_rate":
        if rl_split != "train":
            return json.dumps({"error": "eval split must not call the policy"})
        result = execute(
            args["lang"], args["code"], args.get("input_files") or {}, twice=True
        )
        if not result.ok:
            return json.dumps({"ok": False, "reason": result.reason})
        stats = estimate_pass_rate(
            args["stem"], result.stdout, bool(args.get("numeric", True))
        )
        return json.dumps({"ok": True, "key": result.stdout, **stats})
    if name == "get_cell_failures":
        if rl_split != "train":
            return json.dumps({"error": "eval split must not use policy failures"})
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT q.stem, AVG(a.is_correct) acc FROM attempts a
                    JOIN questions q ON q.id=a.question_id
                    WHERE a.run_id=:r AND a.model_id=:m AND q.topic=:t AND q.band=:b AND q.split='train'
                    GROUP BY q.id HAVING acc < 0.67
                    LIMIT 8
                    """
                ),
                {
                    "r": args["run_id"],
                    "m": args["model_id"],
                    "t": args["topic"],
                    "b": args["band"],
                },
            ).mappings().all()
        return json.dumps([{"stem": r["stem"][:500], "acc": r["acc"]} for r in rows])
    if name == "run_snippet":
        result = execute(args["lang"], args["code"], args.get("input_files") or {}, twice=True)
        return json.dumps({"ok": result.ok, "stdout": result.stdout, "reason": result.reason})
    if name == "check_similarity":
        stem = set((args.get("stem") or "").lower().split())
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT id, stem FROM questions WHERE source='seed' LIMIT 200")
            ).mappings().all()
        scored = []
        for r in rows:
            toks = set(r["stem"].lower().split())
            if not toks:
                continue
            score = len(stem & toks) / len(stem | toks)
            scored.append((score, r["id"]))
        scored.sort(reverse=True)
        return json.dumps([{"id": i, "score": s} for s, i in scored[:5]])
    if name == "submit_candidate":
        raw = json.dumps(args)
        cid = _store_candidate(
            raw,
            args["topic"],
            args["band"],
            gen_id,
            "agent",
            [fail_model] if rl_split == "train" else [],
            rl_split=rl_split,
            calibrate=rl_split == "train",
        )
        return json.dumps({"candidate_id": cid})
    return json.dumps({"error": f"unknown tool {name}"})
