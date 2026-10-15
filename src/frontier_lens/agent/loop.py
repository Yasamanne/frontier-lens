from __future__ import annotations

import json
import os

from openai import OpenAI

from frontier_lens.agent.budget import MAX_STEPS
from frontier_lens.agent.tools import TOOLS, handle, tools_for
from frontier_lens.heatmap import weakest_cell
from frontier_lens.generate import _eval_cells
from frontier_lens.registry import sync_models


def propose_agent(
    run_id: str,
    model_id: str,
    n: int = 10,
    *,
    rl_split: str = "train",
) -> list[str]:
    specs = {m.id: m for m in sync_models()}
    gen = next(m for m in specs.values() if m.role == "generator")
    key = os.getenv(gen.api_key_env or "", "") if gen.api_key_env else "ollama"
    if gen.api_key_env and not key:
        raise RuntimeError(f"missing {gen.api_key_env}")
    client = OpenAI(base_url=gen.base_url, api_key=key or "ollama", timeout=120)
    toolset = tools_for(rl_split)
    if rl_split == "eval":
        cells = _eval_cells()
        cell_topic, cell_band = cells[0]
        intro = (
            f"Create original held-out eval item for topic={cell_topic} band={cell_band}. "
            "Never call estimate_pass_rate or get_cell_failures. Do not use any policy results. "
            "Call submit_candidate when ready."
        )
    else:
        cell = weakest_cell(run_id, model_id)
        if not cell:
            raise RuntimeError("no cell")
        cell_topic, cell_band = cell.topic, cell.band
        intro = (
            f"Create original train item for topic={cell_topic} band={cell_band}. "
            f"Use tools. run_id={run_id} model_id={model_id}. "
            "Call estimate_pass_rate and aim for 0 < pass_rate < 1. "
            "Do not query evaluated model outputs except via get_cell_failures. "
            "Call submit_candidate when ready."
        )

    ids: list[str] = []
    for i in range(n):
        messages = [
            {
                "role": "user",
                "content": f"{intro} Item {i+1}/{n}.",
            }
        ]
        for _step in range(MAX_STEPS):
            resp = client.chat.completions.create(
                model=gen.model, messages=messages, tools=toolset
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                break
            messages.append(msg)
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                if call.function.name == "get_cell_failures":
                    args.setdefault("run_id", run_id)
                    args.setdefault("model_id", model_id)
                    args.setdefault("topic", cell_topic)
                    args.setdefault("band", cell_band)
                result = handle(
                    call.function.name,
                    args,
                    gen_id=gen.id,
                    fail_model=model_id,
                    rl_split=rl_split,
                )
                if call.function.name == "submit_candidate":
                    try:
                        ids.append(json.loads(result)["candidate_id"])
                    except Exception:
                        pass
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            if any(c.function.name == "submit_candidate" for c in msg.tool_calls):
                break
    return ids
