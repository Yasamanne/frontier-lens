"""Per-run logs: prompt, model output, sandbox key, grade. No API keys."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from frontier_lens.config import LOGS_DIR
from frontier_lens.db import get_engine
from frontier_lens.prompt import contest_prompt


def jsonl_path(run_id: str) -> Path:
    return LOGS_DIR / f"{run_id}.jsonl"


def text_path(run_id: str) -> Path:
    return LOGS_DIR / f"{run_id}.txt"


def write_event(run_id: str, event: str, **fields) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "event": event,
        **fields,
    }
    LOGS_DIR.mkdir(exist_ok=True)
    with jsonl_path(run_id).open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with text_path(run_id).open("a") as fh:
        fh.write(_format_text(rec) + "\n")


def _format_text(rec: dict) -> str:
    event = rec.get("event")
    if event == "run_start":
        ts = rec.get("ts", "")
        return (
            f"=== run {rec.get('run_id')} | {rec.get('mode')} | started {ts} ===\n"
            f"batch    {rec.get('notes', 'live')}\n"
            f"models   {rec.get('models')}\n"
            f"k={rec.get('k')}  levels={rec.get('levels')}  limit={rec.get('limit')}\n"
            f"n_items  {rec.get('n_items')}\n"
            f"status   {rec.get('status', 'running')}"
        )
    if event == "contest_sample":
        params = rec.get("params") or {}
        return "\n".join(
            [
                f"--- step sample={rec.get('sample_index')} ---",
                f"model: {rec.get('model_id')} ({rec.get('provider_model')})",
                f"question_id: {rec.get('question_id')}",
                (
                    f"source: {rec.get('source')}  template: {rec.get('template')}  "
                    f"level: {rec.get('level')}  split: {rec.get('split')}  "
                    f"topic: {rec.get('topic')}  band: {rec.get('band')}"
                ),
                f"params: {json.dumps(params)}",
                f"gold_key (sandbox): {rec.get('gold_key')}",
                "",
                "INPUT (prompt):",
                rec.get("prompt") or "",
                "",
                "OUTPUT (model):",
                rec.get("raw_output") or "",
                "",
                (
                    f"parsed: {rec.get('parsed_answer')}  reward: {rec.get('reward')}  "
                    f"correct: {rec.get('is_correct')}  unparsable: {rec.get('unparsable')}  "
                    f"latency_ms: {rec.get('latency_ms')}"
                ),
            ]
        )
    if event == "run_end":
        ts = rec.get("ts", "")
        line = f"=== end {rec.get('run_id')} | status={rec.get('status')} | finished {ts} ==="
        if rec.get("error"):
            line += f"\nerror: {rec.get('error')}"
        return line
    return json.dumps(rec, ensure_ascii=False)


def dump_run(run_id: str) -> Path:
    """Rebuild a text+jsonl log from SQLite attempts (no extra model calls)."""
    engine = get_engine()
    jsonl_path(run_id).unlink(missing_ok=True)
    text_path(run_id).unlink(missing_ok=True)
    with engine.connect() as conn:
        run = conn.execute(
            text("SELECT id, created_at, k, notes, status FROM eval_runs WHERE id=:id"),
            {"id": run_id},
        ).mappings().first()
        if not run:
            raise RuntimeError(f"unknown run {run_id}")
        rows = conn.execute(
            text(
                """
                SELECT a.model_id, a.question_id, a.sample_index, a.raw_output, a.parsed_answer,
                       a.is_correct, a.unparsable, a.latency_ms,
                       q.stem, q.key, q.numeric, q.level, q.topic, q.band, q.split,
                       q.template, q.params_json, q.source, m.model as provider_model
                FROM attempts a
                JOIN questions q ON q.id=a.question_id
                LEFT JOIN models m ON m.id=a.model_id
                WHERE a.run_id=:r
                ORDER BY a.model_id, q.level, q.id, a.sample_index
                """
            ),
            {"r": run_id},
        ).mappings().all()
        models = conn.execute(
            text("SELECT DISTINCT model_id FROM attempts WHERE run_id=:r"),
            {"r": run_id},
        ).scalars().all()
    write_event(
        run_id,
        "run_start",
        mode=run["notes"],
        models=list(models),
        k=run["k"],
        n_items=len({r["question_id"] for r in rows}),
        dumped_from="sqlite",
        status=run["status"],
        created_at=run["created_at"],
    )
    for r in rows:
        params = json.loads(r["params_json"] or "{}")
        write_event(
            run_id,
            "contest_sample",
            model_id=r["model_id"],
            provider_model=r["provider_model"],
            question_id=r["question_id"],
            sample_index=r["sample_index"],
            source=r["source"],
            template=r["template"],
            level=r["level"],
            split=r["split"],
            topic=r["topic"],
            band=r["band"],
            params=params,
            gold_key=r["key"],
            prompt=contest_prompt(r["stem"]),
            raw_output=r["raw_output"],
            parsed_answer=r["parsed_answer"],
            reward=int(bool(r["is_correct"])),
            is_correct=bool(r["is_correct"]),
            unparsable=bool(r["unparsable"]),
            latency_ms=r["latency_ms"],
        )
    write_event(run_id, "run_end", status=run["status"])
    return text_path(run_id)


def logs_for_download(run_id: str) -> dict[str, tuple[str, bytes]]:
    """Return {kind: (filename, bytes)} for Streamlit download_button. Rebuild if missing."""
    txt, jsonl = text_path(run_id), jsonl_path(run_id)
    if not txt.exists() or not jsonl.exists():
        dump_run(run_id)
    return {
        "txt": (txt.name, txt.read_bytes()),
        "jsonl": (jsonl.name, jsonl.read_bytes()),
    }


# ---------------------------------------------------------------------------
# Operation logs (rescue / augment workflows — keyed by an op_id, not run_id)
# ---------------------------------------------------------------------------

def op_jsonl_path(op_id: str) -> Path:
    return LOGS_DIR / f"op_{op_id}.jsonl"


def op_text_path(op_id: str) -> Path:
    return LOGS_DIR / f"op_{op_id}.txt"


_OP_ICONS = {
    "rescue_start":       "=== 🔴 RESCUE",
    "rescue_question":    "  ├─ Q",
    "hint_generated":     "  │  💡 hint",
    "calibration_sample": "  │    sample",
    "hint_result":        "  └─ →",
    "rescue_end":         "=== ✅ RESCUE END",
    "similar_start":      "=== 🟡 AUGMENT",
    "similar_generate":   "  ├─ generate",
    "sandbox_result":     "  │  🔧 sandbox",
    "similar_result":     "  └─ →",
    "similar_end":        "=== ✅ AUGMENT END",
}


def _format_op_event(rec: dict) -> str:
    evt = rec.get("event", "")
    pfx = _OP_ICONS.get(evt, "  ·")
    ts = rec.get("ts", "")[-8:]  # HH:MM:SS only

    if evt == "rescue_start":
        return (
            f"\n{pfx} START  {ts}\n"
            f"    questions={rec.get('n_questions')}  "
            f"run={str(rec.get('run_id',''))[:8]}  "
            f"model={rec.get('model_id')}"
        )
    if evt == "rescue_question":
        return (
            f"\n{pfx} [{rec.get('index')}/{rec.get('total')}]  "
            f"{rec.get('question_id','')[:12]}  "
            f"topic={rec.get('topic')}  band={rec.get('band')}  "
            f"dominant_mode={rec.get('dominant_mode')}"
        )
    if evt == "hint_generated":
        if rec.get("parse_ok"):
            hint = str(rec.get("hint", ""))[:120]
            return f"{pfx}: \"{hint}\""
        else:
            err = rec.get("parse_error", "unknown")
            raw = str(rec.get("raw_response_preview", ""))[:200]
            return f"{pfx} PARSE FAILED: {err}\n    raw: {raw}"
    if evt == "calibration_sample":
        mark = "✓" if rec.get("is_correct") else "✗"
        return f"{pfx} {rec.get('sample_index', '?')+1}: {mark}"
    if evt == "hint_result":
        rl = rec.get("rl_suitable")
        badge = "✅ RL-suitable" if rl else "❌ not suitable"
        return (
            f"{pfx} candidate={rec.get('candidate_id')}  "
            f"pass_rate={rec.get('pass_rate', 0):.2f}  "
            f"({rec.get('n_correct')}/{rec.get('n')})  {badge}"
        )
    if evt == "rescue_end":
        return (
            f"\n{pfx}  rescued={rec.get('n_rescued')}/{rec.get('n_total')}  "
            f"rejected={rec.get('n_rejected')}"
        )
    if evt == "similar_start":
        return (
            f"\n{pfx} START  {ts}\n"
            f"    seeds={rec.get('n_seeds')}  "
            f"n_each={rec.get('n_each')}  "
            f"total_candidates={rec.get('total_candidates')}"
        )
    if evt == "similar_generate":
        return (
            f"\n{pfx} [{rec.get('index')}/{rec.get('total')}]  "
            f"source={rec.get('source_id','')[:12]}  "
            f"topic={rec.get('topic')}  band={rec.get('band')}  "
            f"attempt={rec.get('attempt', 0)+1}"
        )
    if evt == "sandbox_result":
        ok = rec.get("ok")
        detail = "OK" if ok else f"FAIL ({rec.get('reason', '?')})"
        return f"{pfx}: {detail}"
    if evt == "similar_result":
        rate = rec.get("pass_rate")
        rate_str = f"{rate:.2f}" if rate is not None else "n/a"
        return (
            f"{pfx} candidate={rec.get('candidate_id')}  "
            f"status={rec.get('status')}  "
            f"pass_rate={rate_str}"
        )
    if evt == "similar_end":
        return (
            f"\n{pfx}  pending={rec.get('n_pending')}/{rec.get('n_total')}  "
            f"rejected={rec.get('n_rejected')}"
        )
    # fallback
    return f"  · [{ts}] {evt}  " + "  ".join(f"{k}={v}" for k, v in rec.items()
                                               if k not in {"ts", "op_id", "event"})


def write_op_event(op_id: str, event: str, **fields) -> None:
    """Write one structured event to the operation log (both JSONL and text)."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "op_id": op_id,
        "event": event,
        **fields,
    }
    LOGS_DIR.mkdir(exist_ok=True)
    with op_jsonl_path(op_id).open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with op_text_path(op_id).open("a") as fh:
        fh.write(_format_op_event(rec) + "\n")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    path = dump_run(args.run)
    print("wrote", path)


if __name__ == "__main__":
    main()
