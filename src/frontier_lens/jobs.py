"""Background jobs for the long operations other than evaluation runs.

Kinds: ``rescue`` (dead-zone hint ladder), ``augment`` (similar questions),
``calibrate`` (difficulty knobs) and ``seed`` (rebuild the seed bank). The page
stores the job config in ``op_jobs`` and launches::

    python -m frontier_lens.jobs JOB_ID

as a detached process, so refreshing or leaving the page does not stop it. Each
finished unit of work (a rescued question, a generated candidate, a calibrated
topic) is saved to ``op_job_items`` as it completes; resuming an interrupted or
failed job skips those. The formatted log goes to ``logs/op_JOB_ID.stdout.log``
and structured events to ``logs/op_JOB_ID.{txt,jsonl}``.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import text

from frontier_lens import runstate
from frontier_lens.config import LOGS_DIR
from frontier_lens.db import get_engine
from frontier_lens.runlog import write_op_event

TABLE = "op_jobs"
KINDS = ("rescue", "augment", "calibrate", "seed")


# ── Job records ──────────────────────────────────────────────────────────

def stdout_log_path(job_id: str) -> Path:
    return LOGS_DIR / f"op_{job_id}.stdout.log"


def tail(job_id: str, n: int = 30) -> str:
    return runstate.tail_file(stdout_log_path(job_id), n)


def load(job_id: str) -> dict | None:
    return runstate.load_run(job_id, table=TABLE)


def config(row: dict) -> dict:
    return json.loads(row["config_json"])


def results(row: dict | None) -> dict:
    """``{"items": [...], "skipped": [...]}`` saved so far for a job."""
    out: dict[str, list[dict]] = {"items": [], "skipped": []}
    if not row:
        return out
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT skipped, item_json FROM op_job_items WHERE job_id=:j ORDER BY id"),
            {"j": row["id"]},
        ).all()
    for skipped, item_json in rows:
        out["skipped" if skipped else "items"].append(json.loads(item_json))
    return out


def create(kind: str, cfg: dict, planned: int) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind!r}")
    job_id = f"{kind}_{uuid.uuid4().hex[:8]}"
    now = runstate.now_iso()
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO op_jobs (
                    id, kind, created_at, status, config_json, heartbeat_at, planned, done
                ) VALUES (:id, :kind, :ts, 'queued', :cfg, :ts, :planned, 0)
                """
            ),
            {
                "id": job_id,
                "kind": kind,
                "ts": now,
                "cfg": json.dumps(cfg, default=str),
                "planned": planned,
            },
        )
    return job_id


def launch(job_id: str) -> int:
    """Start (or resume) *job_id* in a detached process; return its pid."""
    row = load(job_id)
    if row is None:
        raise RuntimeError(f"unknown job {job_id}")
    # A freshly created job is active (queued, fresh heartbeat) but has no pid yet.
    if runstate.is_active(row) and row.get("pid"):
        raise RuntimeError(f"job {job_id} is already running")
    # Fresh heartbeat gives the child STALE_SEC to start; clearing pid makes
    # any older process still on this job exit (see runstate.owned_by). The
    # child takes it over with runstate.claim.
    runstate.set_fields(
        job_id, table=TABLE,
        status="queued", heartbeat_at=runstate.now_iso(), error=None, pid=None,
    )
    pid = runstate.spawn_detached("frontier_lens.jobs", job_id, stdout_log_path(job_id))
    runstate.set_fields(job_id, table=TABLE, pid=pid)
    return pid


def start(kind: str, cfg: dict, planned: int) -> str:
    job_id = create(kind, cfg, planned)
    launch(job_id)
    return job_id


def stop(job_id: str) -> None:
    """Kill the job's process and mark it interrupted (resumable)."""
    row = load(job_id)
    if row is None:
        return
    runstate.terminate(row)
    if runstate.finish(job_id, "interrupted", table=TABLE, error="stopped by user"):
        write_op_event(job_id, "job_end", status="interrupted", error="stopped by user")


def active(kinds: tuple[str, ...] = KINDS) -> list[dict]:
    runstate.reap_stale(table=TABLE)
    return [r for r in runstate.active_runs(table=TABLE) if r["kind"] in kinds]


def recent(kind: str, limit: int = 10) -> list[dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM op_jobs WHERE kind=:k ORDER BY created_at DESC LIMIT :n"),
            {"k": kind, "n": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


# ── Log formatting (one line per event, tagged with its question) ────────

def format_rescue_event(event: str, **f) -> str:
    if event == "rescue_start":
        return f"[START] {f.get('n_questions')}q | k={f.get('calibration_k')} | max_hints={f.get('max_hints')}"
    if event == "rescue_question":
        return (f"\n[Q {f.get('index')}/{f.get('total')}] {str(f.get('question_id', ''))[:12]}  "
                f"{f.get('topic')}/{f.get('band')}  key={str(f.get('gold_key', ''))[:20]}")
    if event == "rescue_question_skip":
        return f"  ⏭ skipped: {f.get('reason')}"
    if event == "hint_attempt_start":
        return f"\n  ── Hint round {f.get('level', '?')}/{f.get('max_hints', '?')} ──"
    if event == "hint_generated":
        if f.get("parse_ok"):
            return f"  💡 L{f.get('level', '?')} (✓): {str(f.get('hint', ''))[:100]}"
        return (f"  ⚠ L{f.get('level', '?')} parse FAILED: {f.get('parse_error', '')} | "
                f"raw: {str(f.get('raw_response_preview', ''))[:150]}")
    if event == "calibration_sample":
        return f"    sample {f.get('sample_index', 0) + 1}: {'✓' if f.get('is_correct') else '✗'}"
    if event == "hint_attempt_result":
        badge = ("✅ rescued!" if f.get("rl_suitable")
                 else f"❌ still failing ({f.get('n_correct')}/{f.get('n')})")
        return f"  L{f.get('level', '?')}: {badge}  pass_rate={f.get('pass_rate', 0):.2f}"
    if event == "hint_result":
        if f.get("rl_suitable"):
            return (f"  ✅ RESCUED L{f.get('rescue_level')} | cand={f.get('candidate_id')} "
                    f"pass_rate={f.get('pass_rate', 0):.2f}")
        return f"  ❌ UNRESCUABLE after {f.get('n_hints_tried')} hint(s) → SFT export"
    if event == "rescue_end":
        return f"\n[DONE] rescued={f.get('n_rescued')}/{f.get('n_total')}  rejected={f.get('n_rejected')}"
    return f"[{event}]"


def format_augment_event(event: str, **f) -> str:
    if event == "similar_start":
        return (f"[START] seeds={f.get('n_seeds')} n_each={f.get('n_each')} "
                f"total={f.get('total_candidates')} k={f.get('calibration_k')} "
                f"mode={f.get('mode', '')} max_fix={f.get('max_fix_iterations', '')}")
    if event == "similar_generate":
        return (f"\n[{f.get('index')}/{f.get('total')}] source={str(f.get('source_id', ''))[:12]}  "
                f"{f.get('topic')}/{f.get('band')}")
    if event in ("sandbox_result", "review_sandbox"):
        return f"  🔧 sandbox: {'OK ✓' if f.get('ok') else 'FAIL ✗ (' + str(f.get('reason', '')) + ')'}"
    if event == "calibration_sample":
        return f"  sample {f.get('sample_index', 0) + 1}: {'✓' if f.get('is_correct') else '✗'}"
    if event == "similar_result":
        rate = f.get("pass_rate")
        rate_str = f"{rate:.2f}" if rate is not None else "n/a"
        icon = "✅" if f.get("status") == "pending" else "❌"
        rev = f" [review={f.get('review_score')}]" if f.get("review_score") else ""
        return f"  {icon} cand={f.get('candidate_id', '?')} status={f.get('status')} pass_rate={rate_str}{rev}"
    if event == "similar_end":
        return f"\n[DONE] pending={f.get('n_pending')}/{f.get('n_total')}"
    if event == "review_generate":
        return f"  📝 generating (iter {f.get('fix_iter', 0)})…"
    if event == "review_start":
        return f"  🔍 reviewing (iter {f.get('fix_iter', 0)})…"
    if event == "review_result":
        badge = "✅ PASSED" if f.get("passed", False) else "❌ FAILED"
        issues = "; ".join(f.get("issues", []))
        scores = " | ".join(
            f"{k.replace('_', ' ')[:8]}={v}" for k, v in f.get("criterion_scores", {}).items()
        )
        return (f"  {badge} score={f.get('score', 0)}/100  {scores}\n"
                + (f"    issues: {issues}" if issues else ""))
    if event == "review_fix":
        issues = "; ".join(str(i) for i in f.get("issues", []))[:120]
        return f"\n  🔧 Fix iter {f.get('fix_iter')}: score was {f.get('score')} — issues: {issues}"
    if event == "agent_tool_call":
        return f"  ↳ step {f.get('step')} tool={f.get('tool')}"
    if event == "agent_end":
        cid = f.get("candidate_id")
        return f"  🏁 done  steps={f.get('steps_used')}  {'cand=' + cid if cid else 'no submit'}"
    return f"[{event}]"


def _print_event(line: str, fields: dict) -> None:
    # Items run concurrently, so tag each line with its source question.
    tag = fields.get("question_id") or fields.get("source_id")
    if tag:
        nl = "\n" if line.startswith("\n") else ""
        line = f"{nl}[{str(tag)[:8]}] {line.lstrip(chr(10))}"
    print(line, flush=True)


# ── Runner ───────────────────────────────────────────────────────────────

class _Results:
    """Finished work for one job; :meth:`add` saves each item as it completes."""

    def __init__(self, job_id: str, row: dict) -> None:
        self.job_id = job_id
        saved = results(row)
        self.items: list[dict] = saved["items"]
        self.skipped: list[dict] = saved["skipped"]
        self._lock = threading.Lock()

    def add(self, item: dict, *, skipped: bool = False) -> None:
        with self._lock:
            (self.skipped if skipped else self.items).append(item)
            with get_engine().begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO op_job_items (job_id, skipped, item_json) "
                        "VALUES (:j, :s, :item)"
                    ),
                    {"j": self.job_id, "s": int(skipped), "item": json.dumps(item, default=str)},
                )
                conn.execute(
                    text("UPDATE op_jobs SET done=:n WHERE id=:j"),
                    {"n": len(self.items) + len(self.skipped), "j": self.job_id},
                )


def _log_fn(job_id: str, fmt: Callable[..., str], on_event: Callable[..., bool]) -> Callable:
    """Event sink: *on_event* may consume an event (return True) instead of logging it."""
    def log_fn(event: str, **fields) -> None:
        if on_event(event, fields):
            return
        write_op_event(job_id, event, **fields)
        _print_event(fmt(event, **fields), fields)
    return log_fn


def _run_rescue(job_id: str, cfg: dict, res: _Results) -> None:
    from frontier_lens.generate import rescue_with_hints

    finished = {r["question_id"] for r in res.items} | {r["question_id"] for r in res.skipped}
    todo = [qid for qid in cfg["question_ids"] if qid not in finished]
    if not todo:
        return

    def on_event(event: str, fields: dict) -> bool:
        if event == "rescue_question_done":
            res.add(fields["result"])
            return True  # saved as a job item; too large for the event log
        if event == "rescue_question_skip":
            res.add({"question_id": fields["question_id"], "reason": fields.get("reason")},
                    skipped=True)
        return False

    rescue_with_hints(
        cfg["run_id"], cfg["model_id"], todo,
        log_fn=_log_fn(job_id, format_rescue_event, on_event),
        calibration_k=cfg["calibration_k"], max_hints=cfg["max_hints"],
    )


def _run_augment(job_id: str, cfg: dict, res: _Results) -> None:
    from frontier_lens.generate import (
        generate_similar_questions,
        generate_similar_react,
        generate_similar_react_review,
        generate_similar_with_review,
    )

    n_each = cfg["n_each"]
    have = Counter(r["source_id"] for r in res.items)
    # Group seeds by how many candidates they still need so one call per group
    # tops each seed up to n_each.
    by_missing: dict[int, list[dict]] = defaultdict(list)
    for seed_q in cfg["seeds"]:
        missing = n_each - have[seed_q["id"]]
        if missing > 0:
            by_missing[missing].append(seed_q)

    def on_event(event: str, fields: dict) -> bool:
        if event == "similar_result":
            res.add({
                "source_id": fields["source_id"],
                "candidate_id": fields.get("candidate_id") or "",
                "status": fields.get("status"),
                "pass_rate": fields.get("pass_rate"),
                "reason": fields.get("auto_reject_reason"),
            })
        return False

    log_fn = _log_fn(job_id, format_augment_event, on_event)
    for missing, seeds in sorted(by_missing.items()):
        common = {"run_id": cfg["run_id"], "n_each": missing,
                  "log_fn": log_fn, "calibration_k": cfg["calibration_k"]}
        if cfg["mode"] == "react_review":
            generate_similar_react_review(
                seeds, max_loops=cfg["max_loops"],
                max_fix_iterations=cfg["max_fix_iterations"], **common,
            )
        elif cfg["mode"] == "review":
            generate_similar_with_review(
                seeds, max_fix_iterations=cfg["max_fix_iterations"], **common
            )
        elif cfg["mode"] == "react":
            generate_similar_react(seeds, max_loops=cfg["max_loops"], **common)
        else:
            generate_similar_questions(seeds, **common)


def _run_calibrate(job_id: str, cfg: dict, res: _Results) -> None:
    from frontier_lens.calibrate import (
        GRIDS,
        KNOBS_PATH,
        measure_rates,
        print_knob_report,
        write_knobs,
    )
    from frontier_lens.registry import policy_spec

    policy = policy_spec()
    k = cfg["k"]
    finished = {r["topic"] for r in res.items}
    for topic in cfg["topics"]:
        if topic in finished:
            continue
        runstate.set_fields(job_id, table=TABLE, phase=topic)
        print(f"[{topic}] measuring {len(GRIDS[topic])} grid points at k={k}…", flush=True)
        rates = measure_rates(topic, k=k)
        res.add({"topic": topic, "rates": rates})  # kept for resume
        print(f"[{topic}] pass rates {[round(r, 2) for r in rates]}", flush=True)
    # Write once, after every topic is measured: the file is marked
    # source=policy, so a partial write would pass off unmeasured topics.
    write_knobs(
        {r["topic"]: r["rates"] for r in res.items},
        policy_id=policy.id, policy_model=policy.model, k=k,
    )
    print_knob_report()
    print("wrote", KNOBS_PATH, flush=True)


def _run_seed(job_id: str, cfg: dict, res: _Results) -> None:
    from frontier_lens.seed import seed

    res.add({"inserted": seed(force=bool(cfg.get("force")))})


_RUNNERS: dict[str, Callable[[str, dict, _Results], None]] = {
    "rescue": _run_rescue,
    "augment": _run_augment,
    "calibrate": _run_calibrate,
    "seed": _run_seed,
}


def run(job_id: str) -> None:
    """Run a stored job in this process. Safe to call again to resume."""
    row = load(job_id)
    if row is None:
        raise SystemExit(f"unknown job {job_id}")
    runner = _RUNNERS[row["kind"]]
    cfg = config(row)
    res = _Results(job_id, row)
    # Claim before the heartbeat starts: it exits the process when the job is
    # not running under this pid. Fails if Stop was clicked before we started.
    if not runstate.claim(job_id, table=TABLE):
        print(f"job {job_id} was stopped before it started", flush=True)
        return
    with runstate.heartbeat(job_id, table=TABLE):
        print(f"job {job_id} status=running kind={row['kind']} done={row.get('done') or 0}"
              f"/{row.get('planned')}", flush=True)
        try:
            runner(job_id, cfg, res)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if runstate.finish(job_id, "failed", table=TABLE, error=err[:2000]):
                write_op_event(job_id, "job_end", status="failed", error=err)
                print(f"job {job_id} status=failed error={err}", flush=True)
            raise
        except KeyboardInterrupt:
            if runstate.finish(job_id, "interrupted", table=TABLE, error="interrupted (Ctrl+C)"):
                write_op_event(job_id, "job_end", status="interrupted")
            raise
        if runstate.finish(job_id, "complete", table=TABLE, phase="done", error=None):
            write_op_event(job_id, "job_end", status="complete")
            print(f"job {job_id} status=complete", flush=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run or resume a stored background job")
    p.add_argument("job_id")
    p.add_argument("--launched", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.launched:
        # launch() already refused active jobs; from the CLI, check here.
        runstate.reap_stale(table=TABLE)
        row = load(args.job_id)
        if row is None:
            raise SystemExit(f"unknown job {args.job_id}")
        if runstate.is_active(row):
            raise SystemExit(
                f"job {args.job_id} is already running (pid {row.get('pid')}); stop it first"
            )
        runstate.set_fields(args.job_id, table=TABLE, status="queued", pid=None)  # claimable again
    run(args.job_id)


if __name__ == "__main__":
    main()
