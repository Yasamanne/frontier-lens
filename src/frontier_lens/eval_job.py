"""Background evaluation jobs that outlive the Streamlit page.

The Evaluate page stores the run config in ``eval_runs`` and launches::

    python -m frontier_lens.eval_job RUN_ID

as a detached process that writes to ``logs/RUN_ID.stdout.log``. Nothing ties
the job to the browser session, so refreshing or leaving the page does not stop
it; the page reads progress back from ``attempts``. The same command resumes an
interrupted or failed run from the CLI.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import text

from frontier_lens import runstate
from frontier_lens.config import LOGS_DIR, POLICY_K, REFERENCE_K
from frontier_lens.contest import (
    mark_complete,
    mark_failed,
    mark_interrupted,
    run_contest,
    run_verifier,
)
from frontier_lens.db import get_engine


def stdout_log_path(run_id: str) -> Path:
    return LOGS_DIR / f"{run_id}.stdout.log"


def tail(run_id: str, n: int = 20) -> str:
    return runstate.tail_file(stdout_log_path(run_id), n)


def create(config: dict) -> str:
    """Store a queued run with *config* (see ``contest.run_config``) and return its id."""
    run_id = uuid.uuid4().hex[:12]
    k = config.get("k")
    runstate.insert_run(
        run_id,
        config["question_ids"],
        max(POLICY_K, REFERENCE_K) if k is None else k,
        config.get("notes"),
        config,
        status="queued",
    )
    return run_id


def launch(run_id: str) -> int:
    """Start (or resume) *run_id* in a detached process; return its pid."""
    row = runstate.load_run(run_id)
    if row is None:
        raise RuntimeError(f"unknown run {run_id}")
    # A freshly created run is active (queued, fresh heartbeat) but has no pid yet.
    if runstate.is_active(row) and row.get("pid"):
        raise RuntimeError(f"run {run_id} is already running")
    if not row.get("config_json"):
        raise RuntimeError(f"run {run_id} has no stored config; it cannot be resumed")
    # Fresh heartbeat gives the child STALE_SEC to start before it looks dead;
    # clearing pid makes any older process still on this run exit
    # (see runstate.owned_by). The child takes it over with runstate.claim.
    runstate.set_fields(
        run_id, status="queued", heartbeat_at=runstate.now_iso(), error=None, pid=None
    )
    pid = runstate.spawn_detached("frontier_lens.eval_job", run_id, stdout_log_path(run_id))
    runstate.set_fields(run_id, pid=pid)
    return pid


def start(config: dict) -> str:
    run_id = create(config)
    launch(run_id)
    return run_id


def stop(run_id: str) -> None:
    """Kill the job's process and mark the run interrupted (resumable)."""
    row = runstate.load_run(run_id)
    if row is None:
        return
    runstate.terminate(row)
    mark_interrupted(run_id, "stopped by user")


def flag_for_sme(question_ids: list[str]) -> None:
    if not question_ids:
        return
    with get_engine().begin() as conn:
        for qid in question_ids:
            conn.execute(
                text("UPDATE questions SET sme_flagged=1, sme_source='evaluate' WHERE id=:id"),
                {"id": qid},
            )


def run_job(run_id: str) -> None:
    """Primary contest, then verifier on all-wrong items, then SME flags. Idempotent."""
    cfg = runstate.run_config(runstate.load_run(run_id))
    if cfg is None:
        raise SystemExit(f"run {run_id} has no stored config")
    models = cfg["models"]
    verifier = cfg.get("verifier_model")
    try:
        run_contest(
            models,
            k=cfg.get("k"),
            levels=cfg.get("levels") or [1, 2, 3],
            replay=bool(cfg.get("replay")),
            question_ids=cfg["question_ids"],
            notes=cfg.get("notes"),
            reuse_prev_runs=bool(cfg.get("reuse_prev_runs")),
            judge_model_id=cfg.get("judge_model_id"),
            run_id=run_id,
            finalize=False,
        )
    except Exception as exc:
        # run_contest records failures in its sampling loop (then this is a
        # no-op); this covers setup errors such as no questions or unknown model.
        mark_failed(run_id, exc)
        raise
    with runstate.heartbeat(run_id):
        try:
            if verifier:
                k_eff = cfg["k"] if cfg.get("k") is not None else POLICY_K
                failed = runstate.zero_correct_questions(run_id, models)
                runstate.set_fields(
                    run_id, phase="verifier", planned_verifier=len(failed) * k_eff
                )
                if failed:
                    print(f"verifier {verifier} on {len(failed)} all-wrong question(s)", flush=True)
                    run_verifier(run_id, verifier, failed, k_eff)
            runstate.set_fields(run_id, phase="flagging")
            flag_for_sme(
                runstate.zero_correct_questions(run_id, models + ([verifier] if verifier else []))
            )
            mark_complete(run_id)
        except Exception as exc:
            mark_failed(run_id, exc)
            raise


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run or resume a stored evaluation run")
    p.add_argument("run_id")
    p.add_argument("--launched", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.launched:
        # launch() already refused active runs; from the CLI, check here.
        runstate.reap_stale()
        row = runstate.load_run(args.run_id)
        if row is None:
            raise SystemExit(f"unknown run {args.run_id}")
        if runstate.is_active(row):
            raise SystemExit(
                f"run {args.run_id} is already running (pid {row.get('pid')}); "
                "stop it from the Evaluate page first"
            )
        runstate.set_fields(args.run_id, status="queued", pid=None)  # claimable again
    run_job(args.run_id)


if __name__ == "__main__":
    main()
