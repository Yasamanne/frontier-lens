"""Background job state in SQLite: config for resume, heartbeat, progress.

Used for evaluation runs (``eval_runs``) and for the other long operations —
rescue, augmentation, calibration, seeding (``op_jobs``, see
:mod:`frontier_lens.jobs`). Generic helpers take ``table=``; the default is
``eval_runs``.

A job is *active* while its status is ``queued``/``running`` and its heartbeat
is fresh. A job whose process died (container restart, Ctrl+C, kill) stops
beating and is marked ``interrupted`` by :func:`reap_stale`. Interrupted and
failed jobs that have a stored config can be resumed; finished work is skipped.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from frontier_lens.config import ROOT
from frontier_lens.db import get_engine
from frontier_lens.ids import item_set_hash

ACTIVE = ("queued", "running")
RESUMABLE = ("interrupted", "failed")
HEARTBEAT_SEC = 5.0
# Generous: SQLite busy_timeout is 30s, so one heartbeat write can stall that long.
STALE_SEC = 90.0

_COMMON_FIELDS = {"status", "config_json", "pid", "heartbeat_at", "phase", "error"}
# Updatable columns per job table; also the whitelist for table names in SQL.
_TABLE_FIELDS = {
    "eval_runs": _COMMON_FIELDS | {"planned_primary", "planned_verifier"},
    "op_jobs": _COMMON_FIELDS | {"planned", "done"},
}
# Jobs this process moved to a final status itself; their heartbeat stops
# quietly instead of treating the status change as a takeover.
_finished_here: set[tuple[str, str]] = set()


def _check(table: str, fields: dict) -> None:
    if table not in _TABLE_FIELDS:
        raise ValueError(f"unknown job table {table!r}")
    unknown = set(fields) - _TABLE_FIELDS[table]
    if unknown:
        raise ValueError(f"unknown {table} fields: {sorted(unknown)}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def active_cutoff(now: datetime | None = None) -> str:
    """ISO timestamp; a heartbeat at or after it counts as alive."""
    return ((now or datetime.now(timezone.utc)) - timedelta(seconds=STALE_SEC)).isoformat()


# SQL: no other evaluation run is alive. Used inside single statements so the
# check and the write are one atomic step even with several writers.
NO_OTHER_ACTIVE_RUN = (
    "NOT EXISTS (SELECT 1 FROM eval_runs o WHERE o.id != :id "
    "AND o.status IN ('queued','running') AND o.heartbeat_at >= :cutoff)"
)


def insert_run(
    run_id: str,
    question_ids: list[str],
    k: int,
    notes: str | None,
    config: dict,
    status: str,
    *,
    exclusive: bool = False,
) -> bool:
    """Insert a run row. With ``exclusive=True`` do nothing (return False) if another run is alive."""
    guard = f"WHERE {NO_OTHER_ACTIVE_RUN}" if exclusive else ""
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                f"""
                INSERT INTO eval_runs (
                    id, created_at, item_set_hash, status, k, notes, config_json, heartbeat_at
                ) SELECT :id, :ts, :h, :status, :k, :notes, :cfg, :ts {guard}
                """
            ),
            {
                "id": run_id,
                "ts": now_iso(),
                "h": item_set_hash(question_ids),
                "status": status,
                "k": k,
                "notes": notes,
                "cfg": json.dumps(config),
                "cutoff": active_cutoff(),
            },
        )
    return result.rowcount > 0


def load_run(run_id: str, *, table: str = "eval_runs") -> dict | None:
    _check(table, {})
    with get_engine().connect() as conn:
        row = conn.execute(
            text(f"SELECT * FROM {table} WHERE id=:id"), {"id": run_id}
        ).mappings().first()
    return dict(row) if row else None


def run_config(row: dict | None) -> dict | None:
    if not row or not row.get("config_json"):
        return None
    return json.loads(row["config_json"])


def set_fields(run_id: str, *, table: str = "eval_runs", **fields) -> None:
    _check(table, fields)
    if not fields:
        return
    assignments = ", ".join(f"{name}=:{name}" for name in fields)
    with get_engine().begin() as conn:
        conn.execute(
            text(f"UPDATE {table} SET {assignments} WHERE id=:id"),
            {**fields, "id": run_id},
        )


def finish(run_id: str, status: str, *, table: str = "eval_runs", **fields) -> bool:
    """Move an active job to a final *status*; no-op if it already left queued/running.

    Guards against overwriting a job that completed, failed, or was stopped by
    someone else in the meantime. Returns whether the row changed.
    """
    _check(table, fields)
    assignments = ", ".join(["status=:status"] + [f"{name}=:{name}" for name in fields])
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                f"UPDATE {table} SET {assignments} "
                "WHERE id=:id AND status IN ('queued','running')"
            ),
            {**fields, "status": status, "id": run_id},
        )
    if result.rowcount > 0:
        _finished_here.add((table, run_id))
        return True
    return False


def claim(run_id: str, *, table: str = "eval_runs") -> bool:
    """Mark a queued/running job as running under this process.

    Fails (returns False) if the job was stopped or finished in the meantime,
    e.g. Stop clicked just after launch. CLI resumes reset the status to
    ``queued`` first.
    """
    _check(table, {})
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                f"UPDATE {table} SET status='running', pid=:pid, heartbeat_at=:hb, error=NULL "
                "WHERE id=:id AND status IN ('queued','running')"
            ),
            {"id": run_id, "pid": os.getpid(), "hb": now_iso()},
        )
    _finished_here.discard((table, run_id))
    return result.rowcount > 0


def _age_sec(ts: str | None, now: datetime) -> float:
    if not ts:
        return float("inf")
    try:
        then = datetime.fromisoformat(ts)
    except ValueError:
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).total_seconds()


def is_active(row: dict, now: datetime | None = None) -> bool:
    if row.get("status") not in ACTIVE:
        return False
    now = now or datetime.now(timezone.utc)
    return _age_sec(row.get("heartbeat_at"), now) <= STALE_SEC


def is_resumable(row: dict) -> bool:
    return row.get("status") in RESUMABLE and bool(row.get("config_json"))


def reap_stale(now: datetime | None = None, *, table: str = "eval_runs") -> list[str]:
    """Mark queued/running jobs with no recent heartbeat as interrupted."""
    _check(table, {})
    now = now or datetime.now(timezone.utc)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT id, status, heartbeat_at FROM {table} WHERE status IN ('queued','running')")
        ).mappings().all()
    stale = [dict(r) for r in rows if not is_active(dict(r), now)]
    if not stale:
        return []
    with engine.begin() as conn:
        for r in stale:
            # Guard on heartbeat_at so a run that just beat is not clobbered.
            conn.execute(
                text(
                    f"""
                    UPDATE {table} SET status='interrupted'
                    WHERE id=:id AND status IN ('queued','running')
                      AND COALESCE(heartbeat_at, '') = COALESCE(:hb, '')
                    """
                ),
                {"id": r["id"], "hb": r["heartbeat_at"]},
            )
    return [r["id"] for r in stale]


def active_runs(*, table: str = "eval_runs") -> list[dict]:
    _check(table, {})
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT * FROM {table} WHERE status IN ('queued','running') "
                "ORDER BY created_at DESC"
            )
        ).mappings().all()
    return [dict(r) for r in rows if is_active(dict(r))]


def owned_by(row: dict | None, pid: int) -> bool:
    """Whether process *pid* may keep working on this run.

    False once the run was stopped/reaped (status left queued/running) or
    another process took it over by resuming it. Workers :func:`claim` the
    run first, so the stored pid is always theirs while they own it.
    """
    if not row or row.get("status") not in ACTIVE:
        return False
    return row.get("pid") == pid


@contextmanager
def heartbeat(
    run_id: str, interval: float = HEARTBEAT_SEC, *, table: str = "eval_runs"
) -> Iterator[None]:
    """Refresh ``heartbeat_at`` and ``pid`` from a daemon thread while the block runs.

    If the run stops being ours (Stop button, reaped after the machine slept,
    or resumed elsewhere), exit the process so two jobs never sample one run.
    """
    stop = threading.Event()
    pid = os.getpid()

    def beat() -> None:
        while True:
            try:
                if (table, run_id) in _finished_here:
                    return  # we finished it ourselves; nothing left to guard
                row = load_run(run_id, table=table)
                if not owned_by(row, pid):
                    print(
                        f"{run_id} status={row.get('status') if row else 'missing'}: "
                        f"no longer owned by pid {pid}; exiting",
                        flush=True,
                    )
                    sys.stdout.flush()
                    os._exit(3)
                set_fields(run_id, table=table, heartbeat_at=now_iso(), pid=pid)
            except Exception:
                pass  # a missed beat is fine; STALE_SEC covers several
            if stop.wait(interval):
                return

    thread = threading.Thread(target=beat, name=f"heartbeat-{run_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval)


# ── Processes ────────────────────────────────────────────────────────────

def spawn_detached(module: str, job_id: str, log_path: Path) -> int:
    """Run ``python -m <module> <job_id> --launched`` detached; return its pid."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", module, job_id, "--launched"],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            # Own process group: survives the page's script being stopped and
            # lets terminate() signal the job without touching Streamlit.
            start_new_session=True,
        )
    # Reap the child when it exits so finished jobs do not linger as zombies.
    threading.Thread(target=proc.wait, name=f"reap-{job_id}", daemon=True).start()
    return proc.pid


def terminate(row: dict) -> None:
    """SIGTERM the job's process if it is still live (fresh heartbeat)."""
    pid = row.get("pid")
    if not pid or not is_active(row):
        return
    try:
        if hasattr(os, "killpg") and os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def tail_file(path: Path, n: int = 20) -> str:
    """Last *n* lines of a log file (reads at most the final 16 KB)."""
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 16_384))
        lines = fh.read().decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


# ── Evaluation runs ──────────────────────────────────────────────────────

def progress(run_id: str) -> dict:
    """Counts from ``attempts`` so progress survives page reloads."""
    row = load_run(run_id) or {}
    cfg = run_config(row) or {}
    verifier = cfg.get("verifier_model")
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT model_id, COUNT(*) n, SUM(is_correct) correct, SUM(unparsable) unparsable
                FROM attempts WHERE run_id=:r GROUP BY model_id
                """
            ),
            {"r": run_id},
        ).mappings().all()
    primary = {"done": 0, "correct": 0, "unparsable": 0}
    ver = {"done": 0, "correct": 0, "unparsable": 0}
    for r in rows:
        slot = ver if verifier and r["model_id"] == verifier else primary
        slot["done"] += r["n"] or 0
        slot["correct"] += r["correct"] or 0
        slot["unparsable"] += r["unparsable"] or 0
    for slot in (primary, ver):
        slot["wrong"] = slot["done"] - slot["correct"] - slot["unparsable"]
    return {
        "run_id": run_id,
        "status": row.get("status"),
        "phase": row.get("phase"),
        "error": row.get("error"),
        "models": cfg.get("models") or [],
        "verifier": verifier,
        "planned_primary": row.get("planned_primary"),
        "planned_verifier": row.get("planned_verifier"),
        "primary": primary,
        "verifier_counts": ver,
    }


def zero_correct_questions(run_id: str, model_ids: list[str]) -> list[str]:
    """Questions in this run where every sample from *model_ids* was wrong."""
    if not model_ids:
        return []
    ph = ",".join(f":m{i}" for i in range(len(model_ids)))
    params: dict = {"r": run_id, **{f"m{i}": m for i, m in enumerate(model_ids)}}
    with get_engine().connect() as conn:
        ids = conn.execute(
            text(
                f"""
                SELECT question_id FROM attempts
                WHERE run_id=:r AND model_id IN ({ph})
                GROUP BY question_id HAVING SUM(is_correct) = 0
                """
            ),
            params,
        ).scalars().all()
    return sorted(set(ids))
