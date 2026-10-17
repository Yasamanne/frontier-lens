from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from frontier_lens import db, eval_job, runlog, runstate
from frontier_lens.providers.base import Completion

POLICY = "gemma4-26b-a4b"


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(runlog, "LOGS_DIR", tmp_path)
    engine = db.get_engine()
    yield engine
    engine.dispose()


def _add_question(engine, qid: str, key: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO questions (
                    id, topic, level, band, split, template, params_json, stem, code,
                    lang, input_files_json, key, numeric, source
                ) VALUES (
                    :id, 'custom', 2, 'standard', 'train', 'custom', '{}', :stem, '',
                    'python', '{}', :key, 1, 'uploaded'
                )
                """
            ),
            {"id": qid, "stem": f"stem {qid}", "key": key},
        )


def _add_attempt(engine, run_id: str, qid: str, sample: int, correct: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO attempts (run_id, model_id, question_id, sample_index,
                                      raw_output, parsed_answer, is_correct, unparsable)
                VALUES (:r, :m, :q, :s, 'FINAL: 0', '0', :ok, 0)
                """
            ),
            {"r": run_id, "m": POLICY, "q": qid, "s": sample, "ok": correct},
        )


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_reap_stale_marks_dead_runs_interrupted(tmp_db):
    now = datetime.now(timezone.utc)
    cfg = {"models": [POLICY], "k": 1, "question_ids": ["q"]}
    for rid, status in [("dead", "running"), ("alive", "running"), ("done", "complete")]:
        runstate.insert_run(rid, ["q"], 1, None, cfg, status=status)
    runstate.set_fields("dead", heartbeat_at=_iso(now - timedelta(seconds=runstate.STALE_SEC + 5)))
    runstate.set_fields("alive", heartbeat_at=_iso(now))
    runstate.set_fields("done", heartbeat_at=_iso(now - timedelta(days=1)))

    assert runstate.reap_stale(now) == ["dead"]
    assert runstate.load_run("dead")["status"] == "interrupted"
    assert runstate.load_run("alive")["status"] == "running"
    assert runstate.load_run("done")["status"] == "complete"
    assert [r["id"] for r in runstate.active_runs()] == ["alive"]
    assert runstate.is_resumable(runstate.load_run("dead"))


def test_reap_stale_handles_runs_without_heartbeat(tmp_db):
    with tmp_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO eval_runs (id, created_at, item_set_hash, status, k) "
                "VALUES ('legacy', '2026-01-01T00:00:00+00:00', 'h', 'running', 8)"
            )
        )
    assert runstate.reap_stale() == ["legacy"]
    # No stored config: shown as interrupted but not offered for resume.
    assert not runstate.is_resumable(runstate.load_run("legacy"))


def test_heartbeat_updates_pid_and_timestamp(tmp_db):
    runstate.insert_run("hb", ["q"], 1, None, {}, status="queued")
    assert runstate.claim("hb")
    runstate.set_fields("hb", heartbeat_at=None)
    with runstate.heartbeat("hb", interval=0.05):
        pass
    row = runstate.load_run("hb")
    assert row["heartbeat_at"] is not None
    assert row["pid"] is not None


def test_resume_only_samples_missing_triples(tmp_db, monkeypatch):
    from frontier_lens import contest

    _add_question(tmp_db, "q1", "4")
    _add_question(tmp_db, "q2", "5")
    calls: list[str] = []

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def complete(self, prompt, model, **_kwargs):
            calls.append(prompt)
            return Completion(text="FINAL: 4", latency_ms=1, tokens_in=None, tokens_out=None)

    monkeypatch.setattr(contest, "OpenAICompatibleProvider", FakeProvider)

    run_id = eval_job.create(contest.run_config([POLICY], 2, ["q1", "q2"], notes="test"))
    assert runstate.load_run(run_id)["status"] == "queued"
    # One sample survived an earlier, interrupted attempt.
    _add_attempt(tmp_db, run_id, "q1", 1, correct=1)
    runstate.set_fields(run_id, status="queued")  # what launch()/the CLI do on resume

    eval_job.run_job(run_id)

    assert len(calls) == 3  # 4 planned, 1 already stored
    row = runstate.load_run(run_id)
    assert row["status"] == "complete"
    assert row["phase"] == "done"
    assert row["planned_primary"] == 4
    p = runstate.progress(run_id)
    assert p["primary"]["done"] == 4
    assert p["primary"]["correct"] == 2  # q2's key is 5; the fake always answers 4
    assert runstate.zero_correct_questions(run_id, [POLICY]) == ["q2"]
    with tmp_db.connect() as conn:
        flagged = conn.execute(
            text("SELECT id FROM questions WHERE sme_flagged=1")
        ).scalars().all()
    assert flagged == ["q2"]


def test_launch_refuses_a_run_that_is_already_running(tmp_db):
    runstate.insert_run("busy", ["q"], 1, None, {"models": [POLICY]}, status="running")
    runstate.set_fields("busy", pid=12345, heartbeat_at=runstate.now_iso())
    with pytest.raises(RuntimeError, match="already running"):
        eval_job.launch("busy")


def test_setup_error_marks_run_failed(tmp_db):
    from frontier_lens.contest import run_config

    # The stored question no longer exists, so run_contest fails before sampling.
    run_id = eval_job.create(run_config([POLICY], 1, ["missing"], notes="test"))
    with pytest.raises(RuntimeError, match="no questions"):
        eval_job.run_job(run_id)
    row = runstate.load_run(run_id)
    assert row["status"] == "failed"
    assert "no questions" in row["error"]


def test_final_status_is_not_overwritten(tmp_db):
    from frontier_lens.contest import mark_complete, mark_interrupted

    runstate.insert_run("fin", ["q"], 1, None, {}, status="running")
    mark_complete("fin")
    mark_interrupted("fin", "stopped by user")  # e.g. Stop clicked as it finished
    row = runstate.load_run("fin")
    assert row["status"] == "complete"
    assert row["error"] is None


def test_ownership_lost_when_stopped_or_taken_over():
    row = {"status": "running", "pid": 100}
    assert runstate.owned_by(row, 100)
    assert not runstate.owned_by({"status": "interrupted", "pid": 100}, 100)
    assert not runstate.owned_by({"status": "running", "pid": 200}, 100)
    # A resume launch clears the pid; an old process must not take it back.
    assert not runstate.owned_by({"status": "queued", "pid": None}, 100)
    assert not runstate.owned_by(None, 100)


def test_claim_fails_after_stop(tmp_db):
    runstate.insert_run("st", ["q"], 1, None, {}, status="queued")
    runstate.finish("st", "interrupted", error="stopped by user")
    assert not runstate.claim("st")
    assert runstate.load_run("st")["status"] == "interrupted"


def test_heartbeat_stops_quietly_after_own_finish(tmp_db):
    import time

    runstate.insert_run("own", ["q"], 1, None, {}, status="queued")
    assert runstate.claim("own")
    # If the heartbeat treated our own finish as a takeover it would os._exit
    # and take pytest down with it.
    with runstate.heartbeat("own", interval=0.02):
        runstate.finish("own", "complete")
        time.sleep(0.1)
    assert runstate.load_run("own")["status"] == "complete"
