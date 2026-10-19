import pytest

from frontier_lens import db, jobs, runlog, runstate


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(runlog, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "LOGS_DIR", tmp_path)
    engine = db.get_engine()
    yield engine
    engine.dispose()


def _rescue_result(qid: str, rescued: bool) -> dict:
    return {
        "question_id": qid, "topic": "t", "band": "b", "stem_original": "s", "gold_key": "1",
        "original_responses": [], "hint_rounds": [], "candidate_id": "c" if rescued else None,
        "pass_rate": 0.5 if rescued else 0.0, "rl_suitable": rescued, "rescue_level": 1 if rescued else None,
        "hint": "",
    }


def test_rescue_resume_skips_finished_questions(tmp_db, monkeypatch, capsys):
    from frontier_lens import generate

    calls: list[list[str]] = []

    def fake_rescue(run_id, model_id, question_ids, *, log_fn, calibration_k, max_hints):
        calls.append(list(question_ids))
        for qid in question_ids:
            log_fn("rescue_question", question_id=qid, index=1, total=len(question_ids))
            if qid == "q-skip":
                log_fn("rescue_question_skip", question_id=qid, reason="no_verifier_code")
                continue
            log_fn("rescue_question_done", question_id=qid, result=_rescue_result(qid, qid == "q2"))
        return []

    monkeypatch.setattr(generate, "rescue_with_hints", fake_rescue)
    cfg = {"run_id": "r", "model_id": "m", "question_ids": ["q1", "q2", "q-skip"],
           "calibration_k": 2, "max_hints": 3}
    job_id = jobs.create("rescue", cfg, planned=3)
    # q1 finished before the job was interrupted.
    res = jobs._Results(job_id, jobs.load(job_id))
    res.add(_rescue_result("q1", False))
    runstate.set_fields(job_id, table=jobs.TABLE, status="queued")  # as launch() does on resume

    jobs.run(job_id)

    assert calls == [["q2", "q-skip"]]
    row = jobs.load(job_id)
    assert row["status"] == "complete"
    assert row["done"] == 3
    saved = jobs.results(row)
    assert [r["question_id"] for r in saved["items"]] == ["q1", "q2"]
    assert saved["skipped"] == [{"question_id": "q-skip", "reason": "no_verifier_code"}]
    # Formatted lines are tagged with their question (stdout is the job log).
    assert "[q2] " in capsys.readouterr().out

    # Nothing left: a second run does no work.
    runstate.set_fields(job_id, table=jobs.TABLE, status="queued")
    jobs.run(job_id)
    assert len(calls) == 1


def test_augment_resume_tops_up_each_seed(tmp_db, monkeypatch):
    from frontier_lens import generate

    calls: list[tuple[int, list[str]]] = []

    def fake_similar(seeds, *, run_id, n_each, log_fn, calibration_k, max_hint_levels=3):
        calls.append((n_each, [s["id"] for s in seeds]))
        for s in seeds:
            for _ in range(n_each):
                log_fn("similar_result", source_id=s["id"], candidate_id="c",
                       status="pending", pass_rate=0.5)
        return []

    monkeypatch.setattr(generate, "generate_similar_questions", fake_similar)
    seeds = [{"id": i, "stem": "s", "key": "1", "topic": "t", "band": "b"} for i in ("A", "B", "C")]
    cfg = {"run_id": "r", "mode": "one_shot", "seeds": seeds, "n_each": 2,
           "calibration_k": 2, "max_loops": 3, "max_fix_iterations": 3}
    job_id = jobs.create("augment", cfg, planned=6)
    res = jobs._Results(job_id, jobs.load(job_id))
    for source in ("A", "A", "B"):
        res.add({"source_id": source, "candidate_id": "old", "status": "pending", "pass_rate": 0.5})

    jobs.run(job_id)

    assert calls == [(1, ["B"]), (2, ["C"])]
    row = jobs.load(job_id)
    assert row["status"] == "complete"
    assert row["done"] == 6


def test_calibrate_saves_each_topic_and_resumes(tmp_db, monkeypatch):
    from frontier_lens import calibrate, registry

    measured: list[str] = []
    written: list[dict] = []
    monkeypatch.setattr(calibrate, "measure_rates", lambda topic, k: measured.append(topic) or [0.5])
    monkeypatch.setattr(calibrate, "write_knobs", lambda rates, **kw: written.append(rates))
    monkeypatch.setattr(calibrate, "print_knob_report", lambda: None)

    class Policy:
        id, model = "p", "p-model"

    monkeypatch.setattr(registry, "policy_spec", lambda: Policy)
    topics = list(calibrate.GRIDS)[:3]
    job_id = jobs.create("calibrate", {"topics": topics, "k": 2}, planned=3)
    jobs._Results(job_id, jobs.load(job_id)).add({"topic": topics[0], "rates": [0.1]})

    jobs.run(job_id)

    assert measured == topics[1:]
    # One write with every topic, including the one measured before the resume.
    assert written == [{topics[0]: [0.1], topics[1]: [0.5], topics[2]: [0.5]}]
    assert jobs.load(job_id)["status"] == "complete"


def test_failure_is_recorded_and_resumable(tmp_db, monkeypatch):
    from frontier_lens import generate

    def boom(*_args, **_kwargs):
        raise RuntimeError("No generator model configured in models.yaml.")

    monkeypatch.setattr(generate, "rescue_with_hints", boom)
    job_id = jobs.create("rescue", {"run_id": "r", "model_id": "m", "question_ids": ["q1"],
                                    "calibration_k": 2, "max_hints": 3}, planned=1)
    with pytest.raises(RuntimeError):
        jobs.run(job_id)
    row = jobs.load(job_id)
    assert row["status"] == "failed"
    assert "No generator model" in row["error"]
    assert runstate.is_resumable(row)


def test_active_recent_and_stop(tmp_db):
    a = jobs.create("seed", {"force": False}, planned=1)
    b = jobs.create("rescue", {}, planned=1)
    assert {r["id"] for r in jobs.active()} == {a, b}
    assert [r["id"] for r in jobs.active(("seed",))] == [a]
    jobs.stop(a)
    row = jobs.load(a)
    assert row["status"] == "interrupted"
    assert row["error"] == "stopped by user"
    assert [r["id"] for r in jobs.recent("seed")] == [a]
    with pytest.raises(ValueError):
        jobs.create("nope", {}, planned=1)


def test_stopped_calibration_writes_no_knobs(tmp_db, monkeypatch):
    from frontier_lens import calibrate, registry

    written: list[dict] = []

    def measure(topic, k):
        if topic == "boom":
            raise RuntimeError("provider down")
        return [0.5]

    monkeypatch.setattr(calibrate, "measure_rates", measure)
    monkeypatch.setattr(calibrate, "write_knobs", lambda rates, **kw: written.append(rates))
    monkeypatch.setattr(calibrate, "GRIDS", {**calibrate.GRIDS, "boom": []})

    class Policy:
        id, model = "p", "p-model"

    monkeypatch.setattr(registry, "policy_spec", lambda: Policy)
    first = next(iter(calibrate.GRIDS))
    job_id = jobs.create("calibrate", {"topics": [first, "boom"], "k": 2}, planned=2)
    with pytest.raises(RuntimeError):
        jobs.run(job_id)
    assert written == []  # the knobs file never claims unmeasured topics
    assert [r["topic"] for r in jobs.results(jobs.load(job_id))["items"]] == [first]


def test_run_does_nothing_if_stopped_before_start(tmp_db, monkeypatch):
    from frontier_lens import generate

    monkeypatch.setattr(generate, "rescue_with_hints", lambda *a, **k: pytest.fail("ran"))
    job_id = jobs.create("rescue", {"run_id": "r", "model_id": "m", "question_ids": ["q1"],
                                    "calibration_k": 2, "max_hints": 3}, planned=1)
    jobs.stop(job_id)
    jobs.run(job_id)
    assert jobs.load(job_id)["status"] == "interrupted"
