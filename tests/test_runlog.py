import json

from frontier_lens.runlog import jsonl_path, text_path, write_event


def test_write_event_records_prompt_output_and_key(tmp_path, monkeypatch):
    from frontier_lens import runlog

    monkeypatch.setattr(runlog, "LOGS_DIR", tmp_path)
    write_event(
        "testrun",
        "contest_sample",
        model_id="gemma4-26b-a4b",
        question_id="abc",
        sample_index=1,
        template="append_len",
        gold_key="4",
        prompt="INPUT",
        raw_output="FINAL: 4",
        is_correct=True,
    )
    rows = [json.loads(line) for line in jsonl_path("testrun").read_text().splitlines()]
    assert rows[0]["gold_key"] == "4"
    assert rows[0]["prompt"] == "INPUT"
    assert "FINAL: 4" in rows[0]["raw_output"]
    text = text_path("testrun").read_text()
    assert "gold_key (sandbox): 4" in text
    assert "INPUT" in text
    write_event("testrun", "run_end", status="failed", error="NotFoundError: no endpoints")
    text = text_path("testrun").read_text()
    assert "status=failed" in text
    assert "error: NotFoundError: no endpoints" in text


def test_contest_status_line_is_flushed(capsys):
    from frontier_lens.contest import _log_status

    _log_status("abc", "failed", error="NotFoundError: no endpoints")
    line = capsys.readouterr().out.strip()
    assert line.startswith("run abc status=failed")
    assert "error=NotFoundError: no endpoints" in line
