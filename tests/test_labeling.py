import json

from sqlalchemy import text

from frontier_lens import labeling, upload
from frontier_lens.db import get_engine


def _row(stem, topic="custom", level=2, topic_missing=True, level_missing=True, key="4"):
    return {"stem": stem, "key": key, "topic": topic, "level": level, "numeric": True,
            "split": "train", "solution_steps": "", "id": stem,
            "topic_missing": topic_missing, "level_missing": level_missing}


def _reply(*labels):
    return json.dumps([{"id": i, "topic": t, "band": b} for i, (t, b) in enumerate(labels, 1)])


def test_clean_row_flags_missing_topic_and_difficulty():
    bare = upload._clean_row({"stem": "s", "key": "1"}, 1)
    assert (bare["topic_missing"], bare["level_missing"]) == (True, True)
    full = upload._clean_row({"stem": "s", "key": "1", "topic": "algebra", "difficulty": "hard"}, 1)
    assert (full["topic_missing"], full["level_missing"], full["level"]) == (False, False, 3)
    level_only = upload._clean_row({"stem": "s", "key": "1", "level": "3"}, 1)
    assert level_only["level_missing"] is False


def test_normalize_topic_and_parse_reject_junk():
    assert labeling.normalize_topic("Number Theory!") == "number_theory"
    assert labeling.normalize_topic("Other") is None
    assert labeling.normalize_topic("123abc") is None
    got = labeling.parse_labels(
        "```json\n" + _reply(("Algebra", "hard"), ("misc", "weird")) + "\n```", 2
    )
    assert got == {1: ("algebra", "hard"), 2: (None, None)}


def test_only_missing_fields_are_filled_and_band_maps_to_level():
    rows = [
        _row("a"),                                              # both missing
        _row("b", topic="geometry", topic_missing=False),       # keeps topic
        _row("c", level=3, level_missing=False),                # keeps level
        _row("d", topic="x", level=2, topic_missing=False, level_missing=False),  # untouched
    ]
    report = labeling.label_rows(
        rows, known_topics=[],
        complete=lambda p: _reply(("algebra", "hard"), ("calculus", "hard"), ("probability", "standard")),
    )
    assert [(r["topic"], r["level"]) for r in rows] == [
        ("algebra", 3), ("geometry", 3), ("probability", 3), ("x", 2)]
    assert rows[0]["labels"] == {"topic": "llm", "band": "llm"}
    assert rows[1]["labels"] == {"band": "llm"} and rows[2]["labels"] == {"topic": "llm"}
    assert "labels" not in rows[3]
    assert (report.rows_needing_labels, report.rows_labeled, report.failed_batches) == (3, 3, 0)


def test_failed_batch_keeps_defaults_and_is_reported():
    rows = [_row("a"), _row("b")]

    def boom(prompt):
        raise RuntimeError("provider down")

    report = labeling.label_rows(rows, known_topics=[], complete=boom)
    assert [(r["topic"], r["level"]) for r in rows] == [("custom", 2), ("custom", 2)]
    assert (report.rows_labeled, report.failed_batches) == (0, 1)


def test_later_batches_reuse_topics_created_by_the_first():
    rows = [_row(f"q{i}") for i in range(4)]
    prompts: list[str] = []

    def complete(prompt):
        prompts.append(prompt)
        n = prompt.count("[answer:")
        return _reply(*[("number_theory", "standard")] * n)

    report = labeling.label_rows(rows, known_topics=["algebra"], complete=complete, batch_size=2)
    assert report.batches == 2 and len(prompts) == 2
    assert "Existing topics: algebra" in prompts[0]
    assert "algebra, number_theory" in prompts[1]
    assert report.new_topics == ("number_theory",)


def test_labels_are_stored_with_the_question():
    rows = [_row("what is 2+2")]
    labeling.label_rows(rows, known_topics=[], complete=lambda p: _reply(("arithmetic", "standard")))
    assert upload.import_rows(rows, filename="f.csv") == 1
    with get_engine().connect() as conn:
        topic, band, template, params = conn.execute(
            text("SELECT topic, band, template, params_json FROM questions")
        ).one()
    assert (topic, band, template) == ("arithmetic", "standard", "uploaded_arithmetic")
    assert json.loads(params)["label_source"] == {"topic": "llm", "band": "llm"}
    assert upload.uploaded_topics() == ["arithmetic"]


def test_default_complete_turns_reasoning_off_for_the_judge(monkeypatch):
    from frontier_lens.providers import openai_compatible as oc
    from frontier_lens.providers.base import Completion

    seen: dict = {}

    class FakeProvider:
        def __init__(self, *a, **k):
            pass

        def complete(self, prompt, model, **kw):
            seen.update(kw)
            return Completion(text="[]", latency_ms=1, tokens_in=1, tokens_out=1)

    monkeypatch.setattr(oc, "OpenAICompatibleProvider", FakeProvider)
    assert labeling.default_complete()("hi") == "[]"
    assert seen["thinking"] is False
    assert seen["extra_body"] == {"reasoning": {"enabled": False}}
    assert seen["temperature"] == 0.0


def test_provider_merges_extra_body_with_thinking_flag(monkeypatch):
    from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    provider = OpenAICompatibleProvider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    sent: dict = {}

    def fake_call(messages, model, extra_body=None, response_format=None, temperature=None):
        sent["extra"] = extra_body
        return "ok", None

    monkeypatch.setattr(provider, "_call", fake_call)
    provider.complete("p", "m", thinking=False, skip_correction=True,
                      extra_body={"reasoning": {"enabled": False}})
    assert sent["extra"] == {"enable_thinking": False, "reasoning": {"enabled": False}}


def test_provider_sends_temperature_only_when_asked(monkeypatch):
    from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    provider = OpenAICompatibleProvider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    bodies: list[dict] = []

    class Resp:
        usage = None
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]

    monkeypatch.setattr(
        provider.client.chat.completions, "create", lambda **kw: bodies.append(kw) or Resp()
    )
    provider.complete("p", "m", skip_correction=True)
    provider.complete("p", "m", skip_correction=True, temperature=0.0)
    assert "temperature" not in bodies[0]
    assert bodies[1]["temperature"] == 0.0
