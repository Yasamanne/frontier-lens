import json

import pytest
from sqlalchemy import text

from frontier_lens import db, feedback
from frontier_lens.providers.base import Completion


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(db, "_engine", None)
    engine = db.get_engine()
    yield engine
    engine.dispose()


def _candidate(engine, cid, *, topic="python_state", band="hard", stem="stem",
               source="seed1", mode="similar_augment", code="print(4)"):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO candidates (
                    id, topic, band, stem, code, lang, input_files_json, executed_key,
                    status, generator_model, generator_mode, source_fail_ids, created_at
                ) VALUES (
                    :id, :topic, :band, :stem, :code, 'python', '{}', '4',
                    'pending', 'gen', :mode, :src, '2026-10-17T00:00:00+00:00'
                )
                """
            ),
            {"id": cid, "topic": topic, "band": band, "stem": stem, "code": code,
             "mode": mode, "src": json.dumps([source])},
        )


def _review(engine, cid, label, *, reasons=(), note="", kind="sme_queue",
            original_stem=None, ts="2026-10-17T01:00:00+00:00"):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO reviews (candidate_id, label, note, reviewer, decided_at,
                                     kind, reasons, original_stem)
                VALUES (:c, :l, :n, 'sme', :ts, :k, :r, :o)
                """
            ),
            {"c": cid, "l": label, "n": note, "ts": ts, "k": kind,
             "r": json.dumps(list(reasons)), "o": original_stem},
        )


def test_no_feedback(tmp_db):
    assert feedback.context_for("python_state") == feedback.NO_FEEDBACK


def test_context_sections_and_exclusions(tmp_db):
    e = tmp_db
    _candidate(e, "same", stem="Which list is printed?", source="seed1")
    _review(e, "same", "rejected", reasons=["ambiguous"], note="Say which variable is printed")
    _candidate(e, "other", stem="Add 2 and 2", source="seed2")
    _review(e, "other", "trivial", reasons=["too_easy", "ambiguous"])
    _candidate(e, "edited", stem="Final clear stem", source="seed3")
    _review(e, "edited", "hard_item", reasons=["ambiguous"], original_stem="Vague stem")
    _candidate(e, "liked", stem="Nice multi-step state", source="seed4")
    _review(e, "liked", "hard_item", note="Good: three aliasing steps")
    _candidate(e, "auto", stem="Auto promoted", source="seed5")
    _review(e, "auto", "hard_item", note="batch accepted by audit", kind="audit_auto")
    _candidate(e, "elsewhere", topic="shell_pipelines", stem="Other topic", source="seed6")
    _review(e, "elsewhere", "rejected", reasons=["wrong_key"], note="other topic note")
    _candidate(e, "hint", stem="Q. Hint: the answer is 4", source="seed1", mode="light_hint")
    _review(e, "hint", "rejected", reasons=["answer_leak"], note="hint gives it away")

    ctx = feedback.context_for("python_state", source_id="seed1")

    assert "Rejected variants of THIS seed" in ctx
    assert "Say which variable is printed" in ctx
    assert f"{feedback.REASONS['ambiguous']} (×2)" in ctx  # most common reason
    assert "Recent rejections for topic python_state" in ctx
    assert "label trivial" in ctx and "Add 2 and 2" in ctx
    assert '"Vague stem" → "Final clear stem" (fixed: ambiguous or underspecified stem)' in ctx
    assert "Good: three aliasing steps" in ctx
    # Excluded: automatic promotions, other topics, hint feedback.
    assert "batch accepted by audit" not in ctx
    assert "other topic note" not in ctx
    assert "hint gives it away" not in ctx
    # Same-seed rejection comes first.
    assert ctx.index("THIS seed") < ctx.index("Recent rejections")

    hints = feedback.context_for("python_state", hints=True)
    assert "hint gives it away" in hints
    assert "Say which variable is printed" not in hints


def test_context_respects_max_chars(tmp_db):
    for i in range(30):
        _candidate(tmp_db, f"c{i}", stem=f"stem {i} " + "x" * 150, source=f"s{i}")
        _review(tmp_db, f"c{i}", "rejected", reasons=["ambiguous"], note="n" * 200)
    for limit in (120, 600, 2500):
        ctx = feedback.context_for("python_state", max_chars=limit)
        assert len(ctx) <= limit
        assert ctx != feedback.NO_FEEDBACK


def test_promote_records_reasons_and_original_stem(tmp_db):
    from frontier_lens.review import promote

    _candidate(tmp_db, "a", stem="Original stem")
    promote("a", "hard_item", fair=True, novel=True, hard=True, note="better now",
            kind="sme_queue", edited_stem="Edited stem", reasons=["ambiguous"])
    _candidate(tmp_db, "r", stem="Bad stem")
    promote("r", "rejected", fair=False, novel=True, hard=False, note="why",
            kind="sme_queue", reasons=["too_easy"])
    with tmp_db.connect() as conn:
        rows = {r["candidate_id"]: r for r in conn.execute(
            text("SELECT candidate_id, reasons, original_stem, note FROM reviews")
        ).mappings()}
    assert json.loads(rows["a"]["reasons"]) == ["ambiguous"]
    assert rows["a"]["original_stem"] == "Original stem"
    assert json.loads(rows["r"]["reasons"]) == ["too_easy"]
    assert rows["r"]["original_stem"] is None


def test_feedback_reaches_generator_and_reviewer_prompts(tmp_db, monkeypatch):
    from frontier_lens.generate import similar

    _candidate(tmp_db, "bad", stem="Old bad variant", source="seed1")
    _review(tmp_db, "bad", "rejected", reasons=["not_novel"], note="only numbers changed")
    prompts: list[str] = []

    class FakeProvider:
        def __init__(self, *_a, **_k):
            pass

        def complete(self, prompt, model, **_k):
            prompts.append(prompt)
            return Completion(text='{"score": 90, "passed": true}', latency_ms=1,
                              tokens_in=None, tokens_out=None)

    monkeypatch.setattr(similar, "OpenAICompatibleProvider", FakeProvider)
    monkeypatch.setattr(similar, "_store_candidate", lambda *a, **k: "new-cand")
    seed = {"id": "seed1", "stem": "Seed stem", "key": "4", "topic": "python_state", "band": "hard"}
    similar.generate_similar_questions([seed], run_id="r", n_each=1)
    assert "only numbers changed" in prompts[0]
    assert "HUMAN REVIEWER (SME) FEEDBACK" in prompts[0]

    prompts.clear()
    similar._review_similar_question(
        stem="New", code="print(4)", lang="python", computed_key="4",
        seed_stem="Seed stem", seed_key="4", band="hard", prior_stems=[],
        provider=FakeProvider(), model="judge",
        sme_feedback=feedback.context_for("python_state", source_id="seed1"),
    )
    assert "only numbers changed" in prompts[0]
    assert "SME feedback overrides your own taste" in prompts[0]


def test_approved_comments_and_accepted_issues(tmp_db):
    e = tmp_db
    _candidate(e, "despite", stem="Approved but wordy", source="s1")
    _review(e, "despite", "hard_item", reasons=["unnatural"])
    _candidate(e, "rewritten", stem="Clean final", source="s2")
    _review(e, "rewritten", "hard_item", note="tightened the wording", original_stem="Messy draft")

    ctx = feedback.context_for("python_state")

    assert '"Approved but wordy" — approved despite: contrived, unnatural or confusing wording' in ctx
    # An edited approval's note sits on its rewrite line and is not repeated.
    assert '"Messy draft" → "Clean final" — "tightened the wording"' in ctx
    assert ctx.count("tightened the wording") == 1


def test_react_candidates_are_linked_to_their_seed(tmp_db, monkeypatch):
    from types import SimpleNamespace as NS

    from frontier_lens.agent import tools
    from frontier_lens.generate import similar

    calls: list[dict] = []

    def fake_handle(name, args, **kwargs):
        calls.append({"name": name, **kwargs})
        return json.dumps({"candidate_id": "c-react"})

    submit = NS(id="t1", function=NS(name="submit_candidate", arguments=json.dumps(
        {"stem": "s", "lang": "python", "code": "print(4)"})))
    client = NS(chat=NS(completions=NS(
        create=lambda **_k: NS(choices=[NS(message=NS(tool_calls=[submit], content=None))])
    )))
    monkeypatch.setattr(tools, "handle", fake_handle)
    seed = {"id": "seed9", "stem": "Seed", "key": "4", "topic": "python_state", "band": "hard"}

    cid = similar._run_similar_agent(seed, NS(id="gen", model="m"), client, run_id="r")

    assert cid == "c-react"
    assert calls[0]["fail_model"] == "seed9"  # stored as the candidate's source_fail_ids
