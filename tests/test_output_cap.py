from sqlalchemy import text

from frontier_lens import contest
from frontier_lens.db import get_engine
from frontier_lens.providers import openai_compatible as oc
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import load_yaml, sync_models
from frontier_lens.reward import grade, last_final

# ── exact_match_v2: FINAL is matched case-insensitively ─────────────────

def test_final_marker_is_case_insensitive():
    assert last_final("blah\nFinal: 2530") == "2530"
    assert last_final("blah\nfinal: 7\n") == "7"
    assert grade("The answer is solid.  Final: 2530", "2530") == (1, "2530", False)
    assert grade("FINAL: 5", "5") == (1, "5", False)  # the strict form still works


def test_repeated_marker_on_one_line_is_a_loop_not_an_answer():
    loop = "Final: 6      Final: 6      Final: 6      Final: 6"
    assert last_final(loop) is None
    assert grade(loop, "6") == (0, None, True)


def test_no_marker_stays_unparsable_and_hedges_still_score_zero():
    assert grade("the answer is 4, I think", "4") == (0, None, True)
    assert grade("final: 3 or 4", "3")[0] == 0


# ── provider: output cap and correction turn ────────────────────────────

class _Usage:
    def __init__(self, n):
        self.completion_tokens, self.prompt_tokens = n, 10


def _provider(monkeypatch, replies):
    """Provider whose _call returns scripted (text, completion_tokens) and records kwargs."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    provider = OpenAICompatibleProvider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    calls: list[dict] = []
    queue = list(replies)

    def fake_call(messages, model, extra_body=None, response_format=None,
                  temperature=None, max_tokens=None):
        calls.append({"n_messages": len(messages), "max_tokens": max_tokens})
        reply, tokens = queue.pop(0)
        return reply, _Usage(tokens)

    monkeypatch.setattr(provider, "_call", fake_call)
    return provider, calls


def test_max_tokens_is_forwarded_and_a_clean_answer_needs_no_correction(monkeypatch):
    provider, calls = _provider(monkeypatch, [("work\nFINAL: 4", 50)])
    done = provider.complete("p", "m", max_tokens=16000)
    assert calls == [{"n_messages": 1, "max_tokens": 16000}]
    assert done.truncated is False


def test_lowercase_final_counts_so_no_correction_turn(monkeypatch):
    provider, calls = _provider(monkeypatch, [("work\nFinal: 4", 50)])
    provider.complete("p", "m", max_tokens=16000)
    assert len(calls) == 1


def test_missing_final_triggers_one_short_correction_turn(monkeypatch):
    provider, calls = _provider(monkeypatch, [("rambling", 80), ("FINAL: 4", 5)])
    done = provider.complete("p", "m", max_tokens=16000)
    assert [c["n_messages"] for c in calls] == [1, 3]
    assert calls[1]["max_tokens"] == oc.CORRECTION_MAX_TOKENS
    assert "[correction]" in done.text and last_final(done.text) == "4"


def test_a_reply_that_used_the_whole_budget_is_marked_and_not_retried(monkeypatch):
    provider, calls = _provider(monkeypatch, [("cut off mid-sent", 16000)])
    done = provider.complete("p", "m", max_tokens=16000)
    assert len(calls) == 1
    assert done.truncated is True


def test_no_cap_means_no_truncation_flag(monkeypatch):
    provider, calls = _provider(monkeypatch, [("rambling", 99999), ("FINAL: 1", 5)])
    done = provider.complete("p", "m")
    assert calls[0]["max_tokens"] is None and done.truncated is False
    assert len(calls) == 2  # still gets its correction turn


# ── config: the cap is per model ────────────────────────────────────────

def test_only_the_policy_has_an_output_cap_in_models_yaml():
    caps = {m.id: m.max_tokens for m in load_yaml()}
    assert caps["gemma4-26b-a4b"] == 16000
    assert all(v is None for k, v in caps.items() if k != "gemma4-26b-a4b")


def test_contest_sampling_passes_the_models_cap(monkeypatch):
    seen: list[dict] = []

    class FakeProvider:
        def __init__(self, *a, **k):
            pass

        def complete(self, prompt, model, **kw):
            seen.append({"model": model, **kw})
            return contest.Completion(text="FINAL: 4", latency_ms=1, tokens_in=1, tokens_out=1)

    monkeypatch.setattr(contest, "OpenAICompatibleProvider", FakeProvider)
    sync_models()
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO questions (id, topic, level, band, split, template, params_json, stem,
                    code, lang, input_files_json, key, numeric, source)
                VALUES ('q1', 't', 2, 'standard', 'train', 't', '{}', 'what is 2+2', '', 'python',
                        '{}', '4', 1, 'seed')
                """
            )
        )
    contest.run_contest(["gemma4-26b-a4b", "gemma4-31b"], k=1, question_ids=["q1"])
    caps = {c["model"]: c["max_tokens"] for c in seen}
    assert caps == {"google/gemma-4-26b-a4b-it": 16000, "google/gemma-4-31b-it": None}
