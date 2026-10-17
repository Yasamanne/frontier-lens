import json

import httpx2
import pytest

from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider


def test_complete_omits_tools_so_openrouter_keeps_non_tool_providers(monkeypatch):
    captured: dict[str, str] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = request.content.decode()
        return httpx2.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "FINAL: 1"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(
        "frontier_lens.providers.openai_compatible.OpenAI",
        lambda **kwargs: __import__("openai").OpenAI(
            **{**kwargs, "http_client": httpx2.Client(transport=httpx2.MockTransport(handler))}
        ),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    completion = OpenAICompatibleProvider(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    ).complete("hi", "meta-llama/llama-4-scout")

    body = json.loads(captured["body"])
    assert "tools" not in body
    assert body["model"] == "meta-llama/llama-4-scout"
    assert completion.text == "FINAL: 1"


def _install_client(provider: OpenAICompatibleProvider, create) -> None:
    class _Chat:
        completions = type("Completions", (), {"create": staticmethod(create)})()

    provider.client = type("Client", (), {"chat": _Chat()})()


def test_complete_retries_a_non_json_body(monkeypatch):
    monkeypatch.setattr("frontier_lens.providers.openai_compatible.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def create(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise json.JSONDecodeError("Expecting value", "not-json", 0)
        message = type("Message", (), {"content": "FINAL: 7"})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice], "usage": None})()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenAICompatibleProvider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    _install_client(provider, create)
    completion = provider.complete("hi", "m")
    assert calls["n"] == 3
    assert completion.text == "FINAL: 7"


def test_complete_does_not_retry_a_local_error(monkeypatch):
    slept = {"n": 0}
    monkeypatch.setattr(
        "frontier_lens.providers.openai_compatible.time.sleep",
        lambda _s: slept.__setitem__("n", slept["n"] + 1),
    )
    calls = {"n": 0}

    def create(**_kwargs):
        calls["n"] += 1
        raise RuntimeError("bad request shape")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenAICompatibleProvider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    _install_client(provider, create)
    with pytest.raises(RuntimeError, match="bad request shape"):
        provider.complete("hi", "m")
    assert calls["n"] == 1
    assert slept["n"] == 0


def test_replay_gap_names_the_missing_question(tmp_path, monkeypatch):
    path = tmp_path / "recorded_run.json"
    path.write_text(json.dumps({"rows": []}))
    monkeypatch.setattr("frontier_lens.providers.fixture.FIXTURES_PATH", path)
    monkeypatch.setattr(
        "frontier_lens.contest.select_questions",
        lambda levels=None, limit=None, question_ids=None: [{"id": "q-missing", "level": 3}],
    )
    from frontier_lens.contest import replay_gap

    message = replay_gap(["gemma4-26b-a4b"], [3], limit=1)
    assert message is not None
    assert "q-missing" in message
    assert "live" in message
