from __future__ import annotations

import json
import os
import time
import typing

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from frontier_lens.parallel import call_slot
from frontier_lens.providers.base import Completion

RETRY_ATTEMPTS = 4
_RETRY_SLEEP_SEC = (1.0, 2.0, 4.0)


def _retryable(exc: BaseException) -> bool:
    """Transient gateway and transport failures. A 4xx auth error is not one of these."""
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key_env: str | None = None) -> None:
        key = "ollama"
        if api_key_env:
            key = os.getenv(api_key_env) or ""
            if not key:
                raise RuntimeError(f"missing env {api_key_env}")
        headers = {}
        if "openrouter.ai" in (base_url or ""):
            headers = {
                "HTTP-Referer": "https://github.com/Yasamanne/frontier-lens",
                "X-Title": "Frontier Lens",
            }
        self.client = OpenAI(
            base_url=base_url, api_key=key, timeout=120, default_headers=headers
        )

    def _call(
        self,
        messages: list[dict],
        model: str,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> tuple[str, object]:
        """Raw chat call with retry on transient errors. Returns (text, usage)."""
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # Do not send tools. OpenRouter treats "tools": null as a tool-calling
                # request and drops providers that do not support tools. For
                # meta-llama/llama-4-scout that leaves only a BYOK Google endpoint.
                kwargs: dict = {"model": model, "messages": messages}
                if extra_body:
                    kwargs["extra_body"] = extra_body
                if response_format:
                    kwargs["response_format"] = response_format
                with call_slot():
                    resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or "", resp.usage
            except Exception as exc:
                if not _retryable(exc) or attempt + 1 == RETRY_ATTEMPTS:
                    raise
                time.sleep(_RETRY_SLEEP_SEC[min(attempt, len(_RETRY_SLEEP_SEC) - 1)])
        return "", None  # unreachable

    # JSON schema used when json_mode=True on the model spec.
    # The model must produce {"reasoning": "...", "answer": "..."}.
    _JSON_SCHEMA: typing.ClassVar[dict] = {
        "type": "json_schema",
        "json_schema": {
            "name": "contest_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["reasoning", "answer"],
                "additionalProperties": False,
            },
        },
    }

    def complete(
        self,
        prompt: str,
        model: str,
        *,
        thinking: bool = True,
        skip_correction: bool = False,
        json_mode: bool = False,
        extra_body: dict | None = None,
    ) -> Completion:
        """Call the model once (with an optional FINAL: correction turn).

        Pass ``skip_correction=True`` for generator calls that produce JSON —
        the correction turn only makes sense for policy/contest responses that
        must end with a ``FINAL: <answer>`` line.
        """
        from frontier_lens.prompt import CORRECTION_PROMPT

        t0 = time.perf_counter()
        messages: list[dict] = [{"role": "user", "content": prompt}]

        # Build extra_body for models that support thinking control.
        # Qwen3 / DeepSeek on OpenRouter: enable_thinking=False disables
        # extended chain-of-thought, reducing latency on short tasks.
        extra: dict | None = None
        if not thinking:
            extra = {"enable_thinking": False}
        if extra_body:  # provider-specific switches, e.g. OpenRouter's {"reasoning": {...}}
            extra = {**(extra or {}), **extra_body}

        # JSON mode: request structured output so the model cannot forget FINAL:
        # The answer field maps directly to the FINAL: value.
        response_format = self._JSON_SCHEMA if json_mode else None

        text, usage = self._call(messages, model, extra_body=extra,
                                  response_format=response_format)

        # Strip residual <think>…</think> blocks that some models leak even when
        # thinking is disabled (DeepSeek, Qwen3).
        import json as _json
        import re as _re
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

        # JSON mode: convert {"reasoning": "...", "answer": "42"} → FINAL: 42
        if json_mode:
            try:
                parsed = _json.loads(text)
                answer = parsed.get("answer", "")
                reasoning = parsed.get("reasoning", "")
                # Reconstruct as standard reasoning + FINAL: format
                text = f"{reasoning}\nFINAL: {answer}" if reasoning else f"FINAL: {answer}"
            except _json.JSONDecodeError:
                # JSON parsing failed — fall through to correction turn
                pass

        # If the response is missing the FINAL: line, send one correction turn.
        # Only applies to policy/contest calls — skip for generator JSON output.
        if not skip_correction and "FINAL:" not in text and "final:" not in text.lower():
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
                {"role": "user", "content": CORRECTION_PROMPT},
            ]
            try:
                correction_text, correction_usage = self._call(messages, model, extra_body=extra,
                                                                response_format=response_format)
                correction_text = _re.sub(
                    r"<think>.*?</think>", "", correction_text, flags=_re.DOTALL
                ).strip()
                # Append the correction so the full trace is preserved in raw_output
                text = text + "\n[correction]\n" + correction_text
                # Merge token counts
                if correction_usage and usage:
                    try:
                        usage = type("Usage", (), {
                            "prompt_tokens": (
                                getattr(usage, "prompt_tokens", 0) or 0
                            ) + (getattr(correction_usage, "prompt_tokens", 0) or 0),
                            "completion_tokens": (
                                getattr(usage, "completion_tokens", 0) or 0
                            ) + (getattr(correction_usage, "completion_tokens", 0) or 0),
                        })()
                    except Exception:
                        pass
            except Exception:
                pass  # if correction fails, fall back to original text

        ms = int((time.perf_counter() - t0) * 1000)
        return Completion(
            text=text,
            latency_ms=ms,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
        )
