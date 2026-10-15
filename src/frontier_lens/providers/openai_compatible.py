from __future__ import annotations

import os
import time

from openai import OpenAI

from frontier_lens.providers.base import Completion


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key_env: str | None = None) -> None:
        key = "ollama"
        if api_key_env:
            key = os.getenv(api_key_env) or ""
            if not key:
                raise RuntimeError(f"missing env {api_key_env}")
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=120)

    def complete(self, prompt: str, model: str) -> Completion:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            extra_body={"tools": None},
        )
        ms = int((time.perf_counter() - t0) * 1000)
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return Completion(
            text=choice,
            latency_ms=ms,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
        )
