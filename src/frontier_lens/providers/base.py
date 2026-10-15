from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    latency_ms: int
    tokens_in: int | None
    tokens_out: int | None


class Provider(Protocol):
    def complete(self, prompt: str, model: str) -> Completion: ...
