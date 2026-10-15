from __future__ import annotations

from frontier_lens.parse import Parsed
from frontier_lens.reward import grade


def is_correct(parsed: Parsed, key: str, numeric: bool = True) -> bool:
    """Kept for tests; prefer grade() on raw model output."""
    if parsed.unparsable or parsed.answer is None:
        return False
    reward, _, _ = grade(f"FINAL: {parsed.answer}", key, numeric=numeric)
    return reward == 1


def reward_from_raw(raw: str, key: str, numeric: bool = True) -> tuple[int, str | None, bool]:
    return grade(raw, key, numeric=numeric)
