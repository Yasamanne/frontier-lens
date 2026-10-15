from __future__ import annotations

from frontier_lens.parse import Parsed, normalize


def is_correct(parsed: Parsed, key: str, numeric: bool = True) -> bool:
    if parsed.unparsable or parsed.answer is None:
        return False
    return parsed.answer == normalize(key, numeric)
