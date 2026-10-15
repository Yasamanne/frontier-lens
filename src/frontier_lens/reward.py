"""Binary exact-match reward. Id is stable for RL exports."""

from __future__ import annotations

import re
from collections import defaultdict

from frontier_lens.parse import FINAL_RE, normalize

REWARD_FN = "exact_match_v1"
HEDGE_RE = re.compile(r"\b(or|either)\b|/", re.IGNORECASE)


def last_final(raw: str) -> str | None:
    if not raw:
        return None
    matches = FINAL_RE.findall(raw)
    if not matches:
        return None
    return matches[-1].strip()


def is_degenerate(answer: str) -> bool:
    text = answer.strip()
    if not text:
        return True
    if HEDGE_RE.search(text):
        return True
    if "," in text or " and " in text.lower():
        return True
    return False


def grade(raw: str, key: str, numeric: bool = True) -> tuple[int, str | None, bool]:
    """Return (reward, parsed_answer, unparsable). Only the last FINAL: line counts."""
    ans = last_final(raw)
    if ans is None:
        return 0, None, True
    if is_degenerate(ans):
        return 0, ans, False
    got = normalize(ans, numeric)
    gold = normalize(key, numeric)
    return (1 if got == gold else 0), got, False


def answer_space_size(keys: list[str], numeric: bool = True) -> int:
    return len({normalize(k, numeric) for k in keys})


def guessable_templates(template_keys: dict[str, list[str]], small: int = 12) -> list[str]:
    """Flag templates whose gold answers look like a small integer range."""
    flagged = []
    for template, keys in template_keys.items():
        norms = [normalize(k, True) for k in keys]
        ints: list[int] = []
        ok = True
        for n in norms:
            try:
                ints.append(int(n))
            except ValueError:
                ok = False
                break
        if not ok or not ints:
            continue
        span = max(ints) - min(ints) + 1
        if span <= small and len(set(ints)) <= small:
            flagged.append(template)
    return flagged
