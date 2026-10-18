"""Binary exact-match reward with optional LLM-as-judge fallback.

Grading pipeline:
  1. Exact match after normalization (LaTeX-aware).  Fast, zero API calls.
  2. (optional) LLM judge — called only when exact match fails.
     Returns 1 if the judge says the answers are semantically equivalent.

Id is stable for RL exports.

exact_match_v2 (current) differs from v1 only in how the answer line is found:
``FINAL:`` is matched case-insensitively, so ``Final: 42`` counts. A line that repeats
the marker (``Final: 6  Final: 6 ...``, a generation loop) is not an answer.
Everything else (hedges score 0, normalisation, optional judge) is unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from frontier_lens.parse import normalize

REWARD_FN = "exact_match_v2"

# Case-insensitive, unlike parse.FINAL_RE (the v1 pattern).
FINAL_RE_V2 = re.compile(r"final:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MARKER_RE = re.compile(r"final\s*:", re.IGNORECASE)

_JUDGE_PROMPT = """\
You are a strict answer equivalence checker.
Decide if the MODEL ANSWER means the same thing as the GOLD ANSWER.

GOLD ANSWER: {gold}
MODEL ANSWER: {model}

Rules:
- "$B$" and "B" are equivalent (LaTeX vs plain).
- "Yes" and "yes" are equivalent (case).
- "Paris, France" and "Paris" are equivalent (extra context).
- "42" and "forty-two" are NOT equivalent (you must be strict about numbers).
- "A and C" and "A, C" are equivalent.
- If in doubt, answer NO.

Reply with exactly one word: YES or NO.
"""
HEDGE_RE = re.compile(r"\b(or|either)\b|/", re.IGNORECASE)


def last_final(raw: str) -> str | None:
    if not raw:
        return None
    matches = FINAL_RE_V2.findall(raw)
    if not matches:
        return None
    answer = matches[-1].strip()
    if _MARKER_RE.search(answer):  # marker repeated on one line: a loop, not an answer
        return None
    return answer


def is_degenerate(answer: str) -> bool:
    text = answer.strip()
    if not text:
        return True
    if HEDGE_RE.search(text):
        return True
    return "," in text or " and " in text.lower()


def judge_equivalence(
    model_answer: str,
    gold_key: str,
    judge_fn: Callable[[str], str],
) -> bool:
    """Call the LLM judge. Returns True if judge says the answers are equivalent.

    *judge_fn* takes a prompt string and returns the model's raw text response.
    Only call this when exact-match already failed.
    """
    prompt = _JUDGE_PROMPT.format(gold=gold_key.strip(), model=model_answer.strip())
    try:
        response = judge_fn(prompt)
        # Accept if the response contains YES (case-insensitive) and not NO
        response_upper = response.strip().upper()
        return response_upper.startswith("YES")
    except Exception:
        return False  # judge errors are treated as "not equivalent"


def grade(
    raw: str,
    key: str,
    numeric: bool = True,
    *,
    judge_fn: Callable[[str], str] | None = None,
) -> tuple[int, str | None, bool]:
    """Return (reward, parsed_answer, unparsable).

    Pipeline:
    1. Extract the last FINAL: line.
    2. Exact match (LaTeX-aware normalization).
    3. If *judge_fn* is provided and exact match failed: call LLM judge.
    """
    ans = last_final(raw)
    if ans is None:
        return 0, None, True
    if is_degenerate(ans):
        return 0, ans, False
    got = normalize(ans, numeric)
    gold = normalize(key, numeric)
    if got == gold:
        return 1, got, False
    # Exact match failed — try LLM judge if provided
    if judge_fn is not None:
        equivalent = judge_equivalence(ans, key, judge_fn)
        if equivalent:
            return 1, got, False
    return 0, got, False


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
