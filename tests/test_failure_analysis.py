"""Unit tests for failure_analysis.py — no database required."""
from __future__ import annotations

from frontier_lens.failure_analysis import (
    _classify_question,
    _off_by_one,
    failure_summary_text,
)


def _attempt(is_correct=0, unparsable=0, parsed_answer=None):
    return {
        "is_correct": is_correct,
        "unparsable": unparsable,
        "parsed_answer": parsed_answer,
    }


# ── _off_by_one ──────────────────────────────────────────────────────────────

def test_off_by_one_true():
    assert _off_by_one("5", "4") is True
    assert _off_by_one("3", "4") is True


def test_off_by_one_false_exact():
    assert _off_by_one("4", "4") is False


def test_off_by_one_false_far():
    assert _off_by_one("10", "4") is False


def test_off_by_one_non_numeric():
    assert _off_by_one("foo", "bar") is False


# ── _classify_question ───────────────────────────────────────────────────────

def test_classify_all_unparsable():
    attempts = [_attempt(is_correct=0, unparsable=1) for _ in range(8)]
    r = _classify_question(attempts, gold="42", numeric=True)
    assert r["unparsable_frac"] == 1.0
    assert r["off_by_one_frac"] == 0.0


def test_classify_off_by_one():
    # 6 out of 8 are off-by-one
    attempts = [_attempt(is_correct=0, unparsable=0, parsed_answer="43") for _ in range(6)]
    attempts += [_attempt(is_correct=1, unparsable=0, parsed_answer="42") for _ in range(2)]
    r = _classify_question(attempts, gold="42", numeric=True)
    assert r["off_by_one_frac"] == pytest.approx(6 / 8, abs=0.01)


def test_classify_consistent_wrong():
    # all 8 give the same wrong answer
    attempts = [_attempt(is_correct=0, unparsable=0, parsed_answer="99") for _ in range(8)]
    r = _classify_question(attempts, gold="42", numeric=True)
    assert r["consistent_wrong"] == 1
    assert r["high_variance"] == 0


def test_classify_high_variance():
    # 6 different wrong answers out of 6 attempts
    answers = ["10", "20", "30", "40", "50", "60"]
    attempts = [_attempt(is_correct=0, unparsable=0, parsed_answer=a) for a in answers]
    r = _classify_question(attempts, gold="1", numeric=True)
    assert r["high_variance"] == 1
    assert r["consistent_wrong"] == 0


def test_classify_hedge():
    attempts = [_attempt(is_correct=0, unparsable=0, parsed_answer="5 or 6") for _ in range(4)]
    r = _classify_question(attempts, gold="5", numeric=False)
    assert r["hedge_frac"] == 1.0


# ── failure_summary_text ─────────────────────────────────────────────────────

def test_summary_text_no_failures():
    analysis = {"n_failing_questions": 0, "per_question": [], "aggregate": {}, "dominant_mode": "none"}
    text = failure_summary_text(analysis)
    assert "no consistent" in text.lower()


def test_summary_text_with_data():
    analysis = {
        "n_failing_questions": 5,
        "per_question": [],
        "aggregate": {
            "unparsable_frac": 0.7,
            "off_by_one_frac": 0.1,
            "hedge_frac": 0.05,
            "consistent_wrong_frac": 0.1,
            "high_variance_frac": 0.05,
        },
        "dominant_mode": "unparsable_frac",
    }
    text = failure_summary_text(analysis)
    assert "70%" in text
    assert "format" in text.lower()


import pytest  # noqa: E402  (placed here to keep imports at the top clean in practice)
