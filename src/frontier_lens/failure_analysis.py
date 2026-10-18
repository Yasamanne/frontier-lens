"""Failure-mode analysis for policy traces.

For each question in a (topic, band) cell that the policy mostly fails
(pass_rate < 0.5 across k samples), we classify the failure:

  unparsable      — no FINAL: line in one or more outputs
  off_by_one      — numeric answer differs from gold by exactly ±1
  hedge           — answer contains "or"/"either"/"/" (refusal to commit)
  consistent_wrong — all k outputs give the same wrong answer (systematic)
  high_variance   — outputs give ≥3 distinct wrong answers (noise)

The aggregate fractions tell the generator *which kind of difficulty to
inject* into new items, rather than just re-listing failing question stems.
"""

from __future__ import annotations

import re
from collections import defaultdict

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.parse import normalize

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_HEDGE_RE = re.compile(r"\b(or|either)\b|/", re.IGNORECASE)


def _off_by_one(got: str, gold: str) -> bool:
    """True if both parse as numbers and differ by exactly 1."""
    try:
        return abs(float(got) - float(gold)) == 1.0
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# per-question failure breakdown
# ---------------------------------------------------------------------------

def _classify_question(attempts: list[dict], gold: str, numeric: bool) -> dict:
    """Return failure-mode counts for one question's k attempts."""
    n = len(attempts)
    if n == 0:
        return {}

    norm_gold = normalize(gold, numeric)
    n_unparsable = 0
    n_off_by_one = 0
    n_hedge = 0
    seen_wrong: set[str] = set()

    for a in attempts:
        if a["unparsable"]:
            n_unparsable += 1
            continue
        ans = a["parsed_answer"] or ""
        norm_ans = normalize(ans, numeric)
        if norm_ans == norm_gold:
            continue                  # correct — not counted as a failure mode
        seen_wrong.add(norm_ans)
        if numeric and _off_by_one(norm_ans, norm_gold):
            n_off_by_one += 1
        if _HEDGE_RE.search(ans):
            n_hedge += 1

    n_wrong = n - sum(1 for a in attempts if a["is_correct"])
    consistent_wrong = int(len(seen_wrong) == 1 and n_wrong == n)
    high_variance = int(len(seen_wrong) >= max(3, n // 2))

    return {
        "n": n,
        "n_wrong": n_wrong,
        "unparsable_frac": n_unparsable / n,
        "off_by_one_frac": n_off_by_one / n,
        "hedge_frac": n_hedge / n,
        "consistent_wrong": consistent_wrong,
        "high_variance": high_variance,
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def failure_modes(
    run_id: str,
    model_id: str,
    topic: str,
    band: str,
    split: str = "train",
) -> dict:
    """
    Return a failure-mode analysis dict for the given (run, model, topic, band) cell.

    Schema::

        {
          "n_failing_questions": int,
          "per_question": [{"question_id", "stem_snippet", "n", "n_correct", "modes": {...}}],
          "aggregate": {
            "unparsable_frac": float,   # avg fraction of k attempts with no FINAL:
            "off_by_one_frac": float,   # avg fraction that are ±1 off
            "hedge_frac": float,        # avg fraction that hedge
            "consistent_wrong_frac": float,   # share of failing questions with ONE wrong answer
            "high_variance_frac": float,      # share with ≥3 distinct wrong answers
          },
          "dominant_mode": str,
          "per_question": [...],
        }
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT q.id as qid, q.stem, q.key, q.numeric,
                       a.sample_index, a.raw_output, a.parsed_answer,
                       a.is_correct, a.unparsable
                FROM attempts a
                JOIN questions q ON q.id = a.question_id
                WHERE a.run_id  = :r
                  AND a.model_id = :m
                  AND q.topic   = :t
                  AND q.band    = :b
                  AND q.split   = :split
                """
            ),
            {"r": run_id, "m": model_id, "t": topic, "b": band, "split": split},
        ).mappings().all()

    # group by question
    by_q: dict[str, dict] = {}
    for r in rows:
        qid = r["qid"]
        if qid not in by_q:
            by_q[qid] = {
                "stem": r["stem"] or "",
                "gold": r["key"] or "",
                "numeric": bool(r["numeric"]),
                "attempts": [],
            }
        by_q[qid]["attempts"].append(dict(r))

    per_q_out: list[dict] = []
    acc: dict[str, float] = defaultdict(float)
    n_failing = 0

    for qid, info in by_q.items():
        attempts = info["attempts"]
        n = len(attempts)
        n_correct = sum(1 for a in attempts if a["is_correct"])
        pass_rate = n_correct / n if n else 1.0

        if pass_rate >= 0.5:
            continue  # not a failing question

        n_failing += 1
        modes = _classify_question(attempts, info["gold"], info["numeric"])
        per_q_out.append(
            {
                "question_id": qid,
                "stem_snippet": info["stem"][:320],
                "n": n,
                "n_correct": n_correct,
                "modes": modes,
            }
        )
        for key in ("unparsable_frac", "off_by_one_frac", "hedge_frac"):
            acc[key] += modes.get(key, 0.0)
        acc["consistent_wrong"] += modes.get("consistent_wrong", 0)
        acc["high_variance"] += modes.get("high_variance", 0)

    if n_failing == 0:
        return {
            "n_failing_questions": 0,
            "per_question": [],
            "aggregate": {},
            "dominant_mode": "none",
        }

    aggregate = {
        "unparsable_frac": acc["unparsable_frac"] / n_failing,
        "off_by_one_frac": acc["off_by_one_frac"] / n_failing,
        "hedge_frac": acc["hedge_frac"] / n_failing,
        "consistent_wrong_frac": acc["consistent_wrong"] / n_failing,
        "high_variance_frac": acc["high_variance"] / n_failing,
    }
    dominant = max(aggregate, key=aggregate.get)

    return {
        "n_failing_questions": n_failing,
        "per_question": per_q_out,
        "aggregate": aggregate,
        "dominant_mode": dominant,
    }


def zero_pass_questions(
    run_id: str,
    model_id: str,
    split: str = "train",
    *,
    include_uncoded: bool = False,
) -> list[dict]:
    """Return questions where the model got 0 correct across ALL k attempts (pass@k = 0).

    By default only returns questions with non-empty verifier code — uploaded
    questions with hardcoded keys (empty code, multiple-choice answers like
    ``$B$``) are excluded because the sandbox cannot re-run them and rescue
    calibration will always return 0. Pass ``include_uncoded=True`` to list them
    too (they can never be rescued, so they belong straight in the SFT pool).

    Each entry::

        {
          "id": str,
          "stem": str,
          "key": str,
          "numeric": bool,
          "topic": str,
          "band": str,
          "n": int,          # total attempts
          "n_correct": int,  # always 0
        }
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT q.id, q.stem, q.key, q.numeric, q.topic, q.band,
                       COUNT(*) as n, SUM(a.is_correct) as n_correct
                FROM attempts a
                JOIN questions q ON q.id = a.question_id
                WHERE a.run_id  = :r
                  AND a.model_id = :m
                  AND q.split   = :split
                  AND (:uncoded = 1 OR (q.code IS NOT NULL AND TRIM(q.code) != ''))
                  AND COALESCE(q.sme_flagged, 0) = 0
                GROUP BY q.id
                HAVING SUM(a.is_correct) = 0
                   AND (COUNT(*) - SUM(a.unparsable)) > 0
                """
            ),
            {"r": run_id, "m": model_id, "split": split, "uncoded": int(include_uncoded)},
        ).mappings().all()
    return [dict(r) for r in rows]


def rl_suitable_questions(
    run_id: str,
    model_id: str,
    split: str = "train",
) -> list[dict]:
    """Return questions in the learnable band using **content pass rate**.

    A question is RL-suitable when the model sometimes gets it right and
    sometimes gets it wrong *on parseable attempts*.  Unparsable (format)
    failures are excluded from the denominator so that a question where the
    model always answers correctly but occasionally forgets the FINAL: format
    is NOT classified as RL-suitable (it would have content_pass_rate = 1.0).

    Routing:
      n_parseable = 0            → all format failures → excluded
      content_rate = 0           → dead zone → excluded (use zero_pass_questions)
      0 < content_rate < 1       → RL-suitable ✅
      content_rate = 1.0         → too easy (only format failures among wrong) → excluded

    Each entry schema:
      id, stem, key, numeric, topic, band, n, n_correct, n_unparsable, n_parseable,
      content_pass_rate
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT q.id, q.stem, q.key, q.numeric, q.topic, q.band,
                       COUNT(*)          AS n,
                       SUM(a.is_correct) AS n_correct,
                       SUM(a.unparsable) AS n_unparsable,
                       COUNT(*) - SUM(a.unparsable) AS n_parseable
                FROM attempts a
                JOIN questions q ON q.id = a.question_id
                WHERE a.run_id   = :r
                  AND a.model_id = :m
                  AND q.split    = :split
                  AND COALESCE(q.sme_flagged, 0) = 0
                GROUP BY q.id
                HAVING
                  SUM(a.is_correct) > 0
                  AND (COUNT(*) - SUM(a.unparsable)) > 0
                  AND SUM(a.is_correct) < (COUNT(*) - SUM(a.unparsable))
                """
            ),
            {"r": run_id, "m": model_id, "split": split},
        ).mappings().all()

    result = []
    for r in rows:
        d = dict(r)
        np_ = d["n_parseable"]
        d["content_pass_rate"] = round(d["n_correct"] / np_, 3) if np_ else 0.0
        result.append(d)
    return result


def failure_modes_all_cells(run_id: str, model_id: str, split: str = "train") -> list[dict]:
    """Return failure analysis for every (topic, band) cell that has failing questions."""
    engine = get_engine()
    with engine.connect() as conn:
        cells = conn.execute(
            text(
                """
                SELECT DISTINCT q.topic, q.band
                FROM attempts a
                JOIN questions q ON q.id = a.question_id
                WHERE a.run_id = :r AND a.model_id = :m AND q.split = :split
                """
            ),
            {"r": run_id, "m": model_id, "split": split},
        ).all()

    out = []
    for topic, band in cells:
        analysis = failure_modes(run_id, model_id, topic, band, split)
        if analysis["n_failing_questions"] > 0:
            out.append({"topic": topic, "band": band, **analysis})
    return out


def failure_summary_text(analysis: dict) -> str:
    """Concise plain-text summary suitable for injection into a generation prompt."""
    if analysis.get("n_failing_questions", 0) == 0:
        return "(no consistent failure pattern detected — model rarely fails this cell)"

    agg = analysis["aggregate"]
    dominant = analysis["dominant_mode"].replace("_frac", "").replace("_", " ")

    LABELS = {
        "unparsable_frac": "format failure (no FINAL: line)",
        "off_by_one_frac": "off-by-one numeric error (±1 from gold)",
        "hedge_frac": "hedging answer (or / either)",
        "consistent_wrong_frac": "consistent wrong answer (systematic reasoning error)",
        "high_variance_frac": "high variance (many different wrong answers — random noise)",
    }
    lines = [
        f"Policy failure analysis across {analysis['n_failing_questions']} question(s) in this cell:",
        f"  Dominant mode: {dominant}",
    ]
    for key, label in LABELS.items():
        v = agg.get(key, 0.0)
        bar = "█" * int(v * 10)
        lines.append(f"  {label}: {v:.0%} {bar}")

    # actionable instruction per dominant mode
    instructions = {
        "unparsable": (
            "→ Design items where the reasoning chain is long enough to make the model forget "
            "the required FINAL: format — multi-step problems with distractors."
        ),
        "off by one": (
            "→ Include boundary conditions and fence-post counting; the model is close but "
            "slips by 1. Use inclusive/exclusive ranges, 1-indexed vs 0-indexed confusion."
        ),
        "hedge": (
            "→ Make items with a unique, unambiguous answer so hedging always scores 0. "
            "Avoid phrasing that allows 'depends' answers."
        ),
        "consistent wrong": (
            "→ Target the same systematic misconception with varied surface forms. "
            "The model has a fixed wrong mental model for this type."
        ),
        "high variance": (
            "→ Reduce surface noise — use simpler phrasing so failures reflect "
            "genuine capability limits rather than prompt sensitivity."
        ),
    }
    for trigger, note in instructions.items():
        if trigger in dominant:
            lines.append(note)
            break

    return "\n".join(lines)
