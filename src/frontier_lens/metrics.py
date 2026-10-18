from __future__ import annotations

import math
from collections import defaultdict

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.scope import NOT_BROKEN, run_clause
from frontier_lens.stats import headroom, in_learnable_band, pass_at_k, pass_rate, wilson_interval


def new_vs_baseline(run_id: str, model_id: str, topic: str, band: str) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT q.split, q.source, q.id, AVG(a.is_correct) acc
                FROM attempts a JOIN questions q ON q.id=a.question_id
                WHERE a.run_id=:r AND a.model_id=:m AND q.topic=:t AND q.band=:b
                GROUP BY q.id
                """
            ),
            {"r": run_id, "m": model_id, "t": topic, "b": band},
        ).mappings().all()
    base = [r["acc"] for r in rows if r["split"] == "train" and r["source"] == "seed"]
    new = [r["acc"] for r in rows if r["source"] == "generated"]
    def pack(vals):
        n = len(vals)
        succ = sum(1 for v in vals if v >= 0.5)
        lo, hi = wilson_interval(succ, n) if n else (0.0, 1.0)
        return {"n": n, "acc": (sum(vals) / n if n else 0.0), "lo": lo, "hi": hi}

    return {"baseline": pack(base), "generated": pack(new)}


def junk_by_mode() -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT generator_mode, status, auto_reject_reason, COUNT(*) n
                FROM candidates GROUP BY 1,2,3
                """
            )
        ).mappings().all()
    out: dict = defaultdict(list)
    for r in rows:
        out[r["generator_mode"] or "unknown"].append(dict(r))
    return dict(out)


def policy_item_rates(run_id: str, model_id: str, split: str | None = "train") -> list[dict]:
    engine = get_engine()
    with engine.connect() as conn:
        sql = f"""
            SELECT q.id, q.template, q.source, q.generator_mode, q.split, a.is_correct
            FROM attempts a JOIN questions q ON q.id=a.question_id
            WHERE {run_clause("r")} AND a.model_id=:m AND {NOT_BROKEN}
        """
        params: dict = {"r": run_id, "m": model_id}
        if split:
            sql += " AND q.split=:split"
            params["split"] = split
        rows = conn.execute(text(sql), params).mappings().all()
    by: dict[str, dict] = {}
    for r in rows:
        slot = by.setdefault(
            r["id"],
            {
                "id": r["id"],
                "template": r["template"],
                "source": r["source"],
                "generator_mode": r["generator_mode"] or "seed",
                "split": r["split"],
                "correct": 0,
                "n": 0,
            },
        )
        slot["n"] += 1
        slot["correct"] += int(r["is_correct"])
    out = []
    for slot in by.values():
        n, c = slot["n"], slot["correct"]
        rate = pass_rate(c, n)
        out.append(
            {
                **slot,
                "pass_rate": rate,
                "pass_at_k": pass_at_k(n, c, n),
                "headroom": headroom(n, c, n),
                "in_band": in_learnable_band(rate),
            }
        )
    return out


HIST_LABELS = (
    ["0 (out)"]
    + ["0–⅛", "⅛–¼", "¼–⅜", "⅜–½", "½–⅝", "⅝–¾", "¾–⅞", "⅞–1"]
    + ["1 (out)"]
)


def histogram_counts(rates: list[float]) -> list[int]:
    """Count pass rates into ``HIST_LABELS``: exactly 0, eight interior eighths, exactly 1.

    Interior bars include their upper edge, so 1/8 is in "0–⅛" and 1/3 in "¼–⅜",
    whatever k produced the rate.
    """
    counts = [0] * len(HIST_LABELS)
    for r in rates:
        if r <= 0:
            counts[0] += 1
        elif r >= 1:
            counts[-1] += 1
        else:
            counts[min(8, max(1, math.ceil(r * 8 - 1e-9)))] += 1
    return counts


def band_share(items: list[dict]) -> dict:
    n = len(items)
    hits = sum(1 for i in items if i["in_band"])
    by_template: dict[str, list[bool]] = defaultdict(list)
    by_mode: dict[str, list[bool]] = defaultdict(list)
    for i in items:
        by_template[i["template"]].append(i["in_band"])
        by_mode[i["generator_mode"]].append(i["in_band"])

    def pack(flags: list[bool]) -> dict:
        m = len(flags)
        return {"n": m, "share": (sum(flags) / m if m else 0.0)}

    return {
        "n": n,
        "share": (hits / n if n else 0.0),
        "mean_headroom": (sum(i["headroom"] for i in items) / n if n else 0.0),
        "by_template": {k: pack(v) for k, v in sorted(by_template.items())},
        "by_mode": {k: pack(v) for k, v in sorted(by_mode.items())},
    }


def latest_audit() -> dict | None:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM audit_batches ORDER BY created_at DESC LIMIT 1")
        ).mappings().first()
    return dict(row) if row else None


def candidate_band_by_mode() -> dict:
    """Headline generator metric: share of pending/approved items in the learnable band."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT generator_mode, policy_pass_rate, status, auto_reject_reason
                FROM candidates WHERE rl_split='train' OR rl_split IS NULL
                """
            )
        ).mappings().all()
    by: dict[str, dict] = defaultdict(lambda: {"n": 0, "in_band": 0, "out_of_band": 0})
    for r in rows:
        mode = r["generator_mode"] or "unknown"
        by[mode]["n"] += 1
        rate = r["policy_pass_rate"]
        if rate is not None and in_learnable_band(rate):
            by[mode]["in_band"] += 1
        if r["auto_reject_reason"] == "out_of_band":
            by[mode]["out_of_band"] += 1
    out = {}
    for mode, d in by.items():
        out[mode] = {
            **d,
            "share": d["in_band"] / d["n"] if d["n"] else 0.0,
        }
    return out
