from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.stats import wilson_interval


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
