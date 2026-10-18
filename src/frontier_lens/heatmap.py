from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.scope import NOT_BROKEN, run_clause
from frontier_lens.stats import cell_stats


def load_cell_rows(run_id: str, split: str = "train"):
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT q.topic, q.band, q.id as qid, a.model_id, a.is_correct
                FROM attempts a
                JOIN questions q ON q.id = a.question_id
                WHERE {run_clause("run")} AND q.split = :split AND {NOT_BROKEN}
                """
            ),
            {"run": run_id, "split": split},
        ).mappings().all()
    return list(rows)


def heatmap(run_id: str, model_id: str, split: str = "train") -> list:
    rows = [r for r in load_cell_rows(run_id, split) if r["model_id"] == model_id]
    by_item: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for r in rows:
        by_item[(r["topic"], r["band"], r["qid"])].append(bool(r["is_correct"]))
    overall = []
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for (topic, band, qid), samples in by_item.items():
        maj = sum(samples) * 2 >= len(samples)
        agree = len(set(samples)) == 1
        grouped[(topic, band)].append((maj, agree))
        overall.append(maj)
    overall_mean = sum(overall) / len(overall) if overall else 0.0
    cells = []
    for (topic, band), items in sorted(grouped.items()):
        majs = [m for m, _ in items]
        agrees = [a for _, a in items]
        cells.append(cell_stats(topic, band, majs, agrees, overall_mean))
    return cells


def print_heatmap(run_id: str, model_id: str) -> None:
    print(f"run={run_id} model={model_id} split=train")
    print(f"{'topic':<24} {'band':<10} {'n':>3} {'acc':>6} {'ci':>18} {'stab':>6} weak")
    for c in heatmap(run_id, model_id):
        ci = f"[{c.lo:.2f},{c.hi:.2f}]"
        print(
            f"{c.topic:<24} {c.band:<10} {c.n:>3} {c.accuracy:>6.2f} {ci:>18} {c.stability:>6.2f} {c.weak}"
        )


# A cell with fewer items is too noisy to steer generation.
MIN_CELL_ITEMS = 3


def weakest_cell(run_id: str, model_id: str):
    """Cell to target next: confirmed-weak cells first, then lowest accuracy.

    Cells with fewer than ``MIN_CELL_ITEMS`` items are skipped unless every cell is that small.
    """
    cells = heatmap(run_id, model_id)
    eligible = [c for c in cells if c.n >= MIN_CELL_ITEMS] or cells
    ranked = sorted(eligible, key=lambda c: (not c.weak, c.accuracy, c.hi))
    return ranked[0] if ranked else None
