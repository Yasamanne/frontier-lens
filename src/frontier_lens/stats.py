from __future__ import annotations

import math
from dataclasses import dataclass


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class CellStats:
    topic: str
    band: str
    n: int
    accuracy: float
    lo: float
    hi: float
    stability: float
    weak: bool


def majority_correct(sample_correct: list[bool]) -> bool:
    if not sample_correct:
        return False
    return sum(sample_correct) * 2 >= len(sample_correct)


def cell_stats(
    topic: str,
    band: str,
    item_majorities: list[bool],
    item_all_agree: list[bool],
    overall_mean: float,
) -> CellStats:
    n = len(item_majorities)
    successes = sum(item_majorities)
    acc = successes / n if n else 0.0
    lo, hi = wilson_interval(successes, n)
    stability = sum(item_all_agree) / n if n else 0.0
    weak = n > 0 and hi < overall_mean
    return CellStats(topic, band, n, acc, lo, hi, stability, weak)
