from __future__ import annotations

import math
from dataclasses import dataclass

from frontier_lens.config import LEARNABLE_HI, LEARNABLE_LO


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


def pass_rate(correct: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return correct / n


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-c, k)/C(n, k)."""
    if k <= 0 or n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def in_learnable_band(rate: float) -> bool:
    return LEARNABLE_LO < rate < LEARNABLE_HI


def headroom(n: int, c: int, k: int) -> float:
    """pass@k minus pass@1 for the policy (pass@1 == pass_rate)."""
    return pass_at_k(n, c, k) - pass_at_k(n, c, 1)


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
