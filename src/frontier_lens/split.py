"""Hold out entire templates for eval so train items cannot leak into eval."""

from __future__ import annotations

# Whole templates reserved for eval. No policy feedback is used when generating
# or seeding these. Train uses every other L2/L3 template.
EVAL_TEMPLATES = frozenset(
    {
        "state_l3",
        "count_l2",
    }
)


def rl_split(template: str, level: int) -> str:
    if level == 1:
        return "harness"
    if template in EVAL_TEMPLATES:
        return "eval"
    return "train"


def leakage(pairs: list[tuple[str, str]]) -> set[str]:
    """Return template names that appear in both train and eval."""
    by: dict[str, set[str]] = {}
    for template, split in pairs:
        if split in {"train", "eval"}:
            by.setdefault(template, set()).add(split)
    return {t for t, splits in by.items() if splits >= {"train", "eval"}}
