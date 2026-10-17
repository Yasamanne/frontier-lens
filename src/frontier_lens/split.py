"""Hold out entire templates for eval so train items cannot leak into eval."""

from __future__ import annotations

# Eval holds out a disjoint seed range of each generator family, under its own
# template name, so a template never sits in both train and eval.
# Train origins are 0..7. Eval origins are 10000..10003.
EVAL_TEMPLATES = frozenset(
    {
        "shell_pipelines_eval",
        "python_state_eval",
        "constraint_counting_eval",
        "multi_step_units_eval",
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
