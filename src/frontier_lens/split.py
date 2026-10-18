"""Assign harness / train / eval splits.

Each family's eval items come from its own ``{topic}_eval`` template: the same
generator run on a disjoint seed range. No template appears in both train and
eval, but eval is near-distribution, not an unseen task type.
"""

from __future__ import annotations

from frontier_lens.templates import TOPICS

# Eval holds out a disjoint seed range of each generator family, under its own
# template name, so a template never sits in both train and eval.
# Train origins are 0..7. Eval origins are 10000..10003.
# Derived from the topic list so a new family cannot silently skip eval.
EVAL_TEMPLATES = frozenset(f"{topic}_eval" for topic in TOPICS)


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
