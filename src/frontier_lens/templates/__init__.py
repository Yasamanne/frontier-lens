from __future__ import annotations

from frontier_lens.templates.constraint_counting import specs as count_specs
from frontier_lens.templates.multi_step_units import specs as units_specs
from frontier_lens.templates.python_state import specs as state_specs
from frontier_lens.templates.shell_pipelines import specs as shell_specs
from frontier_lens.templates.spec import ItemSpec


def all_specs() -> list[ItemSpec]:
    specs = shell_specs() + state_specs() + units_specs() + count_specs()
    by = {}
    for spec in specs:
        by.setdefault((spec.topic, spec.level), []).append(spec)
    topics = ("shell_pipelines", "python_state", "multi_step_units", "constraint_counting")
    expected = {(t, 1): 2 for t in topics}
    expected.update({(t, lv): 12 for t in topics for lv in (2, 3)})
    for key, n in expected.items():
        got = len(by.get(key, []))
        if got != n:
            raise RuntimeError(f"{key} has {got} items, expected {n}")
    return specs
