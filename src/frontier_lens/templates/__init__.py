from __future__ import annotations

from functools import lru_cache

from frontier_lens.templates.constraint_counting import specs as count_specs
from frontier_lens.templates.fsm_queries import specs as fsm_specs
from frontier_lens.templates.logic_grid import specs as logic_grid_specs
from frontier_lens.templates.minimax import specs as minimax_specs
from frontier_lens.templates.multi_step_units import specs as units_specs
from frontier_lens.templates.procedural import EVAL_SEEDS, TRAIN_SEEDS
from frontier_lens.templates.python_state import specs as state_specs
from frontier_lens.templates.shell_pipelines import specs as shell_specs
from frontier_lens.templates.spec import ItemSpec

_ALL_TOPICS = (
    "shell_pipelines",
    "python_state",
    "multi_step_units",
    "constraint_counting",
    "fsm_queries",   # FSM simulation — state tracking
    "minimax",       # game-tree backward induction
    "logic_grid",    # positional constraint propagation
)


@lru_cache(maxsize=1)
def all_specs() -> list[ItemSpec]:
    specs = (
        shell_specs()
        + state_specs()
        + units_specs()
        + count_specs()
        + fsm_specs()
        + minimax_specs()
        + logic_grid_specs()
    )
    by: dict[tuple[str, int], list[ItemSpec]] = {}
    for spec in specs:
        by.setdefault((spec.topic, spec.level), []).append(spec)
    per_level = len(TRAIN_SEEDS) + len(EVAL_SEEDS)
    expected = {(t, 1): 2 for t in _ALL_TOPICS}
    expected.update({(t, lv): per_level for t in _ALL_TOPICS for lv in (2, 3)})
    for key, n in expected.items():
        got = len(by.get(key, []))
        if got != n:
            raise RuntimeError(f"{key} has {got} items, expected {n}")
    return specs
