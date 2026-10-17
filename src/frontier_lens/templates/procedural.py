"""Shared seed-bank assembly for procedural item generators.

`generate(seed, **knobs)` is pure. `materialize` runs the sandbox and, on
failure, calls generate again at seed+1000, up to five tries.
"""

from __future__ import annotations

from collections.abc import Callable

from frontier_lens.sandbox import execute
from frontier_lens.templates.spec import ItemSpec

TRAIN_SEEDS = tuple(range(8))
EVAL_SEEDS = tuple(range(10_000, 10_004))
MAX_TRIES = 5

GenerateFn = Callable[..., ItemSpec]
ExecuteFn = Callable[[str, str, dict], object]
AcceptFn = Callable[[ItemSpec, str], bool]


def _run(lang: str, code: str, files: dict) -> object:
    return execute(lang, code, files, twice=False)


def materialize(
    generate: GenerateFn,
    seed: int,
    knobs: dict,
    *,
    execute_fn: ExecuteFn | None = None,
    accept: AcceptFn | None = None,
) -> ItemSpec:
    """Return a spec whose program runs. Bump the seed by 1000 between tries."""
    runner = execute_fn or _run
    last = "not run"
    for attempt in range(MAX_TRIES):
        used = seed + attempt * 1000
        try:
            spec = generate(used, **knobs)
        except Exception as exc:
            last = f"generate: {exc}"
            continue
        try:
            result = runner(spec.lang, spec.code, spec.input_files or {})
        except Exception as exc:
            last = f"sandbox: {exc}"
            continue
        if not getattr(result, "ok", False):
            last = getattr(result, "reason", None) or "exec_error"
            continue
        key = getattr(result, "stdout", "") or ""
        if not key or key in spec.stem:
            last = "empty" if not key else "key_in_stem"
            continue
        if accept is not None and not accept(spec, key):
            last = "rejected"
            continue
        spec.params = {**spec.params, "seed": used, "origin_seed": seed}
        return spec
    raise RuntimeError(f"materialize failed after {MAX_TRIES} tries from seed {seed}: {last}")


def bank(
    topic: str,
    generate: GenerateFn,
    harness: list[ItemSpec],
    *,
    accept: AcceptFn | None = None,
) -> list[ItemSpec]:
    """L1 harness plus L2/L3 items at the calibrated knobs.

    Train seeds are 0..7 (`{topic}_train`). Eval seeds are 10000..10003
    (`{topic}_eval`). The two seed ranges never mix.
    """
    from frontier_lens.calibrate import knobs_for, lineage_params

    items = list(harness)
    chosen = knobs_for(topic)
    for level in (2, 3):
        knob = dict(chosen[level])
        lineage = lineage_params(topic, level)
        for split, seeds in (("train", TRAIN_SEEDS), ("eval", EVAL_SEEDS)):
            for origin in seeds:
                spec = materialize(generate, origin, knob, accept=accept)
                spec.topic = topic
                spec.level = level
                spec.template = f"{topic}_{split}"
                spec.params = {**spec.params, "seed_range": split, **lineage}
                items.append(spec)
    return items
