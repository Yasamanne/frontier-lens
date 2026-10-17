"""Choose L2 and L3 knobs from a monotone grid.

Production path: `python -m frontier_lens.calibrate --live` measures the
policy on one probe item per grid point, writes `data/level_knobs.json`
(policy id, k, timestamp, rates, grid indices), then `seed --force` uses
those knobs.

Offline/CI: a synthetic decreasing probe curve so tests need no API.
Set FRONTIER_LENS_REQUIRE_LIVE_KNOBS=1 to refuse that stub.

L2 is the grid point whose pass rate sits in [0.2, 0.8], closest to 0.5,
and that still has a harder point with a lower rate. L3 is the hardest later
point under 0.2 when one exists, otherwise the hardest later point whose rate
is lower than L2.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from frontier_lens.config import DATA_DIR, REQUIRE_LIVE_KNOBS

L2_LO = 0.2
L2_HI = 0.8
PROBE_RATES = [0.95, 0.80, 0.50, 0.35, 0.15, 0.05]
PROBE_SEED = 424242
KNOBS_PATH = DATA_DIR / "level_knobs.json"

# Later points raise the primary knob. L2/L3 are indices into these lists,
# not hand-written item lists.
GRIDS: dict[str, list[dict]] = {
    "python_state": [
        {"steps": 28, "alias_p": 0.35, "containers": 4},
        {"steps": 40, "alias_p": 0.45, "containers": 4},
        {"steps": 52, "alias_p": 0.55, "containers": 5},
        {"steps": 68, "alias_p": 0.65, "containers": 5},
        {"steps": 84, "alias_p": 0.75, "containers": 6},
        {"steps": 100, "alias_p": 0.85, "containers": 6},
    ],
    "shell_pipelines": [
        {"rows": 80, "stages": 4},
        {"rows": 120, "stages": 5},
        {"rows": 160, "stages": 6},
        {"rows": 200, "stages": 6},
        {"rows": 260, "stages": 7},
        {"rows": 320, "stages": 8},
    ],
    "constraint_counting": [
        {"length": 8, "alphabet_size": 4, "n_constraints": 4},
        {"length": 9, "alphabet_size": 3, "n_constraints": 4},
        {"length": 10, "alphabet_size": 3, "n_constraints": 5},
        {"length": 11, "alphabet_size": 3, "n_constraints": 5},
        {"length": 12, "alphabet_size": 3, "n_constraints": 5},
        {"length": 12, "alphabet_size": 3, "n_constraints": 6},
    ],
    "multi_step_units": [
        {"steps": 10},
        {"steps": 12},
        {"steps": 14},
        {"steps": 16},
        {"steps": 18},
        {"steps": 20},
    ],
    # ── New cognitive families ──────────────────────────────────────────────
    # FSM: state-tracking (not arithmetic); difficulty = string length + states
    "fsm_queries": [
        {"n_states": 3, "n_symbols": 2, "string_length": 4},
        {"n_states": 4, "n_symbols": 2, "string_length": 6},
        {"n_states": 4, "n_symbols": 3, "string_length": 8},
        {"n_states": 5, "n_symbols": 3, "string_length": 10},
        {"n_states": 6, "n_symbols": 3, "string_length": 12},
        {"n_states": 7, "n_symbols": 3, "string_length": 14},
    ],
    # Minimax: backward induction; difficulty = tree depth + branching
    # Primary knob (depth) must be non-decreasing for choose_levels()
    "minimax": [
        {"depth": 2, "branching": 2},   # 4 leaves
        {"depth": 2, "branching": 3},   # 9 leaves
        {"depth": 3, "branching": 2},   # 8 leaves
        {"depth": 3, "branching": 3},   # 27 leaves
        {"depth": 4, "branching": 2},   # 16 leaves
        {"depth": 4, "branching": 3},   # 81 leaves
    ],
    # Logic grid: constraint propagation; difficulty = items + clue variety
    "logic_grid": [
        {"n_items": 3, "n_clue_types": 1},
        {"n_items": 4, "n_clue_types": 1},
        {"n_items": 4, "n_clue_types": 2},
        {"n_items": 5, "n_clue_types": 2},
        {"n_items": 5, "n_clue_types": 3},
        {"n_items": 6, "n_clue_types": 3},
    ],
}

PRIMARY = {
    "python_state": "steps",
    "shell_pipelines": "rows",
    "constraint_counting": "length",
    "multi_step_units": "steps",
    "fsm_queries": "string_length",
    "minimax": "depth",
    "logic_grid": "n_items",
}


def choose_levels(rates: list[float]) -> tuple[int, int]:
    """Return (l2_index, l3_index) into a grid ordered easy to hard."""
    if len(rates) < 2:
        raise ValueError("need at least two grid points")
    candidates: list[tuple[float, int, int]] = []
    for i, rate in enumerate(rates):
        if not (L2_LO <= rate <= L2_HI):
            continue
        later = [j for j in range(i + 1, len(rates)) if rates[j] < rate]
        if not later:
            continue
        below = [j for j in later if rates[j] < L2_LO]
        j = (below or later)[-1]
        candidates.append((abs(rate - 0.5), i, j))
    if not candidates:
        raise ValueError(f"no L2/L3 pair in band with a harder lower rate: {rates}")
    _, l2, l3 = min(candidates)
    return l2, l3


def _topic_block(grid: list[dict], l2: int, l3: int, rates: list[float]) -> dict:
    return {
        "2": dict(grid[l2]),
        "3": dict(grid[l3]),
        "l2_index": l2,
        "l3_index": l3,
        "rates": list(rates),
    }


def probe_manifest() -> dict:
    """Synthetic rates so CI/seed work without a policy. Not a measured level."""
    l2, l3 = choose_levels(PROBE_RATES)
    payload: dict = {
        "source": "probe",
        "policy_id": None,
        "policy_model": None,
        "k": None,
        "created_at": None,
        "note": "Synthetic decreasing curve. Run python -m frontier_lens.calibrate --live.",
    }
    for topic, grid in GRIDS.items():
        if len(grid) != len(PROBE_RATES):
            raise RuntimeError(f"{topic} grid length {len(grid)} != probe")
        payload[topic] = _topic_block(grid, l2, l3, PROBE_RATES)
    return payload


def default_knobs() -> dict[str, dict[int, dict]]:
    manifest = probe_manifest()
    return {topic: {2: dict(manifest[topic]["2"]), 3: dict(manifest[topic]["3"])} for topic in GRIDS}


def load_manifest(path: Path | None = None) -> dict:
    target = path or KNOBS_PATH
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        if data.get("source") == "policy":
            return data
    return probe_manifest()


def knobs_for(topic: str) -> dict[int, dict]:
    """Knob settings for levels 2 and 3. Live policy file wins over the probe curve."""
    raw = load_manifest()[topic]
    return {2: dict(raw["2"]), 3: dict(raw["3"])}


def lineage_params(topic: str, level: int) -> dict:
    """Fields stamped onto every L2/L3 seed so levels are auditable."""
    manifest = load_manifest()
    block = manifest[topic]
    idx_key = "l2_index" if level == 2 else "l3_index"
    return {
        "knob_source": manifest.get("source"),
        "knob_policy": manifest.get("policy_id"),
        "knob_k": manifest.get("k"),
        "knob_grid_index": block.get(idx_key),
    }


def require_live_if_configured() -> None:
    if REQUIRE_LIVE_KNOBS and load_manifest().get("source") != "policy":
        raise RuntimeError(
            "FRONTIER_LENS_REQUIRE_LIVE_KNOBS=1 but data/level_knobs.json is missing "
            "or not policy-measured. Run: uv run python -m frontier_lens.calibrate --live"
        )


def print_knob_report() -> None:
    manifest = load_manifest()
    source = manifest.get("source")
    print(
        f"L2/L3 knobs source={source} policy={manifest.get('policy_id')} "
        f"k={manifest.get('k')} at={manifest.get('created_at')}"
    )
    if source != "policy":
        print(
            "WARNING: levels are a probe stub, not measured on the policy. "
            "Production: uv run python -m frontier_lens.calibrate --live && "
            "uv run python -m frontier_lens.seed --force"
        )
    for topic in GRIDS:
        block = manifest[topic]
        print(f"  {topic} L2 {block['2']} L3 {block['3']}")


def _generator(topic: str):
    if topic == "python_state":
        from frontier_lens.templates.python_state import generate
    elif topic == "shell_pipelines":
        from frontier_lens.templates.shell_pipelines import generate
    elif topic == "constraint_counting":
        from frontier_lens.templates.constraint_counting import generate
    elif topic == "multi_step_units":
        from frontier_lens.templates.multi_step_units import generate
    elif topic == "fsm_queries":
        from frontier_lens.templates.fsm_queries import generate
    elif topic == "minimax":
        from frontier_lens.templates.minimax import generate
    elif topic == "logic_grid":
        from frontier_lens.templates.logic_grid import generate
    else:
        raise KeyError(topic)
    return generate


def measure_rates(topic: str, complete_fn=None, k: int = 8) -> list[float]:
    """One probe item per grid point. complete_fn defaults to the live policy."""
    from frontier_lens.generate import estimate_pass_rate
    from frontier_lens.sandbox import execute
    from frontier_lens.templates.procedural import materialize

    generate = _generator(topic)
    rates: list[float] = []
    for knob in GRIDS[topic]:
        spec = materialize(generate, PROBE_SEED, dict(knob))
        result = execute(spec.lang, spec.code, spec.input_files, twice=True)
        if not result.ok:
            raise RuntimeError(f"{topic} {knob}: {result.reason} {result.stderr}")
        stats = estimate_pass_rate(
            spec.stem,
            result.stdout,
            spec.numeric,
            complete_fn=complete_fn,
            k=k,
        )
        rates.append(float(stats["pass_rate"]))
    return rates


def write_knobs(
    rates_by_topic: dict[str, list[float]],
    path: Path | None = None,
    *,
    policy_id: str | None = None,
    policy_model: str | None = None,
    k: int | None = None,
) -> dict:
    target = path or KNOBS_PATH
    payload: dict = {}
    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload["source"] = "policy"
    payload["policy_id"] = policy_id
    payload["policy_model"] = policy_model
    payload["k"] = k
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["note"] = "L2/L3 chosen from measured policy pass rates on GRIDS."
    probe = probe_manifest()
    for topic, rates in rates_by_topic.items():
        l2, l3 = choose_levels(rates)
        payload[topic] = _topic_block(GRIDS[topic], l2, l3, rates)
        payload[topic]["measured"] = True
    for topic in GRIDS:
        if topic not in payload:
            payload[topic] = probe[topic]
            payload[topic]["measured"] = False
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pick L2/L3 knobs from policy pass rates")
    parser.add_argument("--live", action="store_true", help="sample the configured policy")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--topics",
        nargs="*",
        default=list(GRIDS),
    )
    args = parser.parse_args()
    if not args.live:
        print_knob_report()
        print("pass --live to measure the policy and write", KNOBS_PATH)
        return
    from frontier_lens.registry import policy_spec

    policy = policy_spec()
    rates = {topic: measure_rates(topic, k=args.k) for topic in args.topics}
    write_knobs(
        rates,
        policy_id=policy.id,
        policy_model=policy.model,
        k=args.k,
    )
    print_knob_report()
    print("wrote", KNOBS_PATH)


if __name__ == "__main__":
    main()
