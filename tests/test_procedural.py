import json

import pytest

from frontier_lens.calibrate import (
    GRIDS,
    PRIMARY,
    PROBE_RATES,
    choose_levels,
    default_knobs,
    load_manifest,
    probe_manifest,
    write_knobs,
)
from frontier_lens.sandbox import SandboxResult, execute
from frontier_lens.templates import all_specs
from frontier_lens.templates.constraint_counting import generate as count_generate
from frontier_lens.templates.multi_step_units import generate as units_generate
from frontier_lens.templates.procedural import materialize
from frontier_lens.templates.python_state import generate as state_generate
from frontier_lens.templates.shell_pipelines import generate as shell_generate
from frontier_lens.templates.spec import ItemSpec


def test_choose_levels_targets_band_and_harder_l3():
    l2, l3 = choose_levels(PROBE_RATES)
    assert 0.2 <= PROBE_RATES[l2] <= 0.8
    assert l3 > l2
    assert PROBE_RATES[l3] < PROBE_RATES[l2]
    assert PROBE_RATES[l3] < 0.2


def test_grids_are_monotone_and_l3_is_harder():
    chosen = default_knobs()
    for topic, grid in GRIDS.items():
        vals = [point[PRIMARY[topic]] for point in grid]
        assert vals == sorted(vals)
        assert vals[-1] > vals[0]
        assert chosen[topic][3][PRIMARY[topic]] > chosen[topic][2][PRIMARY[topic]]


def test_same_seed_same_spec_and_key():
    cases = [
        (state_generate, {"steps": 12, "alias_p": 0.4, "containers": 4}),
        (shell_generate, {"rows": 40, "stages": 4}),
        (count_generate, {"length": 7, "alphabet_size": 3, "n_constraints": 2}),
        (units_generate, {"steps": 6}),
    ]
    for generate, knobs in cases:
        first = generate(3, **knobs)
        second = generate(3, **knobs)
        assert first.stem == second.stem
        assert first.code == second.code
        assert first.params == second.params
        result = execute(first.lang, first.code, first.input_files, twice=True)
        assert result.ok, result.reason
        assert result.stdout not in first.stem
        assert first.params["answer_space_size"] >= 100


def test_sandbox_error_bumps_seed():
    seen: list[int] = []

    def generate(seed: int, **_knobs) -> ItemSpec:
        seen.append(seed)
        return ItemSpec(
            topic="python_state",
            level=2,
            template="python_state_train",
            params={"answer_space_size": 1000, "seed": seed},
            stem="Question: what number does this print?",
            code="print(1)\n",
            lang="python",
        )

    calls = {"n": 0}

    def run(_lang, _code, _files):
        calls["n"] += 1
        if calls["n"] < 3:
            return SandboxResult(False, "", "", "timeout")
        return SandboxResult(True, "0042", "", None)

    spec = materialize(generate, 7, {"steps": 8}, execute_fn=run)
    assert seen == [7, 1007, 2007]
    assert spec.params["seed"] == 2007
    assert spec.params["origin_seed"] == 7


def test_sandbox_errors_stop_after_five_tries():
    seen: list[int] = []

    def generate(seed: int, **_knobs) -> ItemSpec:
        seen.append(seed)
        return ItemSpec(
            topic="python_state",
            level=2,
            template="t",
            params={"seed": seed},
            stem="stem",
            code="print(1)\n",
            lang="python",
        )

    def run(_lang, _code, _files):
        raise TimeoutError("timeout")

    with pytest.raises(RuntimeError):
        materialize(generate, 1, {}, execute_fn=run)
    assert seen == [1, 1001, 2001, 3001, 4001]


def test_bank_seed_ranges_do_not_overlap():
    train = set()
    held = set()
    for spec in all_specs():
        if spec.level == 1:
            continue
        origin = spec.params["origin_seed"]
        used = spec.params["seed"]
        assert (used - origin) % 1000 == 0
        assert 0 <= (used - origin) // 1000 < 5
        key = (spec.topic, origin)
        if spec.template.endswith("_eval"):
            assert spec.params["seed_range"] == "eval"
            assert 10_000 <= origin <= 10_003
            held.add(key)
        else:
            assert spec.template.endswith("_train")
            assert spec.params["seed_range"] == "train"
            assert 0 <= origin <= 7
            train.add(key)
        assert spec.params["answer_space_size"] >= 100
        assert spec.params["knob_source"] in {"probe", "policy"}
        assert spec.params["knob_grid_index"] is not None
    assert train
    assert held
    assert not train & held


def test_probe_manifest_is_not_policy_measured():
    manifest = probe_manifest()
    assert manifest["source"] == "probe"
    assert manifest["policy_id"] is None
    assert manifest["k"] is None


def test_write_knobs_records_policy_provenance(tmp_path):
    path = tmp_path / "level_knobs.json"
    rates = {topic: list(PROBE_RATES) for topic in GRIDS}
    payload = write_knobs(
        rates,
        path=path,
        policy_id="gemma4-26b-a4b",
        policy_model="google/gemma-4-26b-a4b-it",
        k=8,
    )
    assert payload["source"] == "policy"
    assert payload["policy_id"] == "gemma4-26b-a4b"
    assert payload["k"] == 8
    assert payload["python_state"]["measured"] is True
    assert "l2_index" in payload["python_state"]
    loaded = load_manifest(path)
    assert loaded["source"] == "policy"
    assert loaded["python_state"]["2"] == default_knobs()["python_state"][2]
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["created_at"]
