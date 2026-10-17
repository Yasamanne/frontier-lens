from frontier_lens.templates import all_specs

_BANNED = ("mutable default", "alias", "method resolution", "mro", "gotcha", "pitfall")


def test_prose_items_hide_verifier_code():
    seen = {"constraint_counting": 0, "multi_step_units": 0}
    for spec in all_specs():
        if spec.level < 2 or spec.topic not in seen:
            continue
        seen[spec.topic] += 1
        assert spec.code.strip() not in spec.stem
        assert "product(" not in spec.stem
        assert "ROUND_HALF_UP" not in spec.stem
        assert "Decimal(" not in spec.stem
    assert all(n > 0 for n in seen.values())


def test_shown_items_include_the_program():
    for spec in all_specs():
        if spec.level < 2:
            continue
        if spec.topic in {"python_state", "shell_pipelines"}:
            assert spec.code.strip() in spec.stem
        low = spec.stem.lower()
        for word in _BANNED:
            assert word not in low
