from frontier_lens.registry import load_yaml


def test_licenses_and_family_split():
    specs = load_yaml()
    assert all(s.license for s in specs)
    ev = {s.family for s in specs if s.role == "evaluated"}
    ge = {s.family for s in specs if s.role == "generator"}
    assert not ev & ge
