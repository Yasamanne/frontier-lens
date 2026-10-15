from frontier_lens.registry import load_yaml, policy_spec


def test_licenses_and_generator_family_split():
    specs = load_yaml()
    assert all(s.license for s in specs)
    policy = {s.family for s in specs if s.role == "policy"}
    reference = {s.family for s in specs if s.role == "reference"}
    generator = {s.family for s in specs if s.role == "generator"}
    assert policy
    assert not generator & (policy | reference)
    assert policy_spec().id == "gemma4-e4b"
    assert policy_spec().role == "policy"
