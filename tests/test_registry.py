from frontier_lens.registry import load_yaml, policy_spec


def test_licenses_and_generator_family_split():
    specs = load_yaml()
    assert all(s.license for s in specs)
    policy = {s.family for s in specs if s.role == "policy"}
    assert policy
    # Generator family may overlap policy/reference — emits a warning, not an error
    # (DeepSeek is intentionally used for both generator and reference roles)
    assert policy_spec().id == "gemma4-26b-a4b"
    assert policy_spec().role == "policy"
    assert policy_spec().base_url and "openrouter.ai" in policy_spec().base_url
