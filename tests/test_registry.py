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


def test_gemma2_27b_is_a_selectable_primary_model_with_a_cap_that_fits_its_context():
    specs = {s.id: s for s in load_yaml()}
    m = specs["gemma2-27b"]
    assert (m.model, m.role, m.family, m.thinking) == ("google/gemma-2-27b-it", "reference", "gemma", False)
    assert m.max_tokens == 4096   # its context window is 8,192 tokens in total
    # the Evaluate page offers Gemma-family policy and reference models as primary models
    primary = [s.id for s in specs.values() if s.role in {"policy", "reference"} and s.family == "gemma"]
    assert {"gemma4-26b-a4b", "gemma4-31b", "gemma2-27b"} <= set(primary)
