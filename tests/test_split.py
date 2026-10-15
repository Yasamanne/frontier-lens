from frontier_lens.seed import assign_splits
from frontier_lens.split import EVAL_TEMPLATES, leakage, rl_split
from frontier_lens.templates import all_specs


def test_template_level_split_has_no_leakage():
    labeled = assign_splits(all_specs())
    pairs = [(spec.template, split) for spec, split in labeled]
    assert not leakage(pairs)
    eval_templates = {spec.template for spec, split in labeled if split == "eval"}
    train_templates = {spec.template for spec, split in labeled if split == "train"}
    assert eval_templates == set(EVAL_TEMPLATES)
    assert not eval_templates & train_templates
    for spec, split in labeled:
        if spec.level == 1:
            assert split == "harness"
        else:
            assert split == rl_split(spec.template, spec.level)
