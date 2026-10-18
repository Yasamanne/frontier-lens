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


def test_every_topic_has_held_out_eval_items():
    from frontier_lens.templates import TOPICS

    labeled = assign_splits(all_specs())
    for topic in TOPICS:
        splits = {split for spec, split in labeled if spec.topic == topic}
        assert {"harness", "train", "eval"} <= splits, f"{topic} splits: {sorted(splits)}"
    for spec, split in labeled:
        # Eval-seed items (the {topic}_eval template) must never land in train.
        assert (split == "eval") == (spec.template.endswith("_eval") and spec.level > 1)
