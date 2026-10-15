from frontier_lens.stats import wilson_interval


def test_wilson_n12():
    lo, hi = wilson_interval(8, 12)
    assert 0 <= lo < 8 / 12 < hi <= 1
    assert hi - lo > 0.2
