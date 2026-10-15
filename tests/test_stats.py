from frontier_lens.stats import (
    headroom,
    in_learnable_band,
    pass_at_k,
    pass_rate,
    wilson_interval,
)


def test_wilson_n12():
    lo, hi = wilson_interval(8, 12)
    assert 0 <= lo < 8 / 12 < hi <= 1
    assert hi - lo > 0.2


def test_pass_at_k_known_values():
    assert pass_at_k(n=8, c=0, k=8) == 0.0
    assert pass_at_k(n=8, c=8, k=8) == 1.0
    assert pass_at_k(n=8, c=1, k=1) == 1 / 8
    # n=8, c=4, k=8 -> n-c=4 < 8 so 1.0
    assert pass_at_k(n=8, c=4, k=8) == 1.0
    # n=8, c=2, k=1 -> 2/8
    assert abs(pass_at_k(8, 2, 1) - 0.25) < 1e-12


def test_learnable_band_edges():
    assert not in_learnable_band(0.0)
    assert not in_learnable_band(1.0)
    assert in_learnable_band(0.125)
    assert in_learnable_band(0.875)


def test_headroom_zero_when_k_is_1():
    assert headroom(n=8, c=3, k=1) == 0.0
    assert headroom(n=8, c=3, k=8) >= 0.0
