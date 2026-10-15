from frontier_lens.reward import REWARD_FN, grade, guessable_templates, last_final


def test_last_final_only():
    raw = "FINAL: 1\nmore text\nFINAL: 9\n"
    assert last_final(raw) == "9"
    reward, ans, unp = grade(raw, "9")
    assert reward == 1
    assert ans == "9"
    assert not unp


def test_hedge_and_multi_answer_score_zero():
    assert grade("FINAL: 3 or 4", "3")[0] == 0
    assert grade("FINAL: 3/4", "3")[0] == 0
    assert grade("FINAL: either 3", "3")[0] == 0
    assert grade("FINAL: 3, 4", "3")[0] == 0
    assert grade("FINAL:  ", "3")[0] == 0
    assert grade("no final here", "3")[0] == 0


def test_exact_match_id_stable():
    assert REWARD_FN == "exact_match_v1"


def test_guessable_small_int_range():
    flagged = guessable_templates({"tiny": ["1", "2", "3"], "wide": ["100", "5000"]})
    assert "tiny" in flagged
    assert "wide" not in flagged
