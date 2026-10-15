from frontier_lens.parse import parse_final
from frontier_lens.scoring import is_correct


def test_parse_final_line():
    parsed = parse_final("reasoning\nFINAL: 12\n", numeric=True)
    assert parsed.answer == "12"
    assert not parsed.unparsable
    assert is_correct(parsed, "12")


def test_missing_final_is_unparsable():
    parsed = parse_final("I think it is 12", numeric=True)
    assert parsed.unparsable
    assert not is_correct(parsed, "12")
