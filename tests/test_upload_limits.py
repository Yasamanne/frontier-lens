from frontier_lens.upload import MAX_FILE_BYTES, MAX_ROWS, validate


def test_oversized_file_gets_a_plain_error_in_megabytes():
    result = validate(b"x" * (MAX_FILE_BYTES + 1), "big.csv")

    assert not result.ok
    assert len(result.errors) == 1
    assert "10.0 MB" in result.errors[0] or "10 MB" in result.errors[0]
    assert "maximum is 10 MB" in result.errors[0]


def test_too_many_rows_is_a_clear_error():
    data = b"stem,key\n" + b"".join(f"q{i},{i}\n".encode() for i in range(MAX_ROWS + 1))

    result = validate(data, "many.csv")

    assert not result.ok
    assert any("Too many rows" in e and "10,000" in e for e in result.errors)


def test_streamlit_upload_limit_matches_the_app_limit():
    import tomllib
    from pathlib import Path

    cfg = tomllib.loads(Path(".streamlit/config.toml").read_text())

    assert cfg["server"]["maxUploadSize"] * 1024 * 1024 == MAX_FILE_BYTES
