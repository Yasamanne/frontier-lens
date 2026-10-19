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


def test_database_failures_get_plain_messages():
    from frontier_lens.upload import friendly_db_error

    assert "busy" in friendly_db_error(Exception("database is locked"))
    assert "disk is full" in friendly_db_error(Exception("database or disk is full"))
    assert "database error" in friendly_db_error(Exception("something else"))


def test_seed_page_shows_a_stored_import_error_as_an_alert():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    page = Path(__file__).resolve().parent.parent / "app" / "pages" / "1_Seed.py"
    at = AppTest.from_file(str(page), default_timeout=60)
    at.session_state["upload_import_error"] = "The database is busy: wait and retry."
    at.run()

    assert not at.exception
    assert any("The database is busy" in e.value for e in at.error)
