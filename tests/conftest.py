import pytest

from frontier_lens import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite file; none may touch the real data/ database.

    Tests that build their own database (e.g. ``tmp_db``) override this after it runs.
    """
    monkeypatch.setattr(db, "DATABASE_URL", f"sqlite:///{tmp_path / 'isolated.db'}")
    monkeypatch.setattr(db, "_engine", None)
    yield
    if db._engine is not None:
        db._engine.dispose()
