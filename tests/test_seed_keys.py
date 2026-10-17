import json

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.sandbox import execute
from frontier_lens.seed import seed


def test_reexecute_seed_keys():
    seed(force=True)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT lang, code, input_files_json, key, stem, level, params_json "
                "FROM questions WHERE source='seed'"
            )
        ).mappings().all()
    # Seed count depends on the number of templates × levels × items per cell.
    # Assert it's within a reasonable range rather than an exact number that
    # breaks every time a new template is added.
    assert len(rows) >= 100, f"expected ≥100 seed questions, got {len(rows)}"
    for row in rows:
        files = json.loads(row["input_files_json"] or "{}")
        result = execute(row["lang"], row["code"], files, twice=True)
        assert result.ok, result.reason
        assert result.stdout == row["key"]
        if row["level"] >= 2:
            assert row["key"] not in row["stem"]
            params = json.loads(row["params_json"])
            assert params["answer_space_size"] >= 100
