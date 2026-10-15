from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.sandbox import execute
from frontier_lens.seed import seed


def test_reexecute_seed_keys():
    seed(force=True)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT lang, code, input_files_json, key FROM questions WHERE source='seed'")
        ).mappings().all()
    assert len(rows) == 104
    import json

    for row in rows:
        files = json.loads(row["input_files_json"] or "{}")
        result = execute(row["lang"], row["code"], files, twice=True)
        assert result.ok, result.reason
        assert result.stdout == row["key"]
