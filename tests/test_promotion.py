from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.review import promote
from frontier_lens.seed import seed


def test_auto_rejected_cannot_promote():
    seed(force=False)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT OR REPLACE INTO candidates (
                    id, topic, band, stem, code, lang, input_files_json, status, auto_reject_reason,
                    generator_model, generator_mode, created_at
                ) VALUES (
                    'c-unsafe', 'python_state', 'hard', '', 'import os', 'python', '{}',
                    'auto_rejected', 'unsafe', 'glm-5.3', 'one_shot', '2026-01-01T00:00:00Z'
                )
                """
            )
        )
    try:
        promote("c-unsafe", "hard_item", fair=True, novel=True, hard=True)
        raise AssertionError("should have failed")
    except RuntimeError as exc:
        assert "auto-rejected" in str(exc)
