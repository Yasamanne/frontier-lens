from pathlib import Path

from sqlalchemy import text
from streamlit.testing.v1 import AppTest

from frontier_lens import runstate
from frontier_lens.db import get_engine

PAGE = Path(__file__).resolve().parent.parent / "app" / "pages" / "5_Dataset.py"
POLICY = "gemma4-26b-a4b"


def _seed_run_with_an_approved_candidate_and_no_rl_seeds() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO questions (id, topic, level, band, split, template, params_json, stem, "
            "code, lang, input_files_json, key, numeric, source) VALUES "
            "('q1', 'custom', 2, 'standard', 'train', 'custom', '{}', 'stem', '', 'python', "
            "'{}', '4', 1, 'uploaded')"
        ))
        conn.execute(text(
            "INSERT INTO candidates (id, topic, band, stem, code, lang, input_files_json, "
            "executed_key, status, generator_model, generator_mode, source_fail_ids, created_at, "
            "policy_pass_rate) VALUES ('c1', 'custom', 'standard', 'new stem', 'print(4)', "
            "'python', '{}', '4', 'approved', 'gen', 'similar_augment', '[\"q1\"]', "
            "'2026-10-19T00:00:00+00:00', 0.5)"
        ))
    runstate.insert_run("r", ["q1"], 1, None, {}, status="complete")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO attempts (run_id, model_id, question_id, sample_index, raw_output, "
            "parsed_answer, is_correct, unparsable) VALUES ('r', :m, 'q1', 0, 'FINAL: 5', '5', 0, 0)"
        ), {"m": POLICY})


def test_dataset_page_loads_when_there_are_approved_questions_but_no_rl_seeds():
    """The CSV helpers must be imported for every branch, not only when RL seeds exist."""
    _seed_run_with_an_approved_candidate_and_no_rl_seeds()

    at = AppTest.from_file(str(PAGE), default_timeout=60).run()

    assert not at.exception, [e.value for e in at.exception]
    assert any("SME-approved generated questions" in m.value for m in at.markdown)
