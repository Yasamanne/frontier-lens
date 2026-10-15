from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.review import (
    audit_decision,
    close_audit_batch,
    draw_audit_sample,
    promote,
    suspicious_reason,
)
from frontier_lens.seed import seed
from frontier_lens.stats import wilson_interval


def test_audit_wilson_upper_bound_accepts_clean_sample():
    # 0 defects / 50: Wilson 95% upper bound is below 10%.
    decision = audit_decision(0, 50, 0.10)
    assert decision["accepted"]
    assert decision["hi"] < 0.10
    _lo, hi = wilson_interval(0, 50)
    assert abs(decision["hi"] - hi) < 1e-12


def test_audit_wilson_upper_bound_rejects_when_hi_crosses_threshold():
    # 5/50 = 10% point estimate; Wilson upper bound exceeds 10%.
    decision = audit_decision(5, 50, 0.10)
    assert not decision["accepted"]
    assert decision["hi"] > 0.10


def test_close_audit_batch_accepts_and_promotes_unsampled():
    seed(force=False)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM candidates WHERE id LIKE 'audit-%'"))
        for i in range(3):
            conn.execute(
                text(
                    """
                    INSERT OR REPLACE INTO candidates (
                        id, topic, band, stem, code, lang, input_files_json, executed_key,
                        numeric, status, generator_model, generator_mode, created_at, rl_split
                    ) VALUES (
                        :id, 'python_state', 'hard', 'what?', 'print(1)\n', 'python', '{}', '1',
                        1, 'pending', 'glm-5.3', 'audit_test', '2026-01-01T00:00:00Z', 'train'
                    )
                    """
                ),
                {"id": f"audit-{i}"},
            )
    batch = draw_audit_sample(n=2, generator_mode="audit_test", threshold=0.84)
    with engine.connect() as conn:
        sampled = conn.execute(
            text(
                "SELECT id FROM candidates WHERE audit_batch_id=:b AND audit_sampled=1"
            ),
            {"b": batch},
        ).scalars().all()
    assert len(sampled) == 2
    for cid in sampled:
        promote(cid, "hard_item", fair=True, novel=True, hard=True)
    decision = close_audit_batch(batch)
    assert decision["accepted"]
    with engine.connect() as conn:
        statuses = dict(
            conn.execute(
                text(
                    "SELECT id, status FROM candidates WHERE id LIKE 'audit-%'"
                )
            ).all()
        )
    assert set(statuses.values()) == {"approved"}


def test_suspicious_conflicting_finals():
    assert (
        suspicious_reason("FINAL: 1\nwait\nFINAL: 9\n", "9") == "conflicting_finals"
    )
    assert suspicious_reason("I guess 3 or 4\nFINAL: 9\n", "9") == "hedged_then_correct"
    assert suspicious_reason("clear reasoning\nFINAL: 9\n", "9") is None
