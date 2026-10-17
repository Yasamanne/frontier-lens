from __future__ import annotations

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from frontier_lens.config import DATABASE_URL

_engine: Engine | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    family TEXT NOT NULL,
    provider TEXT NOT NULL,
    base_url TEXT,
    model TEXT NOT NULL,
    quant TEXT,
    thinking INTEGER DEFAULT 0,
    license TEXT NOT NULL,
    api_key_env TEXT
);
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    level INTEGER NOT NULL,
    band TEXT NOT NULL,
    split TEXT NOT NULL,
    template TEXT NOT NULL,
    params_json TEXT NOT NULL,
    stem TEXT NOT NULL,
    code TEXT NOT NULL,
    lang TEXT NOT NULL,
    input_files_json TEXT NOT NULL,
    key TEXT NOT NULL,
    numeric INTEGER DEFAULT 1,
    source TEXT NOT NULL,
    generator_model TEXT,
    generator_mode TEXT,
    source_fail_ids TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    item_set_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    k INTEGER NOT NULL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    raw_output TEXT,
    parsed_answer TEXT,
    is_correct INTEGER,
    unparsable INTEGER,
    latency_ms INTEGER,
    tokens_in INTEGER,
    tokens_out INTEGER,
    UNIQUE(run_id, model_id, question_id, sample_index)
);
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    topic TEXT,
    band TEXT,
    stem TEXT,
    code TEXT,
    lang TEXT,
    input_files_json TEXT,
    executed_key TEXT,
    numeric INTEGER DEFAULT 1,
    status TEXT,
    auto_reject_reason TEXT,
    generator_model TEXT,
    generator_mode TEXT,
    source_fail_ids TEXT,
    created_at TEXT,
    policy_pass_rate REAL,
    pass_at_k REAL,
    rl_split TEXT,
    audit_batch_id TEXT
);
CREATE TABLE IF NOT EXISTS audit_batches (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    n INTEGER NOT NULL,
    defects INTEGER NOT NULL,
    defect_lo REAL NOT NULL,
    defect_hi REAL NOT NULL,
    accepted INTEGER NOT NULL,
    threshold REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    fair INTEGER,
    novel INTEGER,
    hard_for_right_reason INTEGER,
    label TEXT,
    note TEXT,
    reviewer TEXT,
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    tool TEXT,
    args_json TEXT,
    result_json TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER
);
CREATE TABLE IF NOT EXISTS suspicious_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    reason TEXT,
    raw_output TEXT,
    status TEXT,
    reviewer TEXT,
    note TEXT,
    decided_at TEXT,
    UNIQUE(run_id, model_id, question_id, sample_index)
);
"""


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            future=True,
            connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
        )

        @event.listens_for(_engine, "connect")
        def _wal(dbapi_conn, _):  # type: ignore[no-untyped-def]
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        with _engine.begin() as conn:
            for stmt in SCHEMA.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            _migrate(conn)
    return _engine


def _migrate(conn) -> None:
    extras = {
        "candidates": [
            ("policy_pass_rate", "REAL"),
            ("pass_at_k", "REAL"),
            ("rl_split", "TEXT"),
            ("audit_batch_id", "TEXT"),
            ("audit_sampled", "INTEGER"),
            ("run_id", "TEXT"),
        ],
        "reviews": [("kind", "TEXT")],
        # Background-run state; see frontier_lens.runstate.
        "eval_runs": [
            ("config_json", "TEXT"),
            ("pid", "INTEGER"),
            ("heartbeat_at", "TEXT"),
            ("phase", "TEXT"),
            ("planned_primary", "INTEGER"),
            ("planned_verifier", "INTEGER"),
            ("error", "TEXT"),
        ],
        "questions": [("reward_fn", "TEXT"), ("audit_batch_id", "TEXT"), ("upload_filename", "TEXT"), ("sme_flagged", "INTEGER"), ("sme_source", "TEXT")],
    }
    for table, cols in extras.items():
        existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
        for name, typ in cols:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {typ}"))
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS suspicious_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                reason TEXT,
                raw_output TEXT,
                status TEXT,
                reviewer TEXT,
                note TEXT,
                decided_at TEXT,
                UNIQUE(run_id, model_id, question_id, sample_index)
            )
            """
        )
    )
    # Background jobs other than evaluation runs; see frontier_lens.jobs.
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS op_jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                pid INTEGER,
                heartbeat_at TEXT,
                phase TEXT,
                planned INTEGER,
                done INTEGER DEFAULT 0,
                error TEXT
            )
            """
        )
    )
    # One row per finished unit of work, appended as it completes.
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS op_job_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                skipped INTEGER NOT NULL DEFAULT 0,
                item_json TEXT NOT NULL
            )
            """
        )
    )
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_op_job_items_job ON op_job_items(job_id)"))
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS audit_batches (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                n INTEGER NOT NULL,
                defects INTEGER NOT NULL,
                defect_lo REAL NOT NULL,
                defect_hi REAL NOT NULL,
                accepted INTEGER NOT NULL,
                threshold REAL NOT NULL
            )
            """
        )
    )
