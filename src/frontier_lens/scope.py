"""Shared SQL scope for the Weaknesses analyses (heatmap, pass rates, failure modes).

Queries alias ``attempts`` as ``a`` and ``questions`` as ``q``.
"""

from __future__ import annotations

# Pass this as ``run_id`` to analyse every completed run together instead of one.
POOLED = "*"
# Append this to a run id to analyse that run plus every completed run. The chosen run is
# always included, even when it is interrupted or failed, so pooling never shows less than it.
POOL_SUFFIX = "+"


def pool_with(run_id: str) -> str:
    """Scope string for *run_id* pooled with all completed runs."""
    return run_id + POOL_SUFFIX

# Questions an SME marked broken have a suspect key, so they must not skew rates.
NOT_BROKEN = "COALESCE(q.sme_flagged, 0) != 2"


# Dataset key of questions that did not come from an uploaded file (seed and generated).
DATASET_BUILTIN = "__builtin__"


def dataset_key_sql() -> str:
    """SQL expression for a question's dataset: its upload file name, or ``DATASET_BUILTIN``."""
    return (
        "CASE WHEN q.source = 'uploaded' "
        f"THEN COALESCE(NULLIF(q.upload_filename, ''), '(unnamed)') ELSE '{DATASET_BUILTIN}' END"
    )


def dataset_clause(param: str = "ds") -> str:
    """SQL condition limiting to one dataset; a NULL bind value means every dataset."""
    return f"(:{param} IS NULL OR {dataset_key_sql()} = :{param})"


def run_clause(param: str = "r") -> str:
    """SQL condition on ``a.run_id`` for the bound run id *param* (a run id or ``POOLED``)."""
    done = "(SELECT id FROM eval_runs WHERE status = 'complete')"
    return (
        f"(a.run_id = :{param} "
        f"OR (:{param} = '{POOLED}' AND a.run_id IN {done}) "
        f"OR (substr(:{param}, -1) = '{POOL_SUFFIX}' AND "
        f"(a.run_id = substr(:{param}, 1, length(:{param}) - 1) OR a.run_id IN {done})))"
    )
