"""Shared SQL scope for the Gap Analysis queries (heatmap, pass rates, failure modes).

Queries alias ``attempts`` as ``a`` and ``questions`` as ``q``.

A *scope* string says which runs to read:

- ``"<run id>"``    that run only
- ``"<run id>+"``   that run plus every completed run (see :func:`pool_with`)
- ``"*"``           every completed run

and an optional *dataset* key limits the questions to one file (or the built-in questions).
"""

from __future__ import annotations

POOLED = "*"
POOL_SUFFIX = "+"

# Questions an SME marked broken have a suspect key, so they must not skew rates.
NOT_BROKEN = "COALESCE(q.sme_flagged, 0) != 2"

# Dataset key of questions that did not come from an uploaded file (seed and generated).
DATASET_BUILTIN = "__builtin__"


def pool_with(run_id: str) -> str:
    """Scope for *run_id* pooled with all completed runs.

    The chosen run is always included, even when it is interrupted or failed, so pooling
    never shows less than the run on its own.
    """
    return run_id + POOL_SUFFIX


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
    """SQL condition on ``a.run_id`` for the bound scope string *param* (see the module docstring)."""
    done = "(SELECT id FROM eval_runs WHERE status = 'complete')"
    return (
        f"(a.run_id = :{param} "
        f"OR (:{param} = '{POOLED}' AND a.run_id IN {done}) "
        f"OR (substr(:{param}, -1) = '{POOL_SUFFIX}' AND "
        f"(a.run_id = substr(:{param}, 1, length(:{param}) - 1) OR a.run_id IN {done})))"
    )
