"""Shared SQL scope for the Weaknesses analyses (heatmap, pass rates, failure modes).

Queries alias ``attempts`` as ``a`` and ``questions`` as ``q``.
"""

from __future__ import annotations

# Pass this as ``run_id`` to analyse every completed run together instead of one.
POOLED = "*"

# Questions an SME marked broken have a suspect key, so they must not skew rates.
NOT_BROKEN = "COALESCE(q.sme_flagged, 0) != 2"


def run_clause(param: str = "r") -> str:
    """SQL condition on ``a.run_id`` for the bound run id *param* (a run id or ``POOLED``)."""
    return (
        f"(a.run_id = :{param} OR (:{param} = '{POOLED}' AND a.run_id IN "
        "(SELECT id FROM eval_runs WHERE status = 'complete')))"
    )
