"""SME review feedback, turned into prompt context for the generators.

When an SME approves or rejects a generated candidate (``review.promote``) the
review keeps the label, structured reasons (:data:`REASONS`), a free-text note
and, if the SME edited the stem, the original. :func:`context_for` condenses
recent reviews for a topic into a block the generator and the AI reviewer see,
so the same mistakes are not repeated and output tracks what reviewers accept.

Not used for held-out eval generation: reviewers see policy pass rates, so
their notes can carry policy signal that the eval split must not use.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from sqlalchemy import bindparam, text

from frontier_lens.db import get_engine
from frontier_lens.review import DEFECT_LABELS

# Reason codes shown as checkboxes in the SME queue; the text is what the
# generator reads.
REASONS: dict[str, str] = {
    "ambiguous": "Ambiguous or underspecified stem",
    "wrong_key": "Gold answer is wrong or does not match what the stem asks",
    "multiple_answers": "More than one defensible answer",
    "too_easy": "Too easy for the band",
    "too_hard": "Unreasonably hard or needs knowledge the stem does not give",
    "not_novel": "Too close to the seed or other variants (e.g. only numbers changed)",
    "off_topic": "Tests a different skill than the seed or topic",
    "answer_leak": "Stem or hint gives away the answer",
    "unnatural": "Contrived, unnatural or confusing wording",
    "format": "Answer format unclear (units, rounding, case, separators)",
}

# Human reviews only: "audit_auto" rows are bulk promotions, not judgements.
_HUMAN_KINDS = ("sme_queue", "audit")
# Rescue candidates are hinted stems; their feedback is about hints.
HINT_MODES = ("light_hint",)

NO_FEEDBACK = "(no SME feedback yet)"


def _clip(s: str | None, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _reason_texts(raw: str | None) -> list[str]:
    try:
        codes = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [REASONS.get(c, c) for c in codes if c]


def _source_ids(raw: str | None) -> list[str]:
    try:
        return [str(x) for x in json.loads(raw or "[]")]
    except json.JSONDecodeError:
        return []


def _load(topic: str | None, hints: bool, limit: int) -> list[dict]:
    mode_sql = "c.generator_mode IN :modes" if hints else (
        "COALESCE(c.generator_mode, '') NOT IN :modes"
    )
    params: dict = {"modes": HINT_MODES, "kinds": _HUMAN_KINDS, "lim": limit}
    topic_sql = ""
    if topic:
        topic_sql = "AND c.topic = :topic"
        params["topic"] = topic
    stmt = text(
        f"""
        SELECT r.label, r.note, r.reasons, r.original_stem, r.decided_at,
               c.stem, c.band, c.source_fail_ids
        FROM reviews r JOIN candidates c ON c.id = r.candidate_id
        WHERE r.kind IN :kinds AND {mode_sql} {topic_sql}
        ORDER BY r.decided_at DESC
        LIMIT :lim
        """
    ).bindparams(bindparam("modes", expanding=True), bindparam("kinds", expanding=True))
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(stmt, params).mappings().all()]


def _why(row: dict) -> str:
    parts = _reason_texts(row["reasons"])
    if row["label"] not in ("rejected", None, ""):
        parts.insert(0, f"label {row['label']}")
    if row["note"]:
        parts.append(f'note: "{_clip(row["note"], 240)}"')
    return "; ".join(parts) or "no reason given"


def _fixed(row: dict) -> str:
    reasons = _reason_texts(row["reasons"])
    return f" (fixed: {'; '.join(reasons).lower()})" if reasons else ""


def _liked(row: dict) -> str:
    parts = [f'"{_clip(row["note"], 200)}"'] if row["note"] else []
    reasons = _reason_texts(row["reasons"])
    if reasons:
        parts.append(f"approved despite: {'; '.join(reasons).lower()}")
    return " — ".join(parts)


def context_for(
    topic: str | None,
    *,
    source_id: str | None = None,
    hints: bool = False,
    max_chars: int = 2500,
) -> str:
    """SME feedback for prompts: same-seed rejections, common reasons, recent
    rejections, SME rewrites and approved examples, newest first, within *max_chars*.

    *hints* selects feedback on rescue hints instead of on generated questions.
    """
    rows = _load(topic, hints, limit=200)
    if not rows:
        return NO_FEEDBACK
    rejected = [r for r in rows if r["label"] in DEFECT_LABELS]
    approved = [r for r in rows if r["label"] not in DEFECT_LABELS]
    same_seed, others = [], []
    for r in rejected:
        is_same = source_id is not None and source_id in _source_ids(r["source_fail_ids"])
        (same_seed if is_same else others).append(r)
    scope = f"topic {topic}" if topic else "all topics"
    what = "hints" if hints else "questions"

    sections: list[tuple[str, list[str]]] = []
    if same_seed:
        sections.append((
            "Rejected variants of THIS seed (do not repeat these problems):",
            [f'- "{_clip(r["stem"], 200)}" — {_why(r)}' for r in same_seed[:5]],
        ))
    counts = Counter(t for r in rejected for t in _reason_texts(r["reasons"]))
    if counts:
        sections.append((
            f"Most common reasons SMEs reject {what} for {scope}:",
            [f"- {reason} (×{n})" for reason, n in counts.most_common(5)],
        ))
    if others:
        sections.append((
            f"Recent rejections for {scope}:",
            [f'- [{r["band"]}] "{_clip(r["stem"], 160)}" — {_why(r)}' for r in others[:5]],
        ))
    edits = [r for r in approved if r["original_stem"] and r["original_stem"] != r["stem"]]
    if edits:
        sections.append((
            "SME rewrites (what was generated → what the SME approved instead):",
            [f'- "{_clip(r["original_stem"], 150)}" → "{_clip(r["stem"], 150)}"' + _fixed(r)
             + (f' — "{_clip(r["note"], 160)}"' if r["note"] else "")
             for r in edits[:3]],
        ))
    # Comments, or issues the SME accepted, on approvals the rewrites above do not cover.
    liked = [r for r in approved if r not in edits and (r["note"] or _reason_texts(r["reasons"]))]
    if liked:
        sections.append((
            "Approved, with reviewer comments (aim for this):",
            [f'- "{_clip(r["stem"], 140)}" — {_liked(r)}' for r in liked[:3]],
        ))
    return _fit(sections, max_chars)


def _fit(sections: list[tuple[str, list[str]]], max_chars: int) -> str:
    """Join sections, dropping lines that would push the text past *max_chars*."""
    out = ""
    for header, lines in sections:
        sep = "\n\n" if out else ""
        block = ""
        for line in lines:
            candidate = f"{block}\n{line}" if block else f"{header}\n{line}"
            if len(out) + len(sep) + len(candidate) > max_chars:
                break
            block = candidate
        if block:
            out += sep + block
    return out or NO_FEEDBACK
