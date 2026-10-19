from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from frontier_lens.config import AUDIT_MAX_DEFECT_HI, AUDIT_SAMPLE_N, SME_NAME
from frontier_lens.db import get_engine
from frontier_lens.ids import item_id
from frontier_lens.parse import normalize
from frontier_lens.reward import FINAL_RE_V2, HEDGE_RE, REWARD_FN, last_final
from frontier_lens.sandbox import execute
from frontier_lens.stats import wilson_interval

DEFECT_LABELS = {"rejected", "broken_item", "trivial"}


def audit_decision(
    defects: int, n: int, threshold: float = AUDIT_MAX_DEFECT_HI
) -> dict:
    lo, hi = wilson_interval(defects, n)
    return {
        "n": n,
        "defects": defects,
        "defect_rate": (defects / n) if n else 0.0,
        "lo": lo,
        "hi": hi,
        "accepted": bool(n > 0 and hi < threshold),
        "threshold": threshold,
    }


def draw_audit_sample(
    n: int = AUDIT_SAMPLE_N,
    *,
    generator_mode: str | None = None,
    rl_split: str = "train",
    threshold: float = AUDIT_MAX_DEFECT_HI,
) -> str:
    """Tag a pending pool with a batch id and mark a random sample for SME labels."""
    batch_id = uuid.uuid4().hex[:12]
    engine = get_engine()
    with engine.begin() as conn:
        params: dict = {"split": rl_split}
        mode_sql = ""
        if generator_mode:
            mode_sql = "AND generator_mode=:mode"
            params["mode"] = generator_mode
        pool = conn.execute(
            text(
                f"""
                SELECT id FROM candidates
                WHERE status='pending' AND COALESCE(rl_split, 'train')=:split
                {mode_sql}
                """
            ),
            params,
        ).scalars().all()
        if not pool:
            raise RuntimeError("no pending candidates to audit")
        for cid in pool:
            conn.execute(
                text(
                    "UPDATE candidates SET audit_batch_id=:b, audit_sampled=0 WHERE id=:id"
                ),
                {"b": batch_id, "id": cid},
            )
        sample = list(pool)
        if len(sample) > n:
            from random import Random

            rng = Random(batch_id)
            sample = rng.sample(sample, n)
        for cid in sample:
            conn.execute(
                text("UPDATE candidates SET audit_sampled=1 WHERE id=:id"),
                {"id": cid},
            )
        conn.execute(
            text(
                """
                INSERT INTO audit_batches (
                    id, created_at, n, defects, defect_lo, defect_hi, accepted, threshold
                ) VALUES (
                    :id, :ts, :n, 0, 0, 1, 0, :thr
                )
                """
            ),
            {
                "id": batch_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "n": len(sample),
                "thr": threshold,
            },
        )
    return batch_id


def sampled_pending(batch_id: str) -> list[str]:
    engine = get_engine()
    with engine.connect() as conn:
        return list(
            conn.execute(
                text(
                    """
                    SELECT id FROM candidates
                    WHERE audit_batch_id=:b AND audit_sampled=1 AND status='pending'
                    """
                ),
                {"b": batch_id},
            ).scalars().all()
        )


def close_audit_batch(batch_id: str, *, promote_pool: bool = True) -> dict:
    """Wilson 95% CI on defect rate. Accept the batch if the upper bound is below threshold."""
    engine = get_engine()
    with engine.begin() as conn:
        labels = conn.execute(
            text(
                """
                SELECT c.id, r.label
                FROM candidates c
                JOIN reviews r ON r.candidate_id = c.id
                WHERE c.audit_batch_id=:b AND c.audit_sampled=1
                """
            ),
            {"b": batch_id},
        ).mappings().all()
        batch = conn.execute(
            text("SELECT n, threshold FROM audit_batches WHERE id=:id"),
            {"id": batch_id},
        ).mappings().one()
        if len(labels) < batch["n"]:
            raise RuntimeError("audit sample is not fully labeled")
        defects = sum(1 for r in labels if r["label"] in DEFECT_LABELS)
        decision = audit_decision(defects, batch["n"], batch["threshold"])
        conn.execute(
            text(
                """
                UPDATE audit_batches
                SET defects=:d, defect_lo=:lo, defect_hi=:hi, accepted=:ok
                WHERE id=:id
                """
            ),
            {
                "id": batch_id,
                "d": defects,
                "lo": decision["lo"],
                "hi": decision["hi"],
                "ok": int(decision["accepted"]),
            },
        )
        unlabeled = conn.execute(
            text(
                """
                SELECT id FROM candidates
                WHERE audit_batch_id=:b AND status='pending'
                  AND COALESCE(audit_sampled, 0)=0
                """
            ),
            {"b": batch_id},
        ).scalars().all()
    if decision["accepted"] and promote_pool:
        for cid in unlabeled:
            promote(
                cid,
                "hard_item",
                fair=True,
                novel=True,
                hard=True,
                note="batch accepted by audit",
                kind="audit_auto",
            )
    return decision


def promote(
    candidate_id: str,
    label: str,
    *,
    fair: bool,
    novel: bool,
    hard: bool,
    note: str = "",
    kind: str = "audit",
    edited_stem: str | None = None,
    edited_code: str | None = None,
    reasons: list[str] | None = None,
) -> None:
    """Approve or reject a candidate.

    If *edited_stem* or *edited_code* are provided (non-empty and different from
    the stored values), the candidate row is updated before promotion so the
    edited version is what lands in the questions table.  The sandbox is always
    re-run on the final code so the gold key is fresh.

    *reasons* (codes from ``feedback.REASONS``), *note* and the pre-edit stem are
    kept on the review and fed back to the generators (``feedback.context_for``).
    """
    engine = get_engine()
    with engine.begin() as conn:
        cand = conn.execute(
            text("SELECT * FROM candidates WHERE id=:id"), {"id": candidate_id}
        ).mappings().first()
        if not cand:
            raise RuntimeError("unknown candidate")
        if cand["status"] == "auto_rejected":
            raise RuntimeError("auto-rejected candidates cannot be promoted")

        # Apply SME edits to the candidate row before proceeding.
        # Falls back to the stored value if the edit is empty or unchanged.
        stem_to_use = edited_stem.strip() if edited_stem and edited_stem.strip() else cand["stem"]
        code_to_use = edited_code.strip() if edited_code and edited_code.strip() else cand["code"]
        if stem_to_use != cand["stem"] or code_to_use != cand["code"]:
            conn.execute(
                text("UPDATE candidates SET stem=:s, code=:c WHERE id=:id"),
                {"s": stem_to_use, "c": code_to_use, "id": candidate_id},
            )

        conn.execute(
            text(
                """
                INSERT INTO reviews (
                    candidate_id, fair, novel, hard_for_right_reason, label, note,
                    reviewer, decided_at, kind, reasons, original_stem
                )
                VALUES (
                    :id, :fair, :novel, :hard, :label, :note, :who, :ts, :kind,
                    :reasons, :original_stem
                )
                """
            ),
            {
                "id": candidate_id,
                "fair": int(fair),
                "novel": int(novel),
                "hard": int(hard),
                "label": label,
                "note": note,
                "who": SME_NAME,
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "reasons": json.dumps(reasons or []),
                "original_stem": cand["stem"] if stem_to_use != cand["stem"] else None,
            },
        )
        if label in DEFECT_LABELS:
            conn.execute(
                text("UPDATE candidates SET status='rejected' WHERE id=:id"),
                {"id": candidate_id},
            )
            return
        files = json.loads(cand["input_files_json"] or "{}")
        result = execute(cand["lang"], code_to_use, files, twice=True, single_line=True)
        if not result.ok:
            conn.execute(
                text(
                    "UPDATE candidates SET status='auto_rejected', auto_reject_reason=:r WHERE id=:id"
                ),
                {"id": candidate_id, "r": result.reason},
            )
            return
        qid = item_id("generated", {"cid": candidate_id}, code_to_use, files)
        split = cand["rl_split"] or "train"
        conn.execute(
            text(
                """
                INSERT OR REPLACE INTO questions (
                    id, topic, level, band, split, template, params_json, stem, code, lang,
                    input_files_json, key, numeric, source, generator_model, generator_mode,
                    source_fail_ids, reward_fn, audit_batch_id
                ) VALUES (
                    :id, :topic, :level, :band, :split, 'generated', :params, :stem, :code, :lang,
                    :files, :key, :numeric, 'generated', :gm, :mode, :fails, :rfn, :ab
                )
                """
            ),
            {
                "id": qid,
                "topic": cand["topic"],
                "level": 3 if cand["band"] == "hard" else 2,
                "band": cand["band"],
                "split": split,
                "params": json.dumps({"candidate": candidate_id}),
                "stem": stem_to_use,
                "code": code_to_use,
                "lang": cand["lang"],
                "files": cand["input_files_json"],
                "key": result.stdout,
                "numeric": cand["numeric"],
                "gm": cand["generator_model"],
                "mode": cand["generator_mode"],
                "fails": cand["source_fail_ids"],
                "rfn": REWARD_FN,
                "ab": cand["audit_batch_id"],
            },
        )
        conn.execute(
            text("UPDATE candidates SET status='approved', executed_key=:k WHERE id=:id"),
            {"id": candidate_id, "k": result.stdout},
        )


def suspicious_reason(raw: str, key: str, numeric: bool = True) -> str | None:
    """Right FINAL but the trace looks like a lucky guess or a hedge."""
    finals = FINAL_RE_V2.findall(raw or "")
    if len(finals) >= 2:
        norms = {normalize(f, numeric) for f in finals}
        if len(norms) > 1:
            return "conflicting_finals"
    last = last_final(raw)
    if not last:
        return None
    before = (raw or "")[: (raw or "").lower().rfind("final:")]
    if HEDGE_RE.search(before):
        return "hedged_then_correct"
    return None


def flag_suspicious(run_id: str, model_id: str) -> int:
    engine = get_engine()
    inserted = 0
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT a.question_id, a.sample_index, a.raw_output, q.key, q.numeric
                FROM attempts a JOIN questions q ON q.id=a.question_id
                WHERE a.run_id=:r AND a.model_id=:m AND a.is_correct=1
                """
            ),
            {"r": run_id, "m": model_id},
        ).mappings().all()
        for r in rows:
            reason = suspicious_reason(r["raw_output"] or "", r["key"], bool(r["numeric"]))
            if not reason:
                continue
            conn.execute(
                text(
                    """
                    INSERT OR IGNORE INTO suspicious_samples (
                        run_id, question_id, model_id, sample_index, reason, raw_output, status
                    ) VALUES (
                        :r, :q, :m, :s, :reason, :raw, 'pending'
                    )
                    """
                ),
                {
                    "r": run_id,
                    "q": r["question_id"],
                    "m": model_id,
                    "s": r["sample_index"],
                    "reason": reason,
                    "raw": r["raw_output"],
                },
            )
            inserted += 1
    return inserted
