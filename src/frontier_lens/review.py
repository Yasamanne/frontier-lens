from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import text

from frontier_lens.config import SME_NAME
from frontier_lens.db import get_engine
from frontier_lens.ids import item_id
from frontier_lens.sandbox import execute


def promote(candidate_id: str, label: str, *, fair: bool, novel: bool, hard: bool, note: str = "") -> None:
    engine = get_engine()
    with engine.begin() as conn:
        cand = conn.execute(
            text("SELECT * FROM candidates WHERE id=:id"), {"id": candidate_id}
        ).mappings().first()
        if not cand:
            raise RuntimeError("unknown candidate")
        if cand["status"] == "auto_rejected":
            raise RuntimeError("auto-rejected candidates cannot be promoted")
        conn.execute(
            text(
                """
                INSERT INTO reviews (candidate_id, fair, novel, hard_for_right_reason, label, note, reviewer, decided_at)
                VALUES (:id, :fair, :novel, :hard, :label, :note, :who, :ts)
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
            },
        )
        if label in {"rejected", "broken_item", "trivial"}:
            conn.execute(
                text("UPDATE candidates SET status='rejected' WHERE id=:id"),
                {"id": candidate_id},
            )
            return
        files = json.loads(cand["input_files_json"] or "{}")
        result = execute(cand["lang"], cand["code"], files, twice=True)
        if not result.ok:
            conn.execute(
                text("UPDATE candidates SET status='auto_rejected', auto_reject_reason=:r WHERE id=:id"),
                {"id": candidate_id, "r": result.reason},
            )
            return
        qid = item_id("generated", {"cid": candidate_id}, cand["code"], files)
        conn.execute(
            text(
                """
                INSERT OR REPLACE INTO questions (
                    id, topic, level, band, split, template, params_json, stem, code, lang,
                    input_files_json, key, numeric, source, generator_model, generator_mode, source_fail_ids
                ) VALUES (
                    :id, :topic, :level, :band, 'generated', 'generated', :params, :stem, :code, :lang,
                    :files, :key, :numeric, 'generated', :gm, :mode, :fails
                )
                """
            ),
            {
                "id": qid,
                "topic": cand["topic"],
                "level": 3 if cand["band"] == "hard" else 2,
                "band": cand["band"],
                "params": json.dumps({"candidate": candidate_id}),
                "stem": cand["stem"],
                "code": cand["code"],
                "lang": cand["lang"],
                "files": cand["input_files_json"],
                "key": result.stdout,
                "numeric": cand["numeric"],
                "gm": cand["generator_model"],
                "mode": cand["generator_mode"],
                "fails": cand["source_fail_ids"],
            },
        )
        conn.execute(
            text("UPDATE candidates SET status='approved', executed_key=:k WHERE id=:id"),
            {"id": candidate_id, "k": result.stdout},
        )
