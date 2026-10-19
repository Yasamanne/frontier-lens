from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.sandbox import execute

from .calibration import estimate_pass_rate
from .utils import _extract_json

# Modes whose calibrated candidates must land in the learnable band (0 < pass rate < 1):
# a question the policy always or never solves gives RL no signal, so it is rejected
# before it reaches the SME queue.
BAND_FILTERED_MODES = ("one_shot", "similar_augment")


def _store_candidate(
    raw: str,
    topic: str,
    band: str,
    gen_id: str,
    mode: str,
    fail_ids: list[str],
    *,
    run_id: str | None = None,
    rl_split: str = "train",
    calibrate: bool = True,
    complete_fn=None,
    log_fn=None,
    calibration_k: int | None = None,
) -> str:
    cid = uuid.uuid4().hex[:12]
    status = "pending"
    reason = None
    key = None
    payload = None
    rate = None
    pak = None
    try:
        payload = _extract_json(raw)
    except Exception:
        status, reason = "auto_rejected", "json_parse_error"
    stem = code = lang = ""
    files = {}
    numeric = True
    if payload:
        stem = payload.get("stem") or ""
        code = payload.get("code") or ""
        lang = payload.get("lang") or "python"
        files = payload.get("input_files") or {}
        numeric = bool(payload.get("numeric", True))
        # Infer topic/band from model response when the caller didn't supply them
        if not topic:
            topic = payload.get("topic") or ""
        if not band:
            band = payload.get("band") or ""
        result = execute(lang, code, files, twice=True)
        if log_fn:
            log_fn("sandbox_result", ok=result.ok, reason=result.reason if not result.ok else None)
        if not result.ok:
            status, reason = "auto_rejected", result.reason or "exec_error"
        else:
            key = result.stdout
            if rl_split == "train" and calibrate and status == "pending":
                stats = estimate_pass_rate(
                    stem, key, numeric,
                    complete_fn=complete_fn, log_fn=log_fn,
                    k=calibration_k,
                )
                rate = stats["pass_rate"]
                pak = stats["pass_at_k"]
                if mode in BAND_FILTERED_MODES and not stats["in_learnable_band"]:
                    status, reason = "auto_rejected", "out_of_band"
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO candidates (
                    id, topic, band, stem, code, lang, input_files_json, executed_key, numeric,
                    status, auto_reject_reason, generator_model, generator_mode, source_fail_ids,
                    created_at, policy_pass_rate, pass_at_k, rl_split, run_id
                ) VALUES (
                    :id, :topic, :band, :stem, :code, :lang, :files, :key, :numeric,
                    :status, :reason, :gm, :mode, :fails, :ts, :rate, :pak, :split, :run_id
                )
                """
            ),
            {
                "id": cid,
                "topic": topic,
                "band": band,
                "stem": stem,
                "code": code,
                "lang": lang,
                "files": json.dumps(files),
                "key": key,
                "numeric": int(numeric),
                "status": status,
                "reason": reason,
                "gm": gen_id,
                "mode": mode,
                "fails": json.dumps(fail_ids),
                "ts": datetime.now(timezone.utc).isoformat(),
                "rate": rate,
                "pak": pak,
                "split": rl_split,
                "run_id": run_id,
            },
        )
    print(f"candidate {cid} {status} {reason} split={rl_split} pass_rate={rate}")
    return cid


def _store_hinted_candidate(
    source_qid: str,
    hinted_stem: str,
    code: str,
    lang: str,
    files: dict,
    key: str,
    numeric: bool,
    topic: str,
    band: str,
    gen_id: str,
    stats: dict,
    *,
    run_id: str | None = None,
) -> str:
    """Persist a hinted stem (same code/key as the original question) as a candidate."""
    cid = uuid.uuid4().hex[:12]
    rate = stats["pass_rate"]
    pak = stats["pass_at_k"]
    if stats["in_learnable_band"]:
        status, reason = "pending", None
    else:
        status, reason = "auto_rejected", "hint_not_sufficient"
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO candidates (
                    id, topic, band, stem, code, lang, input_files_json, executed_key, numeric,
                    status, auto_reject_reason, generator_model, generator_mode, source_fail_ids,
                    created_at, policy_pass_rate, pass_at_k, rl_split, run_id
                ) VALUES (
                    :id, :topic, :band, :stem, :code, :lang, :files, :key, :numeric,
                    :status, :reason, :gm, 'light_hint', :fails, :ts, :rate, :pak, 'train', :run_id
                )
                """
            ),
            {
                "id": cid,
                "topic": topic,
                "band": band,
                "stem": hinted_stem,
                "code": code,
                "lang": lang,
                "files": json.dumps(files),
                "key": key,
                "numeric": int(numeric),
                "status": status,
                "reason": reason,
                "gm": gen_id,
                "fails": json.dumps([source_qid]),
                "ts": datetime.now(timezone.utc).isoformat(),
                "rate": rate,
                "pak": pak,
                "run_id": run_id,
            },
        )
    print(f"hint-candidate {cid} source={source_qid} {status} pass_rate={rate:.3f}")
    return cid
