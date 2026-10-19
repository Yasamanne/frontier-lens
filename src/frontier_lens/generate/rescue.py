from __future__ import annotations

import json

from sqlalchemy import text

from frontier_lens.config import POLICY_K
from frontier_lens.db import get_engine
from frontier_lens.feedback import context_for as sme_feedback_for
from frontier_lens.parallel import map_with_events
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import sync_models

from .calibration import _calibrate_collect, calibrating_with
from .candidates import _store_hinted_candidate
from .prompts import _LEVEL_INSTRUCTIONS, PROGRESSIVE_HINT_PROMPT
from .utils import _extract_json, generator_kwargs, generator_timeout


def hint_ladder(
    stem: str,
    key: str,
    numeric: bool,
    *,
    provider,
    gen,
    k: int,
    max_hints: int,
    failure_mode: str,
    hint_feedback: str,
    complete_fn=None,
    log_fn=None,
    question_id: str = "",
) -> dict:
    """Add up to *max_hints* progressive hints to *stem*, calibrating the policy after each.

    Level 1 frames the domain, level 2 names the technique, level 3 points at a common
    mistake; each level builds on the previous stem. The generator is told the gold *key*
    (and told not to reveal it). Stops at the first level where the policy lands in the
    learnable band (0 < pass rate < 1).

    Returns ``{"rounds", "rescued", "level", "stem", "stats", "pass_rate"}``: ``stem`` and
    ``stats`` belong to the rescuing level (the last tried level if there was none).
    """
    def _log(event: str, **fields):
        if log_fn:
            log_fn(event, **fields)

    current_stem = stem
    rounds: list[dict] = []
    stats: dict | None = None
    rescued = False
    level_reached: int | None = None

    for level in range(1, max_hints + 1):
        max_level_key = max(_LEVEL_INSTRUCTIONS.keys())
        level_instr = _LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS[max_level_key])

        _log("hint_attempt_start", question_id=question_id, level=level,
             max_hints=max_hints, current_stem_preview=current_stem[:120])

        prompt = PROGRESSIVE_HINT_PROMPT.format(
            gold_key=key,
            failure_mode=failure_mode,
            hint_level=level,
            max_hints=max_hints,
            current_stem=current_stem,
            level_instruction=level_instr,
            sme_feedback=hint_feedback,
        )
        completion = provider.complete(
            prompt, gen.model, skip_correction=True, **generator_kwargs(gen)
        )

        hint_text = ""
        hinted_stem = current_stem
        parse_ok = False
        parse_error = None
        try:
            payload = _extract_json(completion.text)
            hinted_stem = payload.get("stem") or current_stem
            hint_text = payload.get("hint") or ""
            parse_ok = True
        except Exception as exc:
            parse_error = str(exc)
            hint_text = "(parse error — using stem from previous level)"

        _log("hint_generated", question_id=question_id, level=level,
             hint=hint_text, parse_ok=parse_ok, parse_error=parse_error,
             raw_response_preview=(completion.text or "")[:400])

        def _cal_log(event: str, _level=level, **fields):
            _log(event, question_id=question_id, level=_level, **fields)

        stats = _calibrate_collect(
            hinted_stem, key, numeric, k=k, complete_fn=complete_fn, log_fn=_cal_log,
        )
        rounds.append({
            "level": level,
            "hint": hint_text,
            "hinted_stem": hinted_stem,
            "pass_rate": stats["pass_rate"],
            "rl_suitable": stats["in_learnable_band"],
            "responses": stats["responses"],
        })
        _log("hint_attempt_result", question_id=question_id, level=level,
             pass_rate=stats["pass_rate"], n_correct=stats["correct"], n=stats["n"],
             rl_suitable=stats["in_learnable_band"], policy_model=stats["policy_model"])

        current_stem = hinted_stem  # the next level builds on this
        if stats["in_learnable_band"]:
            rescued, level_reached = True, level
            break

    return {
        "rounds": rounds,
        "rescued": rescued,
        "level": level_reached,
        "stem": current_stem,
        "stats": stats,
        "pass_rate": stats["pass_rate"] if stats else 0.0,
    }


def _rescue_with_hints(
    run_id: str,
    model_id: str,
    question_ids: list[str],
    *,
    complete_fn=None,
    log_fn=None,
    calibration_k: int | None = None,
    max_hints: int = 3,
) -> list[dict]:
    """For each pass@k=0 question, apply up to *max_hints* progressive hint levels.

    The generator is given the gold answer so hints can be precisely targeted,
    but is explicitly told not to reveal it.

      L1 → domain / conceptual framing only
      L2 → specific technique or property (not how to apply it)
      L3 → common mistake or edge case the model is likely making

    After each level the question is calibrated with *calibration_k* samples.
    If the model enters the learnable band (0 < pass_rate < 1) the hinted stem
    is stored as a ``pending`` candidate and the loop stops for that question.

    If all levels fail the question data (original + per-level responses) is
    available for SFT export via :func:`generate_rescue_sft_csv`.

    Returns a list of result dicts with keys:
    ``question_id, topic, band, stem_original, gold_key, original_responses,
    hint_rounds, candidate_id, pass_rate, rl_suitable, rescue_level, hint``
    """
    from frontier_lens.failure_analysis import failure_modes

    def _log(event: str, **fields):
        if log_fn:
            log_fn(event, **fields)

    k = calibration_k if calibration_k is not None else POLICY_K

    specs = {m.id: m for m in sync_models()}
    gen = next((m for m in specs.values() if m.role == "generator"), None)
    if gen is None:
        raise RuntimeError("No generator model configured in models.yaml.")
    provider = OpenAICompatibleProvider(
        gen.base_url or "", gen.api_key_env, timeout=generator_timeout(gen)
    )
    engine = get_engine()
    _log("rescue_start", n_questions=len(question_ids), run_id=run_id,
         model_id=model_id, calibration_k=k, max_hints=max_hints)

    def _rescue_one(item: tuple[int, str], _log) -> dict | None:
        """Worker: full hint ladder for one question. Levels stay sequential."""
        i, qid = item
        with engine.connect() as conn:
            q = conn.execute(
                text(
                    "SELECT stem, key, numeric, topic, band, code, lang, input_files_json "
                    "FROM questions WHERE id=:id"
                ),
                {"id": qid},
            ).mappings().first()
        if not q:
            _log("rescue_question_skip", index=i + 1, total=len(question_ids),
                 question_id=qid, reason="not_found")
            return None

        fa = failure_modes(run_id, model_id, q["topic"], q["band"])
        dominant = fa.get("dominant_mode", "unknown")

        _log("rescue_question", index=i + 1, total=len(question_ids),
             question_id=qid, topic=q["topic"], band=q["band"],
             dominant_mode=dominant, gold_key=q["key"],
             stem_preview=(q["stem"] or "")[:120])

        # Fetch original no-hint responses from the contest run for SFT export
        with engine.connect() as conn:
            orig_rows = conn.execute(
                text(
                    "SELECT raw_output, parsed_answer, is_correct FROM attempts "
                    "WHERE run_id=:r AND model_id=:m AND question_id=:q "
                    "ORDER BY sample_index"
                ),
                {"r": run_id, "m": model_id, "q": qid},
            ).mappings().all()
        original_responses = [
            {"raw": r["raw_output"] or "", "parsed": r["parsed_answer"] or "",
             "is_correct": bool(r["is_correct"])}
            for r in orig_rows
        ]

        # SME feedback on earlier rescue hints for this topic (e.g. answer leaks).
        hint_feedback = sme_feedback_for(q["topic"], source_id=qid, hints=True)
        ladder = hint_ladder(
            q["stem"], q["key"], bool(q["numeric"]),
            provider=provider, gen=gen, k=k, max_hints=max_hints,
            failure_mode=dominant, hint_feedback=hint_feedback,
            complete_fn=complete_fn, log_fn=_log, question_id=qid,
        )
        hint_rounds = ladder["rounds"]
        rescued = ladder["rescued"]
        rescue_level = ladder["level"]
        final_pass_rate = ladder["pass_rate"]
        candidate_id: str | None = None
        if rescued:
            files = json.loads(q["input_files_json"] or "{}")
            candidate_id = _store_hinted_candidate(
                source_qid=qid,
                hinted_stem=ladder["stem"],
                code=q["code"] or "", lang=q["lang"] or "python", files=files,
                key=q["key"], numeric=bool(q["numeric"]),
                topic=q["topic"], band=q["band"],
                gen_id=gen.id, stats=ladder["stats"],
                run_id=run_id,
            )
            _log("hint_result", question_id=qid, candidate_id=candidate_id,
                 pass_rate=ladder["stats"]["pass_rate"], n_correct=ladder["stats"]["correct"],
                 n=ladder["stats"]["n"], rl_suitable=True, rescue_level=rescue_level)
        else:
            _log("hint_result", question_id=qid, candidate_id=None,
                 pass_rate=final_pass_rate, rl_suitable=False,
                 rescue_level=None, n_hints_tried=len(hint_rounds))

        result = {
            "question_id": qid,
            "topic": q["topic"], "band": q["band"],
            "stem_original": q["stem"], "gold_key": q["key"],
            "original_responses": original_responses,
            "hint_rounds": hint_rounds,
            "candidate_id": candidate_id,
            "pass_rate": final_pass_rate,
            "rl_suitable": rescued,
            "rescue_level": rescue_level,
            "hint": hint_rounds[-1]["hint"] if hint_rounds else "",
        }
        # Lets a background job save each question as it finishes (resume).
        _log("rescue_question_done", question_id=qid, result=result)
        return result

    # Questions are independent — run them concurrently; events reach log_fn
    # on this thread.
    results = [
        r for r in map_with_events(_rescue_one, enumerate(question_ids), log_fn)
        if r is not None
    ]

    n_rescued = sum(1 for r in results if r["rl_suitable"])
    _log("rescue_end", n_total=len(question_ids),
         n_rescued=n_rescued, n_rejected=len(question_ids) - n_rescued)
    return results


def rescue_with_hints(run_id: str, model_id: str, question_ids: list[str], **kwargs) -> list[dict]:
    """Rescue pass@k=0 questions, calibrating every hint level on *model_id*; see ``_rescue_with_hints``."""
    with calibrating_with(model_id):
        return _rescue_with_hints(run_id, model_id, question_ids, **kwargs)


def generate_rescue_sft_csv(results: list[dict]) -> str:
    """Generate an SFT CSV for questions that could not be rescued.

    Each row = one unrescuable question. Columns:
      question_id, topic, band, gold_key, stem_original,
      original_response_1..k, original_correct_1..k,   ← from the contest run
      hint_1_text, stem_hint_1,
      hint_1_response_1..k, hint_1_correct_1..k,
      hint_2_text, stem_hint_2,  ...  (up to max hints attempted)
      rescue_level_reached
    """
    import csv as _csv
    import io as _io

    unrescued = [r for r in results if not r.get("rl_suitable")]
    if not unrescued:
        return ""

    # Determine max k and max hint levels from data
    max_k = max(
        (len(r.get("original_responses", [])) for r in unrescued), default=1
    )
    max_levels = max(
        (len(r.get("hint_rounds", [])) for r in unrescued), default=0
    )

    meta_cols = ["question_id", "topic", "band", "gold_key", "stem_original"]
    orig_cols = [f"original_response_{i+1}" for i in range(max_k)] + \
                [f"original_correct_{i+1}" for i in range(max_k)]
    hint_cols: list[str] = []
    for lvl in range(1, max_levels + 1):
        hint_cols += [f"hint_{lvl}_text", f"stem_hint_{lvl}"]
        hint_cols += [f"hint_{lvl}_response_{i+1}" for i in range(max_k)]
        hint_cols += [f"hint_{lvl}_correct_{i+1}" for i in range(max_k)]
    all_cols = meta_cols + orig_cols + hint_cols + ["rescue_level_reached"]

    out = _io.StringIO()
    writer = _csv.DictWriter(out, fieldnames=all_cols, extrasaction="ignore")
    writer.writeheader()

    for r in unrescued:
        row: dict = {
            "question_id": r["question_id"],
            "topic": r.get("topic", ""),
            "band": r.get("band", ""),
            "gold_key": r.get("gold_key", ""),
            "stem_original": r.get("stem_original", ""),
            "rescue_level_reached": len(r.get("hint_rounds", [])),
        }
        # Original (no-hint) responses
        for i, resp in enumerate(r.get("original_responses", [])[:max_k]):
            row[f"original_response_{i+1}"] = resp.get("raw", "")
            row[f"original_correct_{i+1}"] = resp.get("is_correct", False)
        # Per-hint-level responses
        for lvl_idx, rnd in enumerate(r.get("hint_rounds", []), start=1):
            row[f"hint_{lvl_idx}_text"] = rnd.get("hint", "")
            row[f"stem_hint_{lvl_idx}"] = rnd.get("hinted_stem", "")
            for i, resp in enumerate(rnd.get("responses", [])[:max_k]):
                row[f"hint_{lvl_idx}_response_{i+1}"] = resp.get("raw", "")
                row[f"hint_{lvl_idx}_correct_{i+1}"] = resp.get("is_correct", False)
        writer.writerow(row)

    return out.getvalue()
