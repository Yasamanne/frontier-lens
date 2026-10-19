from __future__ import annotations

import json

from sqlalchemy import text

from frontier_lens.config import POLICY_K
from frontier_lens.db import get_engine
from frontier_lens.feedback import NO_FEEDBACK
from frontier_lens.feedback import context_for as sme_feedback_for
from frontier_lens.parallel import call_slot, map_with_events
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import sync_models
from frontier_lens.sandbox import execute

from .candidates import _store_candidate
from .prompts import (
    MAX_FIX_ITERATIONS,
    SIMILAR_AGENT_FIX_MESSAGE,
    SIMILAR_AGENT_REVIEW_SYSTEM,
    SIMILAR_AGENT_SYSTEM,
    SIMILAR_FIX_PROMPT,
    SIMILAR_QUESTION_PROMPT,
    SIMILAR_REVIEW_PROMPT,
)
from .utils import _extract_json, _extract_stem_from_raw, _label_or_infer


def _similar_agent_tools() -> list[dict]:
    """Toolset for the similar-question ReAct agent (no policy-calling tools)."""
    from frontier_lens.agent.tools import BASE_TOOLS
    allowed = {"run_snippet", "check_similarity", "submit_candidate"}
    return [t for t in BASE_TOOLS if t["function"]["name"] in allowed]


def _review_similar_question(
    stem: str,
    code: str,
    lang: str,
    computed_key: str,
    seed_stem: str,
    seed_key: str,
    band: str,
    prior_stems: list[str],
    provider: OpenAICompatibleProvider,
    model: str,
    sme_feedback: str,
) -> dict:
    """Call the reviewer LLM. Returns {score, passed, criterion_scores, issues, fix_instructions}."""
    prior_block = (
        "\n".join(f"  Variant {i+1}: {s[:200]}" for i, s in enumerate(prior_stems))
        if prior_stems else "  (none yet)"
    )
    prompt = SIMILAR_REVIEW_PROMPT.format(
        seed_stem=seed_stem[:800],
        seed_key=seed_key,
        generated_stem=stem[:800],
        lang=lang,
        code=code[:600],
        computed_key=computed_key,
        band=band,
        prior_variants=prior_block,
        sme_feedback=sme_feedback,
    )
    completion = provider.complete(prompt, model, skip_correction=True)
    try:
        payload = _extract_json(completion.text)
        return {
            "score": int(payload.get("score", 0)),
            "passed": bool(payload.get("passed", False)),
            "criterion_scores": payload.get("criterion_scores", {}),
            "issues": payload.get("issues", []),
            "fix_instructions": payload.get("fix_instructions", ""),
        }
    except Exception as exc:
        return {
            "score": 0, "passed": False,
            "criterion_scores": {},
            "issues": [f"Review parse error: {exc}"],
            "fix_instructions": "Regenerate the question from scratch.",
        }


def _run_similar_agent(
    source_q: dict,
    gen,
    client,
    *,
    run_id: str | None = None,
    max_loops: int = 3,
    log_fn=None,
    calibration_k: int | None = None,
    index: int = 1,
    total: int = 1,
) -> str | None:
    """Run one ReAct episode to generate a question similar to *source_q*.

    Returns the ``candidate_id`` if the agent called ``submit_candidate``,
    else ``None``.
    """
    from frontier_lens.agent.tools import handle

    def _log(event: str, **fields):
        if log_fn:
            log_fn(event, source_id=source_q["id"], agent_index=index,
                   agent_total=total, **fields)

    system_prompt = SIMILAR_AGENT_SYSTEM.format(
        topic=_label_or_infer(source_q.get("topic"), "topic"),
        band=_label_or_infer(source_q.get("band"), "band"),
        stem=(source_q.get("stem") or "")[:1200],
        max_loops=max_loops,
        sme_feedback=sme_feedback_for(source_q.get("topic"), source_id=source_q["id"]),
    )
    messages = [{"role": "user", "content": system_prompt}]
    toolset = _similar_agent_tools()

    _log("agent_start", topic=source_q["topic"], band=source_q["band"],
         max_loops=max_loops)

    candidate_id: str | None = None

    for step in range(max_loops):
        try:
            with call_slot():
                resp = client.chat.completions.create(
                    model=gen.model,
                    messages=messages,
                    tools=toolset,
                    timeout=120,
                )
        except Exception as exc:
            _log("agent_error", step=step + 1, error=str(exc))
            break

        msg = resp.choices[0].message

        if not msg.tool_calls:
            _log("agent_no_tool_call", step=step + 1,
                 content=(msg.content or "")[:200])
            break

        messages.append(msg)
        submitted_this_step = False

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            tool_name = call.function.name

            # Enforce topic/band from source question when known; otherwise let
            # the agent's inferred values stand (they'll be read from the payload
            # in _store_candidate).
            if tool_name == "submit_candidate":
                if source_q.get("topic"):
                    args["topic"] = source_q["topic"]
                if source_q.get("band"):
                    args["band"] = source_q["band"]
                args["_mode"] = "similar_augment"

            _log(
                "agent_tool_call",
                step=step + 1,
                tool=tool_name,
                args_preview={k: str(v)[:120] for k, v in args.items()
                               if k != "_mode"},
            )

            def _handle_log(evt: str, _step=step, **f):
                _log(f"handle_{evt}", step=_step + 1, **f)

            result_str = handle(
                tool_name, args,
                gen_id=gen.id,
                # Stored as source_fail_ids, like the other similar modes: links the
                # candidate to its seed (SME queue, same-seed feedback).
                fail_model=source_q["id"],
                run_id=run_id,
                rl_split="train",
                log_fn=_handle_log,
                calibration_k=calibration_k,
            )

            _log(
                "agent_tool_result",
                step=step + 1,
                tool=tool_name,
                result_preview=result_str[:300],
            )

            if tool_name == "submit_candidate":
                try:
                    candidate_id = json.loads(result_str).get("candidate_id")
                except Exception:
                    pass
                submitted_this_step = True

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result_str,
            })

        if submitted_this_step:
            break

    _log("agent_end", steps_used=step + 1, candidate_id=candidate_id)
    return candidate_id


def generate_similar_questions(
    source_questions: list[dict],
    *,
    run_id: str | None = None,
    n_each: int = 1,
    complete_fn=None,
    log_fn=None,
    calibration_k: int | None = None,
) -> list[dict]:
    """For each RL-suitable source question generate n_each similar questions.

    ``source_questions`` is a list of dicts with keys:
    ``id, stem, key, numeric, topic, band``.

    *calibration_k* overrides POLICY_K for the calibration step only.

    *log_fn*, if provided, is called at every observable step::

        log_fn(event: str, **fields)

    Each generated question is calibrated; if RL-suitable it is stored as
    ``status='pending'`` with ``generator_mode='similar_augment'`` so it
    surfaces in the SME queue.

    Returns a list of result dicts::

        {
          "source_id": str,
          "candidate_id": str,
          "status": str,    # "pending" | "auto_rejected"
          "pass_rate": float | None,
        }
    """
    def _log(event: str, **fields):
        if log_fn:
            log_fn(event, **fields)

    specs = {m.id: m for m in sync_models()}
    gen = next((m for m in specs.values() if m.role == "generator"), None)
    if gen is None:
        raise RuntimeError("No generator model configured in models.yaml.")
    provider = OpenAICompatibleProvider(gen.base_url or "", gen.api_key_env)
    engine = get_engine()

    total = len(source_questions) * n_each
    _log(
        "similar_start",
        n_seeds=len(source_questions),
        n_each=n_each,
        total_candidates=total,
        calibration_k=calibration_k or POLICY_K,
    )

    def _one(item: tuple[int, dict, int], _log) -> dict:
        """Worker: one candidate for one seed."""
        seed_i, q, attempt = item
        idx = seed_i * n_each + attempt + 1
        _log(
            "similar_generate",
            index=idx,
            total=total,
            source_id=q["id"],
            topic=q["topic"],
            band=q["band"],
            attempt=attempt,
            stem_preview=(q["stem"] or "")[:120],
        )
        prompt = SIMILAR_QUESTION_PROMPT.format(
            topic=_label_or_infer(q.get("topic"), "topic"),
            band=_label_or_infer(q.get("band"), "band"),
            stem=q["stem"],
            seed_key=q.get("key", ""),
            prior_variants="(one-shot mode — no prior variants tracked)",
            sme_feedback=sme_feedback_for(q.get("topic"), source_id=q["id"]),
        )
        completion = provider.complete(prompt, gen.model, skip_correction=True)

        def _cal_log(event: str, _sid=q["id"], _att=attempt, **fields):
            _log(event, source_id=_sid, attempt=_att, **fields)

        cid = _store_candidate(
            completion.text,
            q["topic"],
            q["band"],
            gen.id,
            "similar_augment",
            [q["id"]],
            run_id=run_id,
            rl_split="train",
            calibrate=True,
            complete_fn=complete_fn,
            log_fn=_cal_log,
            calibration_k=calibration_k,
        )
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT status, policy_pass_rate, auto_reject_reason FROM candidates WHERE id=:id"),
                {"id": cid},
            ).mappings().first()
        status = row["status"] if row else "unknown"
        rate = row["policy_pass_rate"] if row else None
        reject_reason = row["auto_reject_reason"] if row else None

        _log(
            "similar_result",
            source_id=q["id"],
            candidate_id=cid,
            status=status,
            pass_rate=rate,
            auto_reject_reason=reject_reason,
        )
        return {
            "source_id": q["id"],
            "candidate_id": cid,
            "status": status,
            "pass_rate": rate,
            "reason": reject_reason,
        }

    # Seeds/attempts are independent — run concurrently; events reach log_fn
    # on this thread.
    items = [(i, q, a) for i, q in enumerate(source_questions) for a in range(n_each)]
    results = map_with_events(_one, items, log_fn)

    n_pending = sum(1 for r in results if r["status"] == "pending")
    _log(
        "similar_end",
        n_total=total,
        n_pending=n_pending,
        n_rejected=total - n_pending,
    )
    return results


def generate_similar_react(
    source_questions: list[dict],
    *,
    run_id: str | None = None,
    n_each: int = 1,
    max_loops: int = 3,
    log_fn=None,
    calibration_k: int | None = None,
) -> list[dict]:
    """ReAct-agent variant of :func:`generate_similar_questions`.

    The agent can call ``run_snippet``, ``check_similarity``, and
    ``submit_candidate`` in up to ``max_loops`` tool-call rounds.
    Calibration (pass-rate estimation) runs automatically inside
    ``submit_candidate`` via ``_store_candidate``.

    Returns the same result-dict schema as :func:`generate_similar_questions`.
    """
    import os as _os

    from openai import OpenAI

    def _log(event: str, **fields):
        if log_fn:
            log_fn(event, **fields)

    specs = {m.id: m for m in sync_models()}
    gen = next((m for m in specs.values() if m.role == "generator"), None)
    if gen is None:
        raise RuntimeError("No generator model configured in models.yaml.")

    api_key = _os.getenv(gen.api_key_env or "", "") if gen.api_key_env else "ollama"
    client = OpenAI(base_url=gen.base_url, api_key=api_key or "ollama", timeout=120)

    engine = get_engine()
    total = len(source_questions) * n_each
    _log(
        "similar_start",
        n_seeds=len(source_questions),
        n_each=n_each,
        total_candidates=total,
        calibration_k=calibration_k or POLICY_K,
        mode="react_agent",
        max_loops=max_loops,
    )

    def _one(item: tuple[int, dict, int], _log) -> dict:
        """Worker: one candidate for one seed."""
        seed_i, q, attempt = item
        idx = seed_i * n_each + attempt + 1
        _log(
            "similar_generate",
            index=idx,
            total=total,
            source_id=q["id"],
            topic=q["topic"],
            band=q["band"],
            attempt=attempt,
            stem_preview=(q.get("stem") or "")[:120],
        )

        candidate_id = _run_similar_agent(
            q, gen, client,
            run_id=run_id,
            max_loops=max_loops,
            log_fn=_log,
            calibration_k=calibration_k,
            index=idx,
            total=total,
        )

        if candidate_id:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT status, policy_pass_rate, auto_reject_reason "
                        "FROM candidates WHERE id=:id"
                    ),
                    {"id": candidate_id},
                ).mappings().first()
            status = row["status"] if row else "unknown"
            rate = row["policy_pass_rate"] if row else None
            reject_reason = row["auto_reject_reason"] if row else None
        else:
            status = "not_submitted"
            rate = None
            reject_reason = "agent_did_not_submit"

        _log(
            "similar_result",
            source_id=q["id"],
            candidate_id=candidate_id or "",
            status=status,
            pass_rate=rate,
            auto_reject_reason=reject_reason,
        )
        return {
            "source_id": q["id"],
            "candidate_id": candidate_id or "",
            "status": status,
            "pass_rate": rate,
            "reason": reject_reason,
        }

    # Seeds/attempts are independent — run concurrently; events reach log_fn
    # on this thread.
    items = [(i, q, a) for i, q in enumerate(source_questions) for a in range(n_each)]
    results = map_with_events(_one, items, log_fn)

    n_pending = sum(1 for r in results if r["status"] == "pending")
    _log(
        "similar_end",
        n_total=total,
        n_pending=n_pending,
        n_rejected=total - n_pending,
    )
    return results


def generate_similar_with_review(
    source_questions: list[dict],
    *,
    run_id: str | None = None,
    n_each: int = 1,
    max_fix_iterations: int = MAX_FIX_ITERATIONS,
    calibration_k: int | None = None,
    log_fn=None,
) -> list[dict]:
    """Generate → Review → Fix loop for similar question generation.

    For each seed question:
    1. Generator (generator model, knows seed's gold answer + prior variants)
       produces a new question.
    2. Reviewer (judge model, HIGH-criteria scoring) checks 5 criteria.
       Score ≥ 75/100 → pass. Below → reviewer returns fix instructions.
    3. Generator fixes using the reviewer's feedback (up to max_fix_iterations).
    4. Passed question → sandbox verify → calibrate → SME queue.

    Prior variants for the same seed are tracked across n_each iterations
    so each generated question is guaranteed distinct from siblings.

    Returns same schema as :func:`generate_similar_questions`.
    """
    from frontier_lens.registry import judge_spec as _judge_spec_fn

    def _log(event: str, **fields):
        if log_fn:
            log_fn(event, **fields)

    k = calibration_k if calibration_k is not None else POLICY_K

    specs = {m.id: m for m in sync_models()}
    gen = next((m for m in specs.values() if m.role == "generator"), None)
    if gen is None:
        raise RuntimeError("No generator model configured in models.yaml.")

    judge = _judge_spec_fn()
    if judge is None:
        raise RuntimeError(
            "No judge model configured in models.yaml (role: judge). "
            "Add a judge entry to use Generate+Review mode."
        )

    gen_provider   = OpenAICompatibleProvider(gen.base_url or "", gen.api_key_env)
    judge_provider = OpenAICompatibleProvider(judge.base_url or "", judge.api_key_env)

    engine = get_engine()
    total = len(source_questions) * n_each
    _log("similar_start", n_seeds=len(source_questions), n_each=n_each,
         total_candidates=total, calibration_k=k,
         mode="generate_review", max_fix_iterations=max_fix_iterations)

    def _one_seed(item: tuple[int, dict], _log) -> list[dict]:
        """Worker: all n_each variants for one seed (sequential — each sees the prior ones)."""
        seed_i, q = item
        seed_results: list[dict] = []
        prior_stems: list[str] = []  # track variants generated for this seed

        # SME review feedback for this seed and topic (rejections, reasons, notes).
        sme_feedback = sme_feedback_for(q.get("topic"), source_id=q["id"])

        for attempt in range(n_each):
            idx = seed_i * n_each + attempt + 1
            _log("similar_generate", index=idx, total=total,
                 source_id=q["id"], topic=q["topic"], band=q["band"],
                 attempt=attempt, stem_preview=(q.get("stem") or "")[:120])

            # ── Generate → Review → Fix loop ─────────────────────────────
            current_raw: str | None = None
            review_result: dict = {}
            passed = False

            def _prior_block(stems: list[str], label: str = "Variant") -> str:
                return "\n".join(f"{label} {i+1}: {s[:200]}" for i, s in enumerate(stems)) or "(none)"

            for fix_iter in range(max_fix_iterations + 1):
                is_fix = fix_iter > 0

                if not is_fix:
                    prompt = SIMILAR_QUESTION_PROMPT.format(
                        topic=_label_or_infer(q.get("topic"), "topic"),
                        band=_label_or_infer(q.get("band"), "band"),
                        stem=q["stem"],
                        seed_key=q.get("key", ""),
                        prior_variants=_prior_block(prior_stems),
                        sme_feedback=sme_feedback,
                    )
                    _log("review_generate", index=idx, fix_iter=0,
                         source_id=q["id"],
                         has_sme_feedback=sme_feedback != NO_FEEDBACK)
                else:
                    prompt = SIMILAR_FIX_PROMPT.format(
                        seed_stem=q["stem"][:600],
                        seed_key=q.get("key", ""),
                        previous_stem=_extract_stem_from_raw(current_raw or ""),
                        score=review_result.get("score", 0),
                        issues="; ".join(review_result.get("issues", [])),
                        fix_instructions=review_result.get("fix_instructions", ""),
                        prior_variants=_prior_block(prior_stems),
                        sme_feedback=sme_feedback,
                    )
                    _log("review_fix", index=idx, fix_iter=fix_iter,
                         source_id=q["id"],
                         score=review_result.get("score"),
                         issues=review_result.get("issues", []))

                completion = gen_provider.complete(prompt, gen.model, skip_correction=True)
                current_raw = completion.text

                # Sandbox check
                try:
                    payload = _extract_json(current_raw)
                except Exception:
                    _log("review_sandbox", index=idx, fix_iter=fix_iter,
                         ok=False, reason="json_parse_error")
                    continue

                stem_gen  = payload.get("stem") or ""
                code_gen  = payload.get("code") or ""
                lang_gen  = payload.get("lang") or "python"
                files_gen = payload.get("input_files") or {}

                sandbox_result = execute(lang_gen, code_gen, files_gen, twice=True, single_line=True)
                _log("review_sandbox", index=idx, fix_iter=fix_iter,
                     ok=sandbox_result.ok, reason=sandbox_result.reason)

                if not sandbox_result.ok:
                    review_result = {
                        "score": 0, "passed": False,
                        "issues": [f"Sandbox error: {sandbox_result.reason}"],
                        "fix_instructions": (
                            "Fix the code so it runs without errors and prints exactly one "
                            "non-empty line, the same on every run."
                        ),
                        "criterion_scores": {},
                    }
                    continue

                computed_key = sandbox_result.stdout

                # ── Reviewer step ─────────────────────────────────────────
                _log("review_start", index=idx, fix_iter=fix_iter,
                     source_id=q["id"], stem_preview=stem_gen[:120])

                review_result = _review_similar_question(
                    stem=stem_gen, code=code_gen, lang=lang_gen,
                    computed_key=computed_key,
                    seed_stem=q["stem"], seed_key=q.get("key", ""),
                    band=_label_or_infer(q.get("band"), "band"),
                    prior_stems=prior_stems,
                    sme_feedback=sme_feedback,
                    provider=judge_provider, model=judge.model,
                )

                _log("review_result", index=idx, fix_iter=fix_iter,
                     source_id=q["id"],
                     score=review_result["score"],
                     passed=review_result["passed"],
                     criterion_scores=review_result["criterion_scores"],
                     issues=review_result["issues"])

                if review_result["passed"]:
                    passed = True
                    break
                # Loop: send fix instructions back to generator

            # ── Store result ──────────────────────────────────────────────
            if passed and current_raw:
                def _cal_log(event: str, _sid=q["id"], _att=attempt, **fields):
                    _log(event, source_id=_sid, attempt=_att, **fields)

                cid = _store_candidate(
                    current_raw, q["topic"], q["band"],
                    gen.id, "similar_augment", [q["id"]],
                    run_id=run_id,
                    rl_split="train", calibrate=True,
                    log_fn=_cal_log, calibration_k=k,
                )
                with engine.connect() as conn:
                    row = conn.execute(
                        text("SELECT status, policy_pass_rate, auto_reject_reason FROM candidates WHERE id=:id"),
                        {"id": cid},
                    ).mappings().first()
                status       = row["status"] if row else "unknown"
                rate         = row["policy_pass_rate"] if row else None
                reject_reason = row["auto_reject_reason"] if row else None

                # Track this stem as a prior variant for subsequent n_each iterations
                try:
                    prior_stems.append(_extract_json(current_raw).get("stem", "")[:300])
                except Exception:
                    pass
            else:
                cid = None
                status = "review_failed"
                rate = None
                reject_reason = f"did not reach score ≥ 75 in {max_fix_iterations + 1} attempts"

            _log("similar_result", source_id=q["id"], candidate_id=cid or "",
                 status=status, pass_rate=rate, auto_reject_reason=reject_reason,
                 review_score=review_result.get("score"),
                 fix_iterations_used=(max_fix_iterations + 1 if not passed
                                      else review_result.get("_fix_iter", 0)))
            seed_results.append({
                "source_id": q["id"],
                "candidate_id": cid or "",
                "status": status,
                "pass_rate": rate,
                "reason": reject_reason,
            })
        return seed_results

    # Seeds are independent — run concurrently; events reach log_fn on this thread.
    results = [
        r for seed_results in map_with_events(_one_seed, enumerate(source_questions), log_fn)
        for r in seed_results
    ]

    n_pending = sum(1 for r in results if r["status"] == "pending")
    _log("similar_end", n_total=total, n_pending=n_pending,
         n_rejected=total - n_pending)
    return results


# ── ReAct agent + Generate + Review ──────────────────────────────────────

_SUBMIT_DRAFT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_draft",
        "description": "Send the finished draft for review (call once per draft)",
        "parameters": {
            "type": "object",
            "properties": {
                "stem": {"type": "string"},
                "lang": {"type": "string"},
                "code": {"type": "string"},
                "input_files": {"type": "object"},
                "numeric": {"type": "boolean"},
                "topic": {"type": "string"},
                "band": {"type": "string"},
            },
            "required": ["stem", "lang", "code", "topic", "band"],
        },
    },
}


def _react_review_tools() -> list[dict]:
    """run_snippet and check_similarity, plus submit_draft (hands the draft to the reviewer).

    There is no tool that stores a candidate or calls the policy: storing and calibration
    happen only after the reviewer passes the draft.
    """
    from frontier_lens.agent.tools import BASE_TOOLS

    allowed = {"run_snippet", "check_similarity"}
    return [t for t in BASE_TOOLS if t["function"]["name"] in allowed] + [_SUBMIT_DRAFT_TOOL]


def _prior_variants_for_seed(seed_id: str, limit: int = 6) -> list[str]:
    """Stems of earlier pending or approved variants of this seed (newest first)."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT stem FROM candidates
                WHERE source_fail_ids LIKE :pat AND status IN ('pending', 'approved')
                  AND generator_mode = 'similar_augment'
                ORDER BY created_at DESC LIMIT :n
                """
            ),
            {"pat": f'%"{seed_id}"%', "n": limit},
        ).scalars().all()
    return [(r or "")[:300] for r in rows]


def _agent_draft(
    messages: list,
    client,
    gen,
    toolset: list[dict],
    *,
    source_q: dict,
    run_id: str | None,
    max_loops: int,
    log_fn,
    calibration_k: int | None,
) -> dict | None:
    """Run up to *max_loops* tool rounds; return the arguments of ``submit_draft`` or None.

    *messages* is extended in place, so the next call continues the same conversation.
    """
    from frontier_lens.agent.tools import handle

    for step in range(max_loops):
        try:
            with call_slot():
                resp = client.chat.completions.create(
                    model=gen.model, messages=messages, tools=toolset, timeout=120,
                )
        except Exception as exc:
            log_fn("agent_error", step=step + 1, error=str(exc))
            return None
        msg = resp.choices[0].message
        if not msg.tool_calls:
            log_fn("agent_no_tool_call", step=step + 1, content=(msg.content or "")[:200])
            return None
        messages.append(msg)
        draft: dict | None = None
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            name = call.function.name
            log_fn("agent_tool_call", step=step + 1, tool=name,
                   args_preview={k: str(v)[:120] for k, v in args.items()})
            if name == "submit_draft":
                draft = args
                result = json.dumps({"received": True, "next": "the reviewer scores this draft"})
            else:
                result = handle(
                    name, args, gen_id=gen.id, fail_model=source_q["id"], run_id=run_id,
                    rl_split="train", log_fn=log_fn, calibration_k=calibration_k,
                )
            log_fn("agent_tool_result", step=step + 1, tool=name, result_preview=result[:300])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        if draft is not None:
            return draft
    return None


def generate_similar_react_review(
    source_questions: list[dict],
    *,
    run_id: str | None = None,
    n_each: int = 1,
    max_loops: int = 3,
    max_fix_iterations: int = MAX_FIX_ITERATIONS,
    calibration_k: int | None = None,
    complete_fn=None,
    log_fn=None,
    _client=None,
    _judge_provider=None,
) -> list[dict]:
    """ReAct agent drafts, a reviewer scores, the agent revises: then sandbox, calibrate, SME.

    For each variant of each seed:

    1. The generator model runs as a tool-using agent (``run_snippet``, ``check_similarity``,
       ``submit_draft``; up to *max_loops* tool rounds per draft). Its prompt carries the
       seed, its gold answer, the earlier variants of the seed (this run and pending or
       approved ones in the database) and the SME feedback (``feedback.context_for``:
       what SMEs rejected and approved on other samples).
    2. The draft's code runs in the sandbox (twice, one non-empty line of output).
    3. The reviewer (judge model) scores five criteria; 75 of 100 passes. The SME feedback
       is in the reviewer's prompt too, and repeating a flagged problem caps the score.
    4. A failed draft sends the reviewer's issues back into the *same conversation* and the
       agent revises, up to *max_fix_iterations* times.
    5. A passed draft is stored with calibration; the band filter in ``_store_candidate``
       rejects it if the policy always or never solves it.

    Returns the same result-dict schema as :func:`generate_similar_with_review`.
    """
    import os as _os

    from openai import OpenAI

    from frontier_lens.registry import judge_spec as _judge_spec_fn

    def _log(event: str, **fields):
        if log_fn:
            log_fn(event, **fields)

    k = calibration_k if calibration_k is not None else POLICY_K
    specs = {m.id: m for m in sync_models()}
    gen = next((m for m in specs.values() if m.role == "generator"), None)
    if gen is None:
        raise RuntimeError("No generator model configured in models.yaml.")
    judge = _judge_spec_fn()
    if judge is None:
        raise RuntimeError(
            "No judge model configured in models.yaml (role: judge). "
            "Add a judge entry to use ReAct + Review mode."
        )
    client = _client
    if client is None:
        api_key = _os.getenv(gen.api_key_env or "", "") if gen.api_key_env else "ollama"
        client = OpenAI(base_url=gen.base_url, api_key=api_key or "ollama", timeout=120)
    judge_provider = _judge_provider or OpenAICompatibleProvider(
        judge.base_url or "", judge.api_key_env
    )
    toolset = _react_review_tools()
    engine = get_engine()
    total = len(source_questions) * n_each
    _log("similar_start", n_seeds=len(source_questions), n_each=n_each, total_candidates=total,
         calibration_k=k, mode="react_review", max_loops=max_loops,
         max_fix_iterations=max_fix_iterations)

    def _one_seed(item: tuple[int, dict], _log) -> list[dict]:
        """Worker: all n_each variants of one seed, one after another (each sees the earlier)."""
        seed_i, q = item
        sme_feedback = sme_feedback_for(q.get("topic"), source_id=q["id"])
        prior_stems = _prior_variants_for_seed(q["id"])
        results: list[dict] = []
        for attempt in range(n_each):
            idx = seed_i * n_each + attempt + 1
            _log("similar_generate", index=idx, total=total, source_id=q["id"],
                 topic=q["topic"], band=q["band"], attempt=attempt,
                 stem_preview=(q.get("stem") or "")[:120])
            prior_block = "\n".join(
                f"  Variant {i + 1}: {stem[:200]}" for i, stem in enumerate(prior_stems)
            ) or "  (none yet)"
            messages: list = [{"role": "user", "content": SIMILAR_AGENT_REVIEW_SYSTEM.format(
                topic=_label_or_infer(q.get("topic"), "topic"),
                band=_label_or_infer(q.get("band"), "band"),
                stem=(q.get("stem") or "")[:1200],
                seed_key=q.get("key", ""),
                prior_variants=prior_block,
                sme_feedback=sme_feedback,
                max_loops=max_loops,
            )}]
            review_result: dict = {}
            draft: dict | None = None
            passed = False
            for fix_iter in range(max_fix_iterations + 1):
                if fix_iter == 0:
                    _log("review_generate", index=idx, fix_iter=0, source_id=q["id"],
                         has_sme_feedback=sme_feedback != NO_FEEDBACK)
                else:
                    messages.append({"role": "user", "content": SIMILAR_AGENT_FIX_MESSAGE.format(
                        score=review_result.get("score", 0),
                        issues="; ".join(review_result.get("issues", [])) or "(none listed)",
                        fix_instructions=review_result.get("fix_instructions", ""),
                    )})
                    _log("review_fix", index=idx, fix_iter=fix_iter, source_id=q["id"],
                         score=review_result.get("score"), issues=review_result.get("issues", []))
                draft = _agent_draft(
                    messages, client, gen, toolset, source_q=q, run_id=run_id,
                    max_loops=max_loops, log_fn=lambda e, _i=idx, **f: _log(
                        e, source_id=q["id"], agent_index=_i, **f),
                    calibration_k=calibration_k,
                )
                if not draft or not draft.get("stem") or not draft.get("code"):
                    draft = None
                    review_result = {
                        "score": 0, "passed": False, "criterion_scores": {},
                        "issues": ["No usable draft was submitted"],
                        "fix_instructions": "Call submit_draft with stem, lang, code, topic and band.",
                    }
                    _log("review_sandbox", index=idx, fix_iter=fix_iter, ok=False,
                         reason="no_draft")
                    continue
                lang = draft.get("lang") or "python"
                files = draft.get("input_files") or {}
                sandbox_result = execute(lang, draft["code"], files, twice=True, single_line=True)
                _log("review_sandbox", index=idx, fix_iter=fix_iter,
                     ok=sandbox_result.ok, reason=sandbox_result.reason)
                if not sandbox_result.ok:
                    review_result = {
                        "score": 0, "passed": False, "criterion_scores": {},
                        "issues": [f"Sandbox error: {sandbox_result.reason}"],
                        "fix_instructions": (
                            "Fix the code so it runs without errors and prints exactly one "
                            "non-empty line, the same on every run."
                        ),
                    }
                    continue
                _log("review_start", index=idx, fix_iter=fix_iter, source_id=q["id"],
                     stem_preview=draft["stem"][:120])
                review_result = _review_similar_question(
                    stem=draft["stem"], code=draft["code"], lang=lang,
                    computed_key=sandbox_result.stdout,
                    seed_stem=q["stem"], seed_key=q.get("key", ""),
                    band=_label_or_infer(q.get("band"), "band"),
                    prior_stems=prior_stems, sme_feedback=sme_feedback,
                    provider=judge_provider, model=judge.model,
                )
                _log("review_result", index=idx, fix_iter=fix_iter, source_id=q["id"],
                     score=review_result["score"], passed=review_result["passed"],
                     criterion_scores=review_result["criterion_scores"],
                     issues=review_result["issues"])
                if review_result["passed"]:
                    passed = True
                    break

            if passed and draft:
                raw = json.dumps({
                    "stem": draft["stem"], "lang": draft.get("lang") or "python",
                    "code": draft["code"], "input_files": draft.get("input_files") or {},
                    "numeric": bool(draft.get("numeric", True)),
                    "topic": draft.get("topic") or "", "band": draft.get("band") or "",
                })

                def _cal_log(event: str, _sid=q["id"], _att=attempt, **fields):
                    _log(event, source_id=_sid, attempt=_att, **fields)

                cid = _store_candidate(
                    raw, q["topic"], q["band"], gen.id, "similar_augment", [q["id"]],
                    run_id=run_id, rl_split="train", calibrate=True,
                    complete_fn=complete_fn, log_fn=_cal_log, calibration_k=k,
                )
                with engine.connect() as conn:
                    row = conn.execute(
                        text("SELECT status, policy_pass_rate, auto_reject_reason "
                             "FROM candidates WHERE id=:id"),
                        {"id": cid},
                    ).mappings().first()
                status = row["status"] if row else "unknown"
                rate = row["policy_pass_rate"] if row else None
                reject_reason = row["auto_reject_reason"] if row else None
                prior_stems.append(draft["stem"][:300])
            else:
                cid, status, rate = None, "review_failed", None
                reject_reason = (
                    f"did not reach score >= 75 in {max_fix_iterations + 1} drafts"
                )
            _log("similar_result", source_id=q["id"], candidate_id=cid or "", status=status,
                 pass_rate=rate, auto_reject_reason=reject_reason,
                 review_score=review_result.get("score"))
            results.append({"source_id": q["id"], "candidate_id": cid or "", "status": status,
                            "pass_rate": rate, "reason": reject_reason})
        return results

    # Seeds are independent: run them concurrently; events reach log_fn on this thread.
    results = [
        r for seed_results in map_with_events(_one_seed, enumerate(source_questions), log_fn)
        for r in seed_results
    ]
    n_pending = sum(1 for r in results if r["status"] == "pending")
    _log("similar_end", n_total=total, n_pending=n_pending, n_rejected=total - n_pending)
    return results
