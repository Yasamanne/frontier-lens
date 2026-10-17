from __future__ import annotations

import json
import re as _re
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from frontier_lens.config import POLICY_K
from frontier_lens.db import get_engine
from frontier_lens.failure_analysis import failure_modes, failure_summary_text
from frontier_lens.heatmap import weakest_cell
from frontier_lens.prompt import contest_prompt
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import policy_spec, sync_models
from frontier_lens.reward import grade
from frontier_lens.sandbox import execute
from frontier_lens.split import EVAL_TEMPLATES
from frontier_lens.stats import in_learnable_band, pass_at_k, pass_rate
from frontier_lens.templates import all_specs


def _extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from a model response.

    Tries (in order):
    1. Markdown fenced block  ``` json { … } ```
    2. Outermost ``{`` … ``}`` in the raw text
    3. Each line that starts with ``{``

    Raises ``ValueError`` if nothing parses.
    """
    # 1. markdown code fence (```json or plain ```)
    m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. outermost { … }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 3. scan line-by-line for a self-contained JSON object
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no valid JSON object found in response ({len(text)} chars)")


def _label_or_infer(value: str | None, field: str) -> str:
    """Return the value if set, else a prompt placeholder telling the model to infer it."""
    if value and value.strip():
        return value.strip()
    return f"(unknown — infer {field} from the seed question)"


MAX_FIX_ITERATIONS = 3   # 3 fix iterations is enough for our shorter questions

SIMILAR_REVIEW_PROMPT = """\
You are a strict question quality reviewer. Score the GENERATED question on
5 criteria (20 pts each, max 100). Pass threshold: 75.

══ SEED QUESTION (what this was generated from) ══
{seed_stem}

Gold answer of seed (known to you): {seed_key}

══ GENERATED QUESTION ══
Stem:
{generated_stem}

Verifier code ({lang}):
{code}

Computed answer (from running the code): {computed_key}

══ PRIOR APPROVED/PENDING VARIANTS ══
{prior_variants}

══ PREVIOUSLY REJECTED VARIANTS (SME rejected these — factor into distinctness score) ══
{rejected_variants}

══ CRITERIA ══

1. CONCEPT ALIGNED (0–20): Tests the same skill/domain as the seed.
   Same type of problem, same difficulty band ({band}).

2. GENUINELY DISTINCT (0–20): Meaningfully different from the seed AND all
   prior variants. Changing only numbers = 0 pts. New scenario/structure = 20 pts.

3. ANSWER UNIQUENESS (0–20): Exactly one correct answer. A knowledgeable
   solver would always reach the same answer.

4. STEM CLARITY (0–20): Unambiguous question. Clear inputs, clear what is
   being asked. No missing information.

5. DIFFICULTY APPROPRIATE (0–20): Neither trivially easy nor impossibly hard
   for band={band}. The model should sometimes pass, sometimes fail.

Return JSON only:
{{
  "score": <0-100 integer>,
  "passed": <true if score >= 75>,
  "criterion_scores": {{
    "concept_aligned": <0-20>,
    "genuinely_distinct": <0-20>,
    "answer_uniqueness": <0-20>,
    "stem_clarity": <0-20>,
    "difficulty_appropriate": <0-20>
  }},
  "issues": ["<issue 1>", "<issue 2>"],
  "fix_instructions": "<specific paragraph telling the generator exactly what to change>"
}}
"""

SIMILAR_FIX_PROMPT = """\
You are revising a question based on reviewer feedback.

SEED QUESTION (match its style/difficulty):
{seed_stem}
Gold answer of seed: {seed_key}

PREVIOUS ATTEMPT (needs fixing):
{previous_stem}

REVIEWER FEEDBACK (score={score}/100, FAILED):
Issues: {issues}
Fix instructions: {fix_instructions}

PRIOR APPROVED/PENDING VARIANTS (must differ from ALL):
{prior_variants}

PREVIOUSLY REJECTED VARIANTS — SME found these bad, avoid these patterns entirely:
{rejected_variants}

Apply the fix instructions and produce an improved question.
Keep what was good; fix only what the reviewer flagged.

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true, "topic": "...", "band": "standard" or "hard"}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf awk;
python may import math, itertools, collections, decimal only.
If topic or band are empty or unknown, infer them from the seed question:
  topic must be one of: python_state, shell_pipelines, constraint_counting, multi_step_units.
  band must be one of: standard, hard.
"""

_LEVEL_INSTRUCTIONS: dict[int, str] = {
    1: (
        "Add ONE sentence about the HIGH-LEVEL DOMAIN or conceptual framing only. "
        "Do NOT name the technique, formula, or method. "
        'Example: "Think about what this structure looks like geometrically."'
    ),
    2: (
        "The model still fails. Add ONE sentence naming the SPECIFIC TECHNIQUE or "
        "mathematical property that applies — but not how to apply it. "
        "Do NOT show any computation. "
        'Example: "The key tool here is the Nullstellensatz."'
    ),
    3: (
        "The model still fails after two hints. Add ONE final sentence pointing out "
        "a COMMON MISTAKE or edge case the model is likely making based on the failure mode. "
        "Do NOT reveal the answer or the key step. "
        'Example: "Be careful: the formula counts ordered, not unordered, pairs."'
    ),
}

PROGRESSIVE_HINT_PROMPT = """\
You are adding a progressive hint to a question a model ALWAYS gets wrong.

GOLD ANSWER (known to you — do NOT reveal it): {gold_key}
FAILURE MODE: {failure_mode}
HINT ROUND: {hint_level} of {max_hints}

CURRENT STEM (may already contain hints from earlier rounds):
{current_stem}

YOUR TASK FOR THIS ROUND:
{level_instruction}

You know the answer is "{gold_key}". Use this knowledge to craft a precise hint that
nudges the solver toward the right approach WITHOUT stating the answer or the decisive step.
The hint should be embedded naturally at the end of the stem (e.g. as "Hint:" or "Note:").

Return JSON only:
{{"stem": "<full updated stem with the new hint appended at the end>", "hint": "<just the new sentence you added>"}}
"""

# Keep the old name as an alias so any external callers don't break
LIGHT_HINT_PROMPT = PROGRESSIVE_HINT_PROMPT

SIMILAR_QUESTION_PROMPT = """\
You write a NEW question that is SIMILAR in style and difficulty to the seed question below.
Topic: {topic}
Band: {band}

Seed question (do NOT copy — create a structurally similar but distinct question):
{stem}

Gold answer of seed (known to you — use to guide difficulty, do NOT reveal): {seed_key}

Prior approved/pending variants for this seed (must differ from ALL of these):
{prior_variants}

PREVIOUSLY REJECTED VARIANTS — SME found these bad, avoid these patterns entirely:
{rejected_variants}

The new question must have a unique, deterministic answer verifiable by running code.
The policy model should sometimes pass and sometimes fail (0 < pass_rate < 1).

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true, "topic": "...", "band": "standard" or "hard"}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf awk;
python may import math, itertools, collections, decimal only. No network, no files outside input_files.
If Topic or Band above are empty or unknown, infer them from the seed question:
  topic must be one of: python_state, shell_pipelines, constraint_counting, multi_step_units.
  band must be one of: standard, hard.
"""

SIMILAR_AGENT_SYSTEM = """\
You are a question designer. Your task: write ONE new question that is similar in \
style and difficulty to the SEED question below, but is genuinely distinct from it.

TOPIC: {topic}
BAND:  {band}

SEED QUESTION:
{stem}

AVAILABLE TOOLS (use sparingly — you have at most {max_loops} tool-call rounds):
  run_snippet   — execute code in a sandbox; use this to verify your verifier code is correct
  check_similarity — confirm your new stem is not too close to existing questions
  submit_candidate — call this ONCE when your question is ready

WORKFLOW (suggested):
  1. Draft a new question + verifier code.
  2. Call run_snippet to confirm the code runs and produces a deterministic answer.
  3. If it fails, fix and re-test (uses one more round).
  4. Call submit_candidate. Done.

RULES:
  - The answer must be a single bare value (numeric or short string).
  - Shell code: only cat cut sort uniq wc head tail tr grep printf awk.
  - Python: only math, itertools, collections, decimal. No network, no external files.
  - If TOPIC or BAND above are empty, infer them from the seed and include them in submit_candidate.
    topic must be one of: python_state, shell_pipelines, constraint_counting, multi_step_units.
    band must be one of: standard, hard.
  - If TOPIC and BAND are already set above, you do NOT need to include them in submit_candidate.
"""

ONE_SHOT_TRAIN = """You write a NEW original train item for open-weight models.
Topic: {topic}
Band: {band}

=== HOW THE POLICY FAILS ON THIS CELL ===
{failure_analysis}

=== FAILING QUESTION STEMS (do NOT clone, use for inspiration only) ===
{failures}

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf awk;
python may import math, itertools, collections, decimal only. No network, no files outside input_files.
Design the item to target the dominant failure mode listed above.
The policy should neither always fail nor always pass (0 < pass_rate < 1).
"""

ONE_SHOT_EVAL = """You write a NEW original held-out eval item for open-weight models.
Topic: {topic}
Band: {band}
Do not use any policy-model results. This item must not be calibrated to a policy.

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf awk;
python may import math, itertools, collections, decimal only. No network, no files outside input_files.
"""


def _eval_cells() -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for spec in all_specs():
        if spec.template in EVAL_TEMPLATES:
            cell = (spec.topic, spec.band)
            if cell not in seen:
                seen.append(cell)
    return seen or [("python_state", "hard")]


def _failures(run_id: str, model_id: str, topic: str, band: str, limit: int = 6) -> str:
    """Return concatenated stems of questions the model mostly fails."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT q.stem, q.id, AVG(a.is_correct) as acc
                FROM attempts a JOIN questions q ON q.id=a.question_id
                WHERE a.run_id=:r AND a.model_id=:m AND q.topic=:t AND q.band=:b AND q.split='train'
                GROUP BY q.id
                HAVING acc < 0.5
                LIMIT :n
                """
            ),
            {"r": run_id, "m": model_id, "t": topic, "b": band, "n": limit},
        ).mappings().all()
    if not rows:
        return "(no listed failures)"
    return "\n---\n".join(r["stem"][:800] for r in rows)


def _failure_analysis_text(run_id: str, model_id: str, topic: str, band: str) -> str:
    """Return a human-readable failure mode analysis for the prompt."""
    analysis = failure_modes(run_id, model_id, topic, band, split="train")
    return failure_summary_text(analysis)


def estimate_pass_rate(
    stem: str,
    key: str,
    numeric: bool = True,
    *,
    complete_fn=None,
    k: int | None = None,
    log_fn=None,
) -> dict:
    """Sample the policy k times. Train-split calibration only — never call on eval.

    If *log_fn* is provided it is called once per sample with::

        log_fn("calibration_sample", sample_index=i, is_correct=bool, reward=int)
    """
    policy = policy_spec()
    n = k if k is not None else POLICY_K
    if complete_fn is None:
        provider = OpenAICompatibleProvider(policy.base_url or "", policy.api_key_env)

        def complete_fn(prompt: str) -> str:
            return provider.complete(prompt, policy.model).text

    correct = 0
    for i in range(n):
        text_out = complete_fn(contest_prompt(stem))
        reward, _, _ = grade(text_out, key, numeric)
        correct += reward
        if log_fn:
            log_fn("calibration_sample", sample_index=i, is_correct=bool(reward), reward=int(reward))
    rate = pass_rate(correct, n)
    return {
        "n": n,
        "correct": correct,
        "pass_rate": rate,
        "pass_at_k": pass_at_k(n, correct, n),
        "in_learnable_band": in_learnable_band(rate),
        "policy_model": policy.id,
    }


def propose_one_shot(
    run_id: str,
    model_id: str,
    n: int = 10,
    *,
    rl_split: str = "train",
) -> list[str]:
    if rl_split not in {"train", "eval"}:
        raise ValueError(f"bad split {rl_split}")
    specs = {m.id: m for m in sync_models()}
    gen = next(m for m in specs.values() if m.role == "generator")
    provider = OpenAICompatibleProvider(gen.base_url or "", gen.api_key_env)
    ids = []
    if rl_split == "eval":
        cells = _eval_cells()
        for i in range(n):
            topic, band = cells[i % len(cells)]
            prompt = ONE_SHOT_EVAL.format(topic=topic, band=band)
            completion = provider.complete(prompt, gen.model, skip_correction=True)
            cid = _store_candidate(
                completion.text,
                topic,
                band,
                gen.id,
                "one_shot",
                [],
                rl_split="eval",
                calibrate=False,
            )
            ids.append(cid)
        return ids

    cell = weakest_cell(run_id, model_id)
    if not cell:
        raise RuntimeError("no heatmap cell")
    fails = _failures(run_id, model_id, cell.topic, cell.band)
    fa_text = _failure_analysis_text(run_id, model_id, cell.topic, cell.band)
    for _ in range(n):
        prompt = ONE_SHOT_TRAIN.format(
            topic=cell.topic, band=cell.band, failures=fails, failure_analysis=fa_text
        )
        completion = provider.complete(prompt, gen.model, skip_correction=True)
        cid = _store_candidate(
            completion.text,
            cell.topic,
            cell.band,
            gen.id,
            "one_shot",
            [model_id],
            run_id=run_id,
            rl_split="train",
            calibrate=True,
        )
        ids.append(cid)
    return ids


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
                if mode == "one_shot" and not stats["in_learnable_band"]:
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


def _calibrate_collect(
    stem: str,
    key: str,
    numeric: bool,
    k: int,
    complete_fn=None,
    log_fn=None,
) -> dict:
    """Like ``estimate_pass_rate`` but also returns individual raw responses.

    Returns::

        {
          "n": k, "correct": int, "pass_rate": float, "pass_at_k": float,
          "in_learnable_band": bool, "policy_model": str,
          "responses": [{"raw": str, "parsed": str, "is_correct": bool}]
        }
    """
    policy = policy_spec()
    if complete_fn is None:
        provider = OpenAICompatibleProvider(policy.base_url or "", policy.api_key_env)

        def complete_fn(prompt: str) -> str:
            return provider.complete(prompt, policy.model).text

    responses: list[dict] = []
    correct = 0
    for i in range(k):
        raw = complete_fn(contest_prompt(stem))
        reward, parsed_ans, _ = grade(raw, key, numeric)
        correct += reward
        responses.append({"raw": raw, "parsed": parsed_ans or "", "is_correct": bool(reward)})
        if log_fn:
            log_fn("calibration_sample", sample_index=i, is_correct=bool(reward), reward=int(reward))

    rate = pass_rate(correct, k)
    return {
        "n": k,
        "correct": correct,
        "pass_rate": rate,
        "pass_at_k": pass_at_k(k, correct, k),
        "in_learnable_band": in_learnable_band(rate),
        "policy_model": policy.id,
        "responses": responses,
    }


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


def rescue_with_hints(
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
    provider = OpenAICompatibleProvider(gen.base_url or "", gen.api_key_env)
    engine = get_engine()
    results: list[dict] = []

    _log("rescue_start", n_questions=len(question_ids), run_id=run_id,
         model_id=model_id, calibration_k=k, max_hints=max_hints)

    for i, qid in enumerate(question_ids):
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
            continue

        if not (q.get("code") or "").strip():
            _log("rescue_question_skip", index=i + 1, total=len(question_ids),
                 question_id=qid,
                 reason="no_verifier_code — uploaded/multiple-choice question")
            continue

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

        # ── Progressive hint loop ─────────────────────────────────────────
        current_stem = q["stem"]
        hint_rounds: list[dict] = []
        rescued = False
        candidate_id: str | None = None
        final_pass_rate = 0.0
        rescue_level: int | None = None

        for level in range(1, max_hints + 1):
            max_level_key = max(_LEVEL_INSTRUCTIONS.keys())
            level_instr = _LEVEL_INSTRUCTIONS.get(
                level, _LEVEL_INSTRUCTIONS[max_level_key]
            )

            _log("hint_attempt_start", question_id=qid, level=level,
                 max_hints=max_hints, current_stem_preview=current_stem[:120])

            prompt = PROGRESSIVE_HINT_PROMPT.format(
                gold_key=q["key"],
                failure_mode=dominant,
                hint_level=level,
                max_hints=max_hints,
                current_stem=current_stem,
                level_instruction=level_instr,
            )
            completion = provider.complete(prompt, gen.model, skip_correction=True)

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

            _log("hint_generated", question_id=qid, level=level,
                 hint=hint_text, parse_ok=parse_ok, parse_error=parse_error,
                 raw_response_preview=(completion.text or "")[:400])

            def _cal_log(event: str, _qid=qid, _level=level, **fields):
                _log(event, question_id=_qid, level=_level, **fields)

            stats = _calibrate_collect(
                hinted_stem, q["key"], bool(q["numeric"]),
                k=k, complete_fn=complete_fn, log_fn=_cal_log,
            )

            hint_rounds.append({
                "level": level,
                "hint": hint_text,
                "hinted_stem": hinted_stem,
                "pass_rate": stats["pass_rate"],
                "rl_suitable": stats["in_learnable_band"],
                "responses": stats["responses"],
            })

            _log("hint_attempt_result", question_id=qid, level=level,
                 pass_rate=stats["pass_rate"],
                 n_correct=stats["correct"], n=stats["n"],
                 rl_suitable=stats["in_learnable_band"])

            final_pass_rate = stats["pass_rate"]
            current_stem = hinted_stem   # next level builds on this

            if stats["in_learnable_band"]:
                files = json.loads(q["input_files_json"] or "{}")
                candidate_id = _store_hinted_candidate(
                    source_qid=qid,
                    hinted_stem=hinted_stem,
                    code=q["code"], lang=q["lang"], files=files,
                    key=q["key"], numeric=bool(q["numeric"]),
                    topic=q["topic"], band=q["band"],
                    gen_id=gen.id, stats=stats,
                    run_id=run_id,
                )
                rescued = True
                rescue_level = level
                _log("hint_result", question_id=qid, candidate_id=candidate_id,
                     pass_rate=stats["pass_rate"], n_correct=stats["correct"],
                     n=stats["n"], rl_suitable=True, rescue_level=level)
                break

        if not rescued:
            _log("hint_result", question_id=qid, candidate_id=None,
                 pass_rate=final_pass_rate, rl_suitable=False,
                 rescue_level=None, n_hints_tried=len(hint_rounds))

        results.append({
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
        })

    n_rescued = sum(1 for r in results if r["rl_suitable"])
    _log("rescue_end", n_total=len(question_ids),
         n_rescued=n_rescued, n_rejected=len(question_ids) - n_rescued)
    return results


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
    results: list[dict] = []

    total = len(source_questions) * n_each
    _log(
        "similar_start",
        n_seeds=len(source_questions),
        n_each=n_each,
        total_candidates=total,
        calibration_k=calibration_k or POLICY_K,
    )

    idx = 0
    for q in source_questions:
        for attempt in range(n_each):
            idx += 1
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
            _rej = _get_rejected_variants(q["id"])
            prompt = SIMILAR_QUESTION_PROMPT.format(
                topic=_label_or_infer(q.get("topic"), "topic"),
                band=_label_or_infer(q.get("band"), "band"),
                stem=q["stem"],
                seed_key=q.get("key", ""),
                prior_variants="(one-shot mode — no prior variants tracked)",
                rejected_variants="\n".join(f"Rejected {i+1}: {s[:200]}" for i, s in enumerate(_rej)) or "(none)",
            )
            completion = provider.complete(prompt, gen.model, skip_correction=True)

            # Forward calibration-sample events from _store_candidate → estimate_pass_rate
            def _cal_log(event: str, _sid=q["id"], _att=attempt, **fields):
                _log(event, source_id=_sid, attempt=_att, **fields)

            # We need to intercept the sandbox result too; patch a wrapper complete_fn
            # that also logs calibration samples via _cal_log.
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
            results.append(
                {
                    "source_id": q["id"],
                    "candidate_id": cid,
                    "status": status,
                    "pass_rate": rate,
                }
            )

    n_pending = sum(1 for r in results if r["status"] == "pending")
    _log(
        "similar_end",
        n_total=total,
        n_pending=n_pending,
        n_rejected=total - n_pending,
    )
    return results


def _similar_agent_tools() -> list[dict]:
    """Toolset for the similar-question ReAct agent (no policy-calling tools)."""
    from frontier_lens.agent.tools import BASE_TOOLS
    allowed = {"run_snippet", "check_similarity", "submit_candidate"}
    return [t for t in BASE_TOOLS if t["function"]["name"] in allowed]


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

    rejected = _get_rejected_variants(source_q["id"])
    rejected_block = (
        "\n".join(f"Rejected {i+1}: {s[:200]}" for i, s in enumerate(rejected))
        if rejected else "(none — no prior rejections for this seed)"
    )
    system_prompt = SIMILAR_AGENT_SYSTEM.format(
        topic=_label_or_infer(source_q.get("topic"), "topic"),
        band=_label_or_infer(source_q.get("band"), "band"),
        stem=(source_q.get("stem") or "")[:1200],
        max_loops=max_loops,
    ) + f"\n\nSME-REJECTED VARIANTS FOR THIS SEED (avoid these patterns entirely):\n{rejected_block}"
    messages = [{"role": "user", "content": system_prompt}]
    toolset = _similar_agent_tools()

    _log("agent_start", topic=source_q["topic"], band=source_q["band"],
         max_loops=max_loops)

    candidate_id: str | None = None

    for step in range(max_loops):
        try:
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
                fail_model="",
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

    results: list[dict] = []
    idx = 0

    for q in source_questions:
        for attempt in range(n_each):
            idx += 1
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
                log_fn=log_fn,
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
            results.append(
                {
                    "source_id": q["id"],
                    "candidate_id": candidate_id or "",
                    "status": status,
                    "pass_rate": rate,
                }
            )

    n_pending = sum(1 for r in results if r["status"] == "pending")
    _log(
        "similar_end",
        n_total=total,
        n_pending=n_pending,
        n_rejected=total - n_pending,
    )
    return results


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
    rejected_stems: list[str] | None = None,
) -> dict:
    """Call the reviewer LLM. Returns {score, passed, criterion_scores, issues, fix_instructions}."""
    prior_block = (
        "\n".join(f"  Variant {i+1}: {s[:200]}" for i, s in enumerate(prior_stems))
        if prior_stems else "  (none yet)"
    )
    rejected_block = (
        "\n".join(f"  Rejected {i+1}: {s[:200]}" for i, s in enumerate(rejected_stems or []))
        if rejected_stems else "  (none)"
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
        rejected_variants=rejected_block,
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

    results: list[dict] = []
    idx = 0

    for q in source_questions:
        prior_stems: list[str] = []   # track variants generated for this seed

        # Fetch SME-rejected variants for this source question (rejection memory)
        rejected_stems = _get_rejected_variants(q["id"])

        for attempt in range(n_each):
            idx += 1
            _log("similar_generate", index=idx, total=total,
                 source_id=q["id"], topic=q["topic"], band=q["band"],
                 attempt=attempt, stem_preview=(q.get("stem") or "")[:120])

            # ── Generate → Review → Fix loop ─────────────────────────────
            current_raw: str | None = None
            review_result: dict = {}
            passed = False

            def _prior_block(stems: list[str], label: str = "Variant") -> str:
                return "\n".join(f"{label} {i+1}: {s[:200]}" for i, s in enumerate(stems)) or "(none)"

            def _rejected_block(stems: list[str]) -> str:
                return "\n".join(f"Rejected {i+1}: {s[:200]}" for i, s in enumerate(stems)) or "(none — no prior rejections)"

            for fix_iter in range(max_fix_iterations + 1):
                is_fix = fix_iter > 0

                if not is_fix:
                    prompt = SIMILAR_QUESTION_PROMPT.format(
                        topic=_label_or_infer(q.get("topic"), "topic"),
                        band=_label_or_infer(q.get("band"), "band"),
                        stem=q["stem"],
                        seed_key=q.get("key", ""),
                        prior_variants=_prior_block(prior_stems),
                        rejected_variants=_rejected_block(rejected_stems),
                    )
                    _log("review_generate", index=idx, fix_iter=0,
                         source_id=q["id"],
                         n_rejected_variants=len(rejected_stems))
                else:
                    prompt = SIMILAR_FIX_PROMPT.format(
                        seed_stem=q["stem"][:600],
                        seed_key=q.get("key", ""),
                        previous_stem=_extract_stem_from_raw(current_raw or ""),
                        score=review_result.get("score", 0),
                        issues="; ".join(review_result.get("issues", [])),
                        fix_instructions=review_result.get("fix_instructions", ""),
                        prior_variants=_prior_block(prior_stems),
                        rejected_variants=_rejected_block(rejected_stems),
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

                sandbox_result = execute(lang_gen, code_gen, files_gen, twice=True)
                _log("review_sandbox", index=idx, fix_iter=fix_iter,
                     ok=sandbox_result.ok, reason=sandbox_result.reason)

                if not sandbox_result.ok:
                    # Sandbox failed — treat as review failure with fix instructions
                    review_result = {
                        "score": 0, "passed": False,
                        "issues": [f"Sandbox error: {sandbox_result.reason}"],
                        "fix_instructions": "Fix the code so it runs without errors and produces a deterministic answer.",
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
                    rejected_stems=rejected_stems,
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
            results.append({
                "source_id": q["id"],
                "candidate_id": cid or "",
                "status": status,
                "pass_rate": rate,
            })

    n_pending = sum(1 for r in results if r["status"] == "pending")
    _log("similar_end", n_total=total, n_pending=n_pending,
         n_rejected=total - n_pending)
    return results


def _get_rejected_variants(source_qid: str, limit: int = 10) -> list[str]:
    """Return stems of candidates previously rejected for this source question.

    Used to populate the 'avoid these patterns' section of generation prompts
    so the generator doesn't repeat what an SME already found bad.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT stem FROM candidates
                WHERE status = 'rejected'
                  AND source_fail_ids LIKE :pat
                ORDER BY created_at DESC
                LIMIT :lim
                """
            ),
            {"pat": f"%{source_qid}%", "lim": limit},
        ).scalars().all()
    return [s for s in rows if s]


def _extract_stem_from_raw(raw: str) -> str:
    """Extract just the stem from a raw generator response for use in fix prompts."""
    try:
        return _extract_json(raw).get("stem", raw[:400])
    except Exception:
        return raw[:400]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--mode", choices=["one_shot", "agent"], default="one_shot")
    p.add_argument("--split", choices=["train", "eval"], default="train")
    args = p.parse_args()
    if args.mode == "agent":
        from frontier_lens.agent.loop import propose_agent

        ids = propose_agent(args.run, args.model, args.n, rl_split=args.split)
    else:
        ids = propose_one_shot(args.run, args.model, args.n, rl_split=args.split)
    print("candidates", ids)


if __name__ == "__main__":
    main()
