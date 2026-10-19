from __future__ import annotations

# A generated question passes review at this total score (five criteria, 20 points each).
# The code compares the score with it; the reviewer model does not decide pass or fail.
REVIEW_PASS_SCORE = 75

MAX_FIX_ITERATIONS = 3  # 3 fix iterations is enough for our shorter questions

SIMILAR_REVIEW_PROMPT = """\
You are a strict question quality reviewer. Score the GENERATED question on
5 criteria (20 pts each, max 100). Pass threshold: {pass_score}.

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

══ HUMAN REVIEWER (SME) FEEDBACK — what SMEs rejected and approved before ══
{sme_feedback}

══ CRITERIA ══

1. CONCEPT ALIGNED (0–20): Tests the same skill/domain as the seed. Same type
   of problem. The difficulty may differ from the seed: do NOT take points off
   for a harder or an easier variant.

2. GENUINELY DISTINCT (0–20): Meaningfully different from the seed AND all
   prior variants. Changing only numbers = 0 pts. New scenario/structure = 20 pts.

3. ANSWER UNIQUENESS (0–20): Exactly one correct answer. A knowledgeable
   solver would always reach the same answer.

4. STEM CLARITY (0–20): Unambiguous question. Clear inputs, clear what is
   being asked. No missing information.

5. SELF-CONTAINED AND CONSISTENT (0–20): Everything needed is in the stem; it
   can be solved by reasoning alone (no outside data or tools); and the question
   asked is exactly what the verifier code computes.

DIFFICULTY IS NOT SCORED. It is measured separately by sampling the policy model,
so do not score it and do not list difficulty as an issue.

SME feedback overrides your own taste: if the generated question repeats a
problem SMEs flagged above, it must not pass (score at most 70) and that
problem must appear in "issues".

Return JSON only:
{{
  "score": <0-100 integer>,
  "passed": <true if score >= {pass_score}>,
  "criterion_scores": {{
    "concept_aligned": <0-20>,
    "genuinely_distinct": <0-20>,
    "answer_uniqueness": <0-20>,
    "stem_clarity": <0-20>,
    "self_contained": <0-20>
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

HUMAN REVIEWER (SME) FEEDBACK — the fix must not reintroduce any of these problems:
{sme_feedback}

Apply the fix instructions and produce an improved question.
Keep what was good; fix only what the reviewer flagged.

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true, "topic": "...", "band": "standard" or "hard"}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf awk;
python may import math, itertools, collections, decimal only.
If topic or band are empty or unknown, infer them from the seed question:
  topic: a short snake_case label for the subject. Reuse the seed's own topic when it has one;
  otherwise name the subject (for example algebra, number_theory, combinatorics, geometry,
  python_state, shell_pipelines).
  band must be one of: standard, hard.
The subject can be anything, including mathematics. For maths the code must compute the exact
answer itself (counting, number theory, combinatorics, exact integer or decimal arithmetic).
Do not write questions whose answer is a proof, an expression, an interval or a symbolic value,
and do not rely on libraries that are not allowed.
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

HUMAN REVIEWER (SME) FEEDBACK ON EARLIER HINTS — do not repeat these problems:
{sme_feedback}

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

HUMAN REVIEWER (SME) FEEDBACK — avoid what SMEs rejected, aim for what they approved:
{sme_feedback}

The new question must have a unique, deterministic answer verifiable by running code.
The policy model should sometimes pass and sometimes fail (0 < pass_rate < 1).

Return JSON only:
{{"stem": "...", "lang": "python" or "shell", "code": "...", "input_files": {{}}, "numeric": true, "topic": "...", "band": "standard" or "hard"}}
Rules: single-line answer; shell allowlist cat cut sort uniq wc head tail tr grep printf awk;
python may import math, itertools, collections, decimal only. No network, no files outside input_files.
If Topic or Band above are empty or unknown, infer them from the seed question:
  topic: a short snake_case label for the subject. Reuse the seed's own topic when it has one;
  otherwise name the subject (for example algebra, number_theory, combinatorics, geometry,
  python_state, shell_pipelines).
  band must be one of: standard, hard.
The subject can be anything, including mathematics. For maths the code must compute the exact
answer itself (counting, number theory, combinatorics, exact integer or decimal arithmetic).
Do not write questions whose answer is a proof, an expression, an interval or a symbolic value,
and do not rely on libraries that are not allowed.
"""

SIMILAR_AGENT_SYSTEM = """\
You are a question designer. Your task: write ONE new question that is similar in \
style and difficulty to the SEED question below, but is genuinely distinct from it.

TOPIC: {topic}
BAND:  {band}

SEED QUESTION:
{stem}

HUMAN REVIEWER (SME) FEEDBACK — avoid what SMEs rejected, aim for what they approved:
{sme_feedback}

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
    topic: a short snake_case label for the subject. Reuse the seed's own topic when it has one;
    otherwise name the subject (for example algebra, number_theory, combinatorics, geometry,
    python_state, shell_pipelines).
    band must be one of: standard, hard.
  - The subject can be anything, including mathematics. For maths the code must compute the
    exact answer itself (counting, number theory, combinatorics, exact integer or decimal
    arithmetic). Do not write questions whose answer is a proof, an expression, an interval
    or a symbolic value, and do not rely on libraries that are not allowed.
  - If TOPIC and BAND are already set above, you do NOT need to include them in submit_candidate.
"""

ONE_SHOT_TRAIN = """You write a NEW original train item for open-weight models.
Topic: {topic}
Band: {band}

=== HOW THE POLICY FAILS ON THIS CELL ===
{failure_analysis}

=== FAILING QUESTION STEMS (do NOT clone, use for inspiration only) ===
{failures}

=== HUMAN REVIEWER (SME) FEEDBACK — avoid what SMEs rejected, aim for what they approved ===
{sme_feedback}

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


SIMILAR_AGENT_REVIEW_SYSTEM = """\
You are a question designer. Your task: write ONE new question that is similar in \
style and difficulty to the SEED question below, but is genuinely distinct from it. \
A strict reviewer will score your draft; if it fails you will get its feedback and revise.

TOPIC: {topic}
BAND:  {band}

SEED QUESTION:
{stem}

Gold answer of the seed (known to you: use it to match difficulty, do NOT reveal it): {seed_key}

PRIOR VARIANTS of this seed (yours must differ from ALL of them):
{prior_variants}

HUMAN REVIEWER (SME) FEEDBACK — what SMEs rejected and approved before. Do not repeat \
the problems below; aim for what they approved:
{sme_feedback}

HOW THE POLICY MODEL (the model being trained) BEHAVES ON THIS SEED. Aim your question at
this weakness:
{policy_weakness}

TARGET DIFFICULTY: the policy should solve your question sometimes but not always (pass rate
strictly between 0 and 1); questions it always or never solves are discarded. Ways to raise
difficulty:
{hardening_levers}

AVAILABLE TOOLS (at most {max_loops} tool-call rounds per draft; the last round must be
submit_draft, so leave room for it):
  run_snippet      — run code in the sandbox; use it to check your verifier code
  check_similarity — check your stem is not too close to existing questions
  submit_draft     — send your draft for review. Call it ONCE per draft.

WORKFLOW:
  1. Draft a new question and its verifier code.
  2. Call run_snippet: the code must run and print exactly one non-empty line, the same \
on every run.
  3. Call submit_draft. The reviewer then scores it; if it fails, its feedback arrives as \
the next message and you revise (test again with run_snippet if you change the code).

RULES:
  - The answer must be a single bare value (numeric or short string), printed on one line.
  - Shell code: only cat cut sort uniq wc head tail tr grep printf awk.
  - Python: only math, itertools, collections, decimal. No network, no external files.
  - If TOPIC or BAND above are empty, infer them from the seed and include them in submit_draft.
    topic: a short snake_case label for the subject. Reuse the seed's own topic when it has one;
    otherwise name the subject (for example algebra, number_theory, combinatorics, geometry,
    python_state, shell_pipelines).
    band must be one of: standard, hard.
  - The subject can be anything, including mathematics. For maths the code must compute the
    exact answer itself (counting, number theory, combinatorics, exact integer or decimal
    arithmetic). Do not write questions whose answer is a proof, an expression, an interval
    or a symbolic value, and do not rely on libraries that are not allowed.
"""

SIMILAR_AGENT_FIX_MESSAGE = """\
REVIEWER FEEDBACK — your draft FAILED (score {score}/100, pass mark {pass_score}).
Issues: {issues}
Fix instructions: {fix_instructions}

Revise the question: keep what was good and change only what the reviewer flagged. The fix \
must not reintroduce any problem from the SME feedback above. If you change the code, test \
it with run_snippet, then call submit_draft again.
"""


HARDENING_LEVERS = """\
  - add one more constraint or condition the solver has to keep track of
  - add reasoning steps: make the answer depend on a longer chain
  - combine two ideas from the topic in one question
  - enlarge the input or the bounds (your code still computes it exactly; by hand it gets error-prone)
  - add boundary cases, off-by-one traps or a rare special case
  - add realistic distractor details that do not change the answer
  - replace a closed-form shortcut with a case analysis"""

SIMILAR_TUNE_HARDER_MESSAGE = """\
CALIBRATION RESULT: the policy model solved your question {correct}/{n} times (pass rate \
{rate:.2f}). It is too easy: RL needs questions the policy solves sometimes but not always.

Make a HARDER version that still has exactly one unambiguous answer. Ways to raise difficulty:
{hardening_levers}
Keep what makes the question clear and change only what raises difficulty. Test the new code \
with run_snippet, then call submit_draft again.
"""

SIMILAR_TUNE_EASIER_MESSAGE = """\
CALIBRATION RESULT: the policy model solved your question 0/{n} times. Its answers were: \
{policy_answers}. Your code computed: {key}.

First check that the question is correct: does the stem ask exactly what the code computes, \
with one unambiguous answer? If something is wrong, fix it. If it is correct but simply too \
hard, make an EASIER version: fewer steps or constraints, a smaller input, or remove a trap. \
Test the new code with run_snippet, then call submit_draft again.
"""
