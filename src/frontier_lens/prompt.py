PROMPT_PREFIX = """\
Solve the task below. Reason step by step. Do NOT run code or use tools.

══════════════════════════════════════════════════════
MANDATORY FORMAT — failure to follow = score ZERO
══════════════════════════════════════════════════════
Your response MUST end with this line, EXACTLY as shown:

    FINAL: <your answer>

Rules (each violation scores 0):
  1. "FINAL:" must be the VERY LAST LINE — nothing written after it.
  2. After "FINAL: " write only the bare answer — no units, no explanation.
  3. Numeric answers: digits only  (e.g.  FINAL: 42   or   FINAL: 0027).
  4. Write "FINAL:" exactly ONCE.
  5. One definite answer only — never "42 or 43", never "either A or B".

Correct example
───────────────
Task: What is 6 × 7?
Reasoning: 6 × 7 = 42.
FINAL: 42
───────────────
"""

CORRECTION_PROMPT = """\
Your previous response did not include the required FINAL: line, or it was not \
the last line. Respond now with ONLY the following — nothing else:
FINAL: <your answer>
"""


def contest_prompt(stem: str) -> str:
    return f"{PROMPT_PREFIX}\n{stem}\n"
