PROMPT_PREFIX = """You will be shown a short program and its input. Do not run anything.
Predict the result and end your reply with exactly one line:
FINAL: <answer>
"""


def contest_prompt(stem: str) -> str:
    return f"{PROMPT_PREFIX}\n{stem}\n"
