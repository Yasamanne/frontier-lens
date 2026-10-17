from __future__ import annotations

from frontier_lens.config import POLICY_K
from frontier_lens.prompt import contest_prompt
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import policy_spec
from frontier_lens.reward import grade
from frontier_lens.stats import in_learnable_band, pass_at_k, pass_rate


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
