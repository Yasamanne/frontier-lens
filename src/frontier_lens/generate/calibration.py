from __future__ import annotations

from contextlib import contextmanager

from frontier_lens.config import POLICY_K
from frontier_lens.parallel import map_unordered
from frontier_lens.prompt import contest_prompt
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import model_spec
from frontier_lens.reward import grade
from frontier_lens.stats import in_learnable_band, pass_at_k, pass_rate

_ACTIVE_MODEL: str | None = None


@contextmanager
def calibrating_with(model_id: str | None):
    """Calibrate on *model_id*, the run's primary model, for everything inside the block.

    A job is its own process, so one process-wide setting is enough and it reaches the
    sampling threads. An explicit ``model_id=`` argument still wins.
    """
    global _ACTIVE_MODEL
    previous, _ACTIVE_MODEL = _ACTIVE_MODEL, model_id
    try:
        yield
    finally:
        _ACTIVE_MODEL = previous


def _calibration_spec(model_id: str | None):
    return model_spec(model_id or _ACTIVE_MODEL)


def _policy_complete(provider, policy, prompt: str) -> str:
    """One policy sample with the same settings Evaluate uses (thinking flag, output cap)."""
    return provider.complete(
        prompt, policy.model, thinking=bool(policy.thinking), max_tokens=policy.max_tokens
    ).text


def estimate_pass_rate(
    stem: str,
    key: str,
    numeric: bool = True,
    *,
    complete_fn=None,
    k: int | None = None,
    log_fn=None,
    model_id: str | None = None,
) -> dict:
    """Sample the policy k times (concurrently). Train-split calibration only — never call on eval.

    If *log_fn* is provided it is called once per sample with::

        log_fn("calibration_sample", sample_index=i, is_correct=bool, reward=int)

    Samples finish out of order; *log_fn* runs on the caller's thread.
    """
    policy = _calibration_spec(model_id)
    n = k if k is not None else POLICY_K
    if complete_fn is None:
        provider = OpenAICompatibleProvider(policy.base_url or "", policy.api_key_env)

        def complete_fn(prompt: str) -> str:
            return _policy_complete(provider, policy, prompt)

    def _sample(_i: int) -> int:
        reward, _, _ = grade(complete_fn(contest_prompt(stem)), key, numeric)
        return reward

    correct = 0
    for i, reward in map_unordered(_sample, range(n)):
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
    model_id: str | None = None,
) -> dict:
    """Like ``estimate_pass_rate`` but also returns individual raw responses.

    Returns::

        {
          "n": k, "correct": int, "pass_rate": float, "pass_at_k": float,
          "in_learnable_band": bool, "policy_model": str,
          "responses": [{"raw": str, "parsed": str, "is_correct": bool}]
        }
    """
    policy = _calibration_spec(model_id)
    if complete_fn is None:
        provider = OpenAICompatibleProvider(policy.base_url or "", policy.api_key_env)

        def complete_fn(prompt: str) -> str:
            return _policy_complete(provider, policy, prompt)

    def _sample(_i: int) -> tuple[str, int, str | None]:
        raw = complete_fn(contest_prompt(stem))
        reward, parsed_ans, _ = grade(raw, key, numeric)
        return raw, reward, parsed_ans

    by_index: dict[int, dict] = {}
    correct = 0
    for i, (raw, reward, parsed_ans) in map_unordered(_sample, range(k)):
        correct += reward
        by_index[i] = {"raw": raw, "parsed": parsed_ans or "", "is_correct": bool(reward)}
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
        "responses": [by_index[i] for i in range(k)],
    }
