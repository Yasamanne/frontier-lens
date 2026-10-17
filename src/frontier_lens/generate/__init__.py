"""frontier_lens.generate — question-generation pipeline.

Public API is re-exported here so all existing import paths continue to work:

    from frontier_lens.generate import estimate_pass_rate
    from frontier_lens.generate import propose_one_shot
    from frontier_lens.generate import rescue_with_hints, generate_rescue_sft_csv
    from frontier_lens.generate import generate_similar_questions
    from frontier_lens.generate import generate_similar_react
    from frontier_lens.generate import generate_similar_with_review

Internal sub-modules:
    .utils        — JSON extraction, string helpers
    .prompts      — all prompt templates and constants
    .calibration  — estimate_pass_rate, _calibrate_collect
    .candidates   — _store_candidate, _store_hinted_candidate, _get_rejected_variants
    .oneshot      — propose_one_shot, _eval_cells, _failures
    .rescue       — rescue_with_hints, generate_rescue_sft_csv
    .similar      — generate_similar_questions/react/with_review
"""

from .calibration import _calibrate_collect, estimate_pass_rate
from .candidates import _get_rejected_variants, _store_candidate, _store_hinted_candidate
from .oneshot import _eval_cells, _failure_analysis_text, _failures, propose_one_shot
from .prompts import (
    _LEVEL_INSTRUCTIONS,
    LIGHT_HINT_PROMPT,
    MAX_FIX_ITERATIONS,
    ONE_SHOT_EVAL,
    ONE_SHOT_TRAIN,
    PROGRESSIVE_HINT_PROMPT,
    SIMILAR_AGENT_SYSTEM,
    SIMILAR_FIX_PROMPT,
    SIMILAR_QUESTION_PROMPT,
    SIMILAR_REVIEW_PROMPT,
)
from .rescue import generate_rescue_sft_csv, rescue_with_hints
from .similar import (
    _review_similar_question,
    _run_similar_agent,
    _similar_agent_tools,
    generate_similar_questions,
    generate_similar_react,
    generate_similar_with_review,
)
from .utils import _extract_json, _extract_stem_from_raw, _label_or_infer

__all__ = [
    "LIGHT_HINT_PROMPT",
    "MAX_FIX_ITERATIONS",
    "ONE_SHOT_EVAL",
    "ONE_SHOT_TRAIN",
    "PROGRESSIVE_HINT_PROMPT",
    "SIMILAR_AGENT_SYSTEM",
    "SIMILAR_FIX_PROMPT",
    "SIMILAR_QUESTION_PROMPT",
    # Prompts
    "SIMILAR_REVIEW_PROMPT",
    "_LEVEL_INSTRUCTIONS",
    "_calibrate_collect",
    "_eval_cells",
    # Utils
    "_extract_json",
    "_extract_stem_from_raw",
    "_failure_analysis_text",
    "_failures",
    "_get_rejected_variants",
    "_label_or_infer",
    "_review_similar_question",
    "_run_similar_agent",
    "_similar_agent_tools",
    # Candidates
    "_store_candidate",
    "_store_hinted_candidate",
    # Calibration
    "estimate_pass_rate",
    "generate_rescue_sft_csv",
    # Similar
    "generate_similar_questions",
    "generate_similar_react",
    "generate_similar_with_review",
    # One-shot
    "propose_one_shot",
    # Rescue / hints
    "rescue_with_hints",
]


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
