# Vision

Frontier Lens is a small RL data engine for open-weight models. It generates verifiable tasks calibrated to a policy model's edge, hardens the reward, audits quality with humans, and keeps a clean held-out eval.

## Who it is for

People who need **RL-ready** task files (prompts, sandbox keys, a pinned reward function) — not another static leaderboard, and not a claim that RL already worked.

## The problem

Static eval sets go stale and do not say *where* a policy fails. Generated items that skip review are often broken. Reward that accepts hedges or the first `FINAL:` line is hackable. Holding out individual items leaks templates into eval. Closed APIs make reruns hard to pin.

## What we build

1. Models answer **execution-verified** tasks (they must not run code). The key is sandbox stdout. Counting and unit items state the problem in prose; Python and shell items show the program.
2. One open model is the **policy** (k=8). Other open models are **references** (k=3). The generator family is disjoint from both.
3. Train items are calibrated so the policy's pass rate sits in the **learnable band** `0 < pass_rate < 1`. Headroom is `pass@8 − pass@1`.
4. **Eval holds out whole templates** (or disjoint parameter regions). The eval generator never calls the policy.
5. Reward is binary exact match on the **last** `FINAL:` line (`exact_match_v1`). Hedges and multi-answers score 0.
6. Humans **audit a random sample** of a generated batch (Wilson 95% CI on the defect rate) and review correct-but-suspicious policy traces.
7. Exports: `rl_tasks.jsonl`, `eval_tasks.jsonl`, `sft_rejection_sampled.jsonl` for open trainers such as TRL GRPO.

## Why not seed from an existing benchmark

- **ARC-AGI-3** — interactive game environments, not questions.
- **Humanity's Last Exam** — copying breaks originality; contamination risk.
- **FrontierMath Tier 4** — private by design.
- **Terminal-Bench 2.0** — Docker sandboxes and agent loops; too few tasks in two days for filled cells.

Lessons: static sets saturate; Terminal-Bench's real strength is **ground truth from execution**, which we borrow in a **single-turn** format.

**Rule:** all items are original templates. No questions, assets, or code copied from other benchmarks or projects.

## Prior art

- **Dynabench** — humans write examples that fool models.
- Refreshing / live benchmarks.
- Adversarial data collection.
- **Terminal-Bench** — execution-graded tasks.
- Open RL trainers (**TRL GRPO**) — we export a loadable JSONL; we do not claim a training win.

**What is different:** generation is calibrated to a measured policy pass rate; keys come from execution, not the generator's claim; audit sampling estimates batch defect rate; eval templates never see the policy.

## Claims

- **Claim:** the system produces **RL-ready** task files under human audit, with a held-out eval that was not fit to the policy.
- **Do not claim:** that a policy was improved (that would be **RL-proven**, and needs a training run on `eval_tasks.jsonl`).

Open-weight only: no closed-source model evaluates, generates, or assists in the pipeline. Tradeoff: no closed frontier reference.
