# Frontier Lens

A small **RL data engine** for open-weight models. It generates verifiable tasks calibrated to a policy model's edge, hardens exact-match reward, audits quality with humans, and keeps a clean held-out eval.

Models predict the output of short programs. They must not run code. Answers end with `FINAL: <answer>`. Keys come from a local sandbox, never from a model's claim. Reward is `exact_match_v1` (last `FINAL:` line only).

**Claim:** the pipeline produces **RL-ready** JSONL (prompts, gold, pinned reward) under human audit.  
**Not claimed:** that RL training improved the policy. That would be **RL-proven**, and it was not run unless the experiment block below says otherwise.

## Who it is for

People who need diagnostic, execution-verified tasks at a policy's learnable band, plus a held-out eval the generator did not fit to that policy.

## Prior art

[Dynabench](https://dynabench.org/), refreshing/live benchmarks, adversarial collection, [Terminal-Bench](https://www.tbench.ai/) (execution-graded tasks), and open RL trainers such as [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer).

**What is different:** generation targets the policy pass-rate band `0 < pass_rate < 1`; keys are executed; eval holds out whole templates; humans audit a sample (Wilson 95% CI) rather than every item; exports load into TRL as `{prompt, ground_truth, reward_fn}`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11. No API keys for seed, tests, the Streamlit viewer, or fixture replay.

```bash
uv python pin 3.11
uv sync --group dev
cp .env.example .env
uv run python -m frontier_lens.seed
uv run pytest
uv run streamlit run app/Home.py
```

The app opens at `http://localhost:8501`. Pages: **Contest**, **Weaknesses** (policy pass-rate histogram + heatmap), **Generate + Review** (audit sample + suspicious queue), **Results** (band share, headroom, audit CI, one-shot vs agent). Seed before Contest. The sidebar role is a switch, not login.

## Live local models (Ollama)

Optional. Local weights are Ollama GGUF **Q4_K_M**. The default **policy** is `gemma4-e4b` (`gemma4:e4b`) talking to `http://localhost:11434/v1`. A connection refused error means Ollama is not running.

```bash
ollama serve
ollama pull gemma4:e4b
uv run python -m frontier_lens.contest --live --models gemma4-e4b --write-fixtures
```

Policy uses k=8; references use k=3 unless you pass `--k`. `--write-fixtures` saves answers to `fixtures/recorded_run.json`. Replay does not call a model:

```bash
uv run python -m frontier_lens.contest --replay --models gemma4-e4b
```

In the app, set Mode to **live** only when Ollama is up (sidebar **Ollama ok**). Role must be Organizer or SME.

## Generate, audit, export

Train split may sample the policy (`estimate_pass_rate`, k=8) and aims for the learnable band. Eval split never calls the policy.

```bash
uv run python -m frontier_lens.generate --run RUN_ID --model gemma4-e4b --mode one_shot --split train
uv run python -m frontier_lens.generate --run RUN_ID --model gemma4-e4b --mode one_shot --split eval
uv run python -m frontier_lens.export
```

JSONL lands in `data/exports/`:

- `rl_tasks.jsonl` — accepted train items, `reward_fn: "exact_match_v1"`, `split: "train"`
- `eval_tasks.jsonl` — same schema **without** `policy_pass_rate` / `pass_at_k`
- `sft_rejection_sampled.jsonl` — correct completions from policy and reference samples

These files are meant to load into TRL GRPO (prompt + reward on `ground_truth` via exact match). Review sampled candidates in the app as SME before treating a generated batch as accepted.

## Seed bank

104 original template items (8 L1 harness, 48 L2, 48 L3) across `shell_pipelines`, `python_state`, `multi_step_units`, `constraint_counting`. Eval holds out whole templates (`state_l3`, `count_l2`). `uv run python -m frontier_lens.seed` is non-interactive.

## Roster

**Policy:** Gemma 4 e4b (local Ollama Q4). **References:** Gemma 4 12B, Qwen 3.8 27B (skip 27B if it does not fit), DeepSeek V4.1 Flash, Llama 4 Scout. **Generator:** GLM-5.3 (family disjoint from policy and references; not evaluated). Licenses are in `models.yaml`. Verify tags before quoting.

## Tests and CI

Parser; last-FINAL reward; sandbox bans; pass@k; learnable-band edges; template-split leakage; Wilson audit math; export Pydantic schema; promotion gate; registry family split; **re-execute every seed key**. GitHub Actions: `uv sync --frozen`, ruff, pytest, fail if `.env` is tracked.

## Experiment

See [`results/experiment.md`](results/experiment.md) for the recorded numbers. Summary of what was actually produced:

| Field | Value |
|---|---|
| Policy model | `gemma4-e4b` (configured). **No live k=8 samples** — Ollama was not running. |
| Items generated per mode | one-shot 0, agent 0 |
| Learnable-band share per mode | n/a (no policy samples) |
| Headroom (pass@8 − pass@1) | n/a |
| Audit defect rate (Wilson 95% CI) | n/a (no generated batch) |
| Held-out eval result | 24 seed items exported (`state_l3`, `count_l2`); **not scored** |
| JSONL | 72 train + 24 eval + 0 SFT rows in `results/` |
| Training run | **None.** RL-ready, not RL-proven. |

Do not hand-author fixture weaknesses. Record `fixtures/recorded_run.json` from a real `--live --write-fixtures` run.

## Limitations

- Evidence is **proxy** (pass rates, band share, audit CI), not a training delta.
- Audit sample is small (default 50); Wilson intervals are wide.
- Single SME (the builder). Agreement is unreported unless a second annotator labels a subset.
- AST/shlex sandbox is not production isolation; BSD vs GNU tools; local Q4 is not hosted precision; no closed-model reference.
- Guessable small-integer answer spaces are flagged in Results, not removed automatically.

## Time spent

To be filled at the end of the build window (target 12–16 hours).
