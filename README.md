# Frontier Lens

Execution-verified eval for **open-weight** models, plus a human-gated way to grow the set aimed at measured failures.

Models predict the output of short programs. They must not run code. Answers end with `FINAL: <answer>`. Keys come from a local sandbox, not from a generator's claim. An SME decides whether a candidate is hard, broken, or trivial.

## Who it is for

Eval engineers and reviewers who need diagnostic items with provenance, not another static leaderboard.

## Prior art

[Dynabench](https://dynabench.org/) (humans write examples that fool models), refreshing/live benchmarks, adversarial collection, and [Terminal-Bench](https://www.tbench.ai/) (execution-graded tasks).

**What is different:** generation targets a measured heatmap cell; keys are executed; a human gate checks fairness and novelty; every item stores template, split, generator family/mode, and source failures.

**Claim:** the system grows a diagnostic set under human control.  
**Not claimed:** that the generator improves models, or that these items measure general coding ability.

## Setup

Requires [uv](https://docs.astral.sh/uv/). No API keys for seed, tests, or fixture replay.

```bash
uv python pin 3.11
uv sync --group dev
uv run python -m frontier_lens.seed
uv run pytest
uv run streamlit run app/Home.py
```

Live local models (optional, Ollama GGUF Q4_K_M — not bitsandbytes, not vLLM on a laptop):

```bash
ollama pull gemma4:12b
uv run python -m frontier_lens.contest --live --k 3 --write-fixtures
```

Hosted open weights (optional) use `OPENROUTER_API_KEY` in `.env` (see `.env.example`). Streamlit Cloud cannot host GGUF; point `base_url` at an inference host or use replay.

## Seed bank

104 original template items (8 L1 harness, 48 L2, 48 L3) across `shell_pipelines`, `python_state`, `multi_step_units`, `constraint_counting`. L2/L3 cells split 6 selection / 6 baseline. `uv run python -m frontier_lens.seed` is non-interactive.

## Roster

Evaluated: Gemma 4 12B and Qwen 3.8 27B (local Ollama Q4; skip 27B if it does not fit), DeepSeek V4.1 Flash and Llama 4 Scout (hosted). Generator: GLM-5.3 (not evaluated). Licenses are listed in `models.yaml`. Verify tags before quoting.

## Tests and CI

Parser, sandbox bans, Wilson CI, promotion gate, registry family split, **re-execute every seed key**. GitHub Actions: `uv sync --frozen`, ruff, pytest, fail if `.env` is tracked.

## Experiment

See [`results/experiment.md`](results/experiment.md) after a real live run. Do not hand-author fixture weaknesses.

## Limitations

Wide CIs at n=6–12; single SME who also designed the generators (disclosed); AST/shlex sandbox is not production isolation; BSD vs GNU tools; local Q4 is not the same as hosted precision; no closed-model reference.

## Time spent

To be filled at the end of the build window (target 12–16 hours).
