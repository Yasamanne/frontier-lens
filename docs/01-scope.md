# Scope

Two-day, experiment-first. The finding is the artifact. Streamlit is a thin viewer over the same library the CLI uses.

## In

- 104 procedural seed items: L1 harness (8), L2 (48), L3 (48). L2/L3 knobs come from `calibrate.py` (live policy rates in production; synthetic probe in CI). No copied benchmark items.
- 4 topics. Train vs eval is a **template-level** split: each family holds out seeds 10000–10003 as `{topic}_eval`, disjoint from train seeds 0–7.
- SQLite on disk. Reviewers run `uv sync`.
- Open-weight models only, via one OpenAI-compatible adapter plus fixture replay.
- Local models: Ollama GGUF, default **Q4_K_M**. Not bitsandbytes, not vLLM on this laptop.
- Policy default: Gemma 4 26B-A4B via OpenRouter. References and a disjoint-family generator in `models.yaml`.
- One-shot vs agent generators; train split may call `estimate_pass_rate`; eval split must not.
- Audit sampling (default n=50, Wilson 95%, accept if upper bound < 10%) plus a suspicious-correct queue.
- JSONL export for TRL GRPO / SFT.

## Out

- Auth, OAuth, real RBAC (sidebar role switch).
- Closed-source models at any pipeline stage.
- Postgres, Docker, Compose, Alembic, vLLM, Hugging Face `transformers` loaders (except an optional stretch LoRA run).
- Evaluated-agent track, image questions, interactive questions.
- Hosting weights inside Streamlit Cloud.
- Claiming RL-proven without a before/after eval measurement.

## Stretch (only after tests pass and JSONL exists)

Tiny LoRA + GRPO on a small open model using `rl_tasks.jsonl`, measured on `eval_tasks.jsonl`. Report honestly, including a null result.

## Cut order if short on time

1. Fifth reference model.
2. Llama 4 Scout.
3. Background Evaluate from the Streamlit dropdown (CLI still runs contests).
4. Agent generator (ship one-shot + post-filter).

**Never cut:** sandbox, executed keys, template holdout, audit gate, recorded fixtures, tests, open-weight-only.

## Hardware

This machine is Intel macOS. Qwen 3.8 27B even at Q4 may not load. Default policy is Gemma 4 e4b. Hosted open-weight APIs are optional and need a key later, not for seed.

## Default commands

```bash
uv sync
uv run python -m frontier_lens.seed
uv run python -m frontier_lens.contest --live --models gemma4-26b-a4b --write-fixtures
uv run python -m frontier_lens.export
uv run streamlit run app/Home.py
uv run pytest
```
