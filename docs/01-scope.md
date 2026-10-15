# Scope

Two-day, experiment-first. The finding is the artifact. Streamlit is a thin viewer over the same library the CLI uses.

## In

- 104 original seed items: L1 harness (~8), L2 standard (48), L3 hard (48).
- 4 topics × 2 heatmap bands (L2/L3) = 8 cells. L1 is not on the heatmap.
- SQLite on disk. Reviewers run `uv sync`.
- Open-weight models only, via one OpenAI-compatible adapter plus fixture replay.
- Local models: Ollama GGUF, default **Q4_K_M**. Not bitsandbytes, not vLLM on this laptop.
- One-shot generator; agent generator if time (cut last among optional features).
- Blind SME review, recorded live fixtures, light CI.

## Out

- Auth, OAuth, real RBAC (sidebar role switch).
- Fine-tuning.
- Postgres, Docker, Compose, Alembic, vLLM, Hugging Face `transformers` loaders.
- Evaluated-agent track, image questions, interactive questions.
- Hosting weights inside Streamlit Cloud.
- Closed-source models at any pipeline stage.

## Cut order if short on time

1. Fifth evaluated model (Kimi).
2. Llama 4 Scout.
3. Background Evaluate from the Streamlit dropdown (CLI still runs contests).
4. Agent generator (ship one-shot).

**Never cut:** sandbox, executed keys, holdout split, SME gate, recorded fixtures, tests.

## Hardware

This machine is Intel macOS. Qwen 3.8 27B even at Q4 may not load. If `ollama pull` fails or RAM is tight, local roster is Gemma 4 12B or `e4b` plus a small third family. Hosted open-weight APIs are optional and need a key later, not for seed.

## Default commands

```bash
uv sync
uv run python -m frontier_lens.seed
uv run python -m frontier_lens.contest --live --k 3
uv run streamlit run app/Home.py
uv run pytest
```
