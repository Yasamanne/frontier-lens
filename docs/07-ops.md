# Ops

## Env

Copy `.env.example` to `.env` (gitignored). Optional: `OPENROUTER_API_KEY` for hosted open weights. `OLLAMA_BASE_URL` defaults to `http://localhost:11434/v1`. `DATABASE_URL` defaults to `sqlite:///data/frontier_lens.db`. `SPEND_CAP_USD` defaults to 5.

No keys are required for seed, tests, or fixture replay.

## Local inference

```bash
ollama pull gemma4:12b
ollama serve
```

Quant: Q4_K_M. Streamlit does not load weights. It calls the OpenAI-compatible `/v1` on the inference host.

## Deployed Streamlit

Streamlit Cloud cannot host GGUF. Point `base_url` at OpenRouter (or similar) via secrets, or at Ollama/llama.cpp/vLLM on a **GPU VM** you control. Never expose an unauthenticated inference URL. Reviewers should use fixture replay.

vLLM is Apache-2.0 (the code is free) but needs NVIDIA GPUs you pay for. Not used in this build.

The AST/shlex sandbox is **not** a production isolation boundary. Later: gVisor, nsjail, or Docker.

## CI

`uv sync --frozen`, ruff, pytest (including key re-execution), fail if `.env` is tracked.

## Spend

Contest and generation honor `SPEND_CAP_USD`. One live Evaluate at a time (SQLite WAL, single writer). Runs resume from the last unanswered item.
