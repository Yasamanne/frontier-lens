# Ops

## Env

Copy `.env.example` to `.env` (gitignored). `OPENROUTER_API_KEY` is required for live contest and generation. `OLLAMA_BASE_URL` is unused unless you add a local row in `models.yaml`. `DATABASE_URL` defaults to `sqlite:///data/frontier_lens.db`. `SPEND_CAP_USD` defaults to 5.

No keys are required for seed, tests, or fixture replay.

## Hosted inference (OpenRouter)

The default roster is OpenRouter open-weight slugs. Streamlit does not load weights; it calls `https://openrouter.ai/api/v1`. Local Ollama remains optional if you add a `localhost` row in `models.yaml`.

## Deployed Streamlit

Streamlit Cloud cannot host GGUF. Point `base_url` at OpenRouter (or similar) via secrets, or at Ollama/llama.cpp/vLLM on a **GPU VM** you control. Never expose an unauthenticated inference URL. Reviewers should use fixture replay.

vLLM is Apache-2.0 (the code is free) but needs NVIDIA GPUs you pay for. Not used in this build.

The AST/shlex sandbox is **not** a production isolation boundary. Later: gVisor, nsjail, or Docker.

## CI

`uv sync --frozen`, ruff, pytest (including key re-execution), fail if `.env` is tracked.

## Spend

Contest and generation honor `SPEND_CAP_USD`. One live Evaluate at a time (SQLite WAL, single writer). Runs resume from the last unanswered item.
