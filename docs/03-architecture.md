# Architecture

CLI is the core. Streamlit is a viewer.

```
templates → sandbox (keys) → SQLite questions
contest.py → openai_compatible | fixture → attempts
stats.py → heatmap / CIs
generate.py / agent → sandbox → candidates → SME → questions
```

## Sandbox

No Docker. Untrusted generated code:

- Python: `ast` allowlist (`math`, `itertools`, `collections`, `decimal`); no `open`/`exec`/`eval`/`__import__`/dunders; `python -I`; temp dir; 2s timeout.
- Shell: `shlex`; allowlist `cat cut sort uniq wc head tail tr grep printf` (no `sed`/`awk` unless a template needs them); reject redirection, `;`, `&&`, backticks, `$(`, absolute paths; temp dir with only the item's files; `LC_ALL=C`; 2s timeout.

Run twice; mismatch → `non_deterministic`. This is **not** a production isolation boundary.

## Providers

`Provider.complete(prompt) -> str`. Adapters: `openai_compatible` (Ollama, OpenRouter, maker APIs — HTTP client only, no closed models) and `fixture`. Tools and code execution disabled. Timeouts, retries, latency and tokens recorded.

Local quant = Ollama GGUF Q4_K_M. Not bitsandbytes. Not vLLM on this laptop.

## Streamlit

Four pages: Contest, Weaknesses, Generate + Review, Results. Sidebar role + health dots. Evaluate launches `contest.py` as a subprocess; the page polls SQLite. Public viewers use fixture replay.
