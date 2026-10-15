# Architecture

CLI is the core. Streamlit is a viewer.

```
templates → sandbox (keys) → split.py (template train/eval) → SQLite questions
contest.py (policy k=8, reference k=3) → openai_compatible | fixture → attempts
stats.py → pass_rate / pass@k / learnable band / heatmap
generate.py / agent (train: estimate_pass_rate; eval: no policy)
  → sandbox → candidates → audit sample → questions
reward.py exact_match_v1 → scoring
export.py → rl_tasks.jsonl / eval_tasks.jsonl / sft_rejection_sampled.jsonl
```

## Sandbox

No Docker. Untrusted generated code:

- Python: `ast` allowlist (`math`, `itertools`, `collections`, `decimal`); no `open`/`exec`/`eval`/`__import__`/dunders; `python -I`; temp dir; 2s timeout.
- Shell: `shlex`; allowlist `cat cut sort uniq wc head tail tr grep printf` (no `sed`/`awk` unless a template needs them); reject redirection, `;`, `&&`, backticks, `$(`, absolute paths; temp dir with only the item's files; `LC_ALL=C`; 2s timeout.

Run twice; mismatch → `non_deterministic`. This is **not** a production isolation boundary.

## Providers

`Provider.complete(prompt) -> str`. Adapters: `openai_compatible` (Ollama, OpenRouter, maker APIs — HTTP client only, no closed models) and `fixture`. Tools and code execution disabled. Timeouts, retries, latency and tokens recorded.

Local quant = Ollama GGUF Q4_K_M. Not bitsandbytes. Not vLLM on this laptop.

## Reward

`reward.grade` is a pure function. Id `exact_match_v1`: last `FINAL:` line only; empty / hedges (`or`, `/`, `either`) / multi-candidate answers → 0; else normalized exact match.

## Streamlit

Four pages: Contest, Weaknesses (pass-rate histogram + heatmap), Generate + Review (audit + suspicious queue), Results (band share, headroom, audit CI, one-shot vs agent). Sidebar role + health dots. Evaluate launches `contest.py` as a subprocess. Public viewers use fixture replay.
