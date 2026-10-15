# Experiment

Recorded 2026-10-15 from this machine. Do not invent live model numbers.

## Status

**RL-ready, not RL-proven.** No LoRA / GRPO training run was done. No live policy samples were recorded in this pass (Ollama was not installed/running, so k=8 fixtures do not exist).

## Seed export (sandbox keys, no policy feedback)

| Field | Value |
|---|---|
| Policy model (configured) | `gemma4-e4b` (`gemma4:e4b` via Ollama) |
| k (policy / reference) | 8 / 3 |
| Seed items | 104 (8 harness, 72 train, 24 eval) |
| Items generated (one-shot / agent) | 0 / 0 — generator not run (no live generator call) |
| Learnable-band share per mode | n/a — no policy k=8 samples |
| Mean headroom pass@8 − pass@1 | n/a |
| Audit defect rate (Wilson 95% CI) | n/a — no generated batch to sample |
| Held-out eval | 24 seed items from templates `state_l3` and `count_l2`; **not scored** by the policy |
| JSONL | `results/rl_tasks.jsonl` 72 rows; `results/eval_tasks.jsonl` 24 rows; `results/sft_rejection_sampled.jsonl` 0 rows |
| Training run | **None.** |

Eval JSONL omits `policy_pass_rate` and `pass_at_k`. Train rows currently have those fields as `null` until a live contest writes attempts.

To fill the live cells:

```bash
ollama serve
ollama pull gemma4:e4b
uv run python -m frontier_lens.contest --live --models gemma4-e4b --write-fixtures
uv run python -m frontier_lens.export --all-train
```
