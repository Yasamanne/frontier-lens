# Experiment

Fill this after a real policy k=8 live run, generation, audit, and export. Do not invent numbers.

## Status

**RL-ready, not RL-proven.** No LoRA / GRPO training run was done.

## Template

- Policy model:
- k (policy / reference): 8 / 3
- Seed items scored:
- Items generated (one-shot / agent), train split:
- Items generated, eval split (no policy calls):
- Learnable-band share (one-shot / agent):
- Mean headroom pass@8 − pass@1:
- Audit: n=, defects=, Wilson 95% CI [, ], accepted/rejected at 10%:
- Held-out eval pass_rate / pass@k:
- JSONL rows: rl_tasks= eval_tasks= sft=
- Training run: none

Record `fixtures/recorded_run.json` from `--live --write-fixtures`.
