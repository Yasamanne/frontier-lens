# Data model

SQLite file: `data/frontier_lens.db` (gitignored). Created on first seed.

## questions

| Column | Notes |
|---|---|
| id | SHA-256 of template + params + code + inputs (stable across reseeds of the same spec) |
| topic | `shell_pipelines`, `python_state`, `multi_step_units`, `constraint_counting` |
| level | 1, 2, 3 |
| band | `harness` (L1), `standard` (L2), `hard` (L3) |
| split | `harness`, `train`, `eval` — **assigned at the template** (L1 is harness). Eval templates never appear in train. |
| template, params_json | provenance |
| stem, code, lang, input_files_json, key | key from sandbox, run twice |
| source | `seed` or `generated` |
| generator_model, generator_mode, source_fail_ids | generated only; empty on eval generation |
| reward_fn | `exact_match_v2` |
| audit_batch_id | generated items accepted via an audit batch |

## models

Registry from `models.yaml`: id, role (`policy` / `reference` / `generator`), family, provider, base_url, model, quant, thinking, license, api_key_env. Loader rejects missing license, missing policy, and generator families that overlap policy or reference.

## eval_runs / attempts

Runs store k (policy default 8, reference default 3). Attempts: sample_index 1..k, raw_output, parsed_answer, is_correct from `exact_match_v2` on the last `FINAL:` line.

Run status: `queued` → `running` → `complete`, or `failed` / `interrupted`. `config_json` holds what is needed to resume (models, k, question ids, judge, verifier). A running job refreshes `heartbeat_at` and `pid` every few seconds; a `queued`/`running` run with no heartbeat for 90 s is marked `interrupted`. `phase` (`primary` / `verifier` / `flagging` / `done`) and `planned_primary` / `planned_verifier` drive the progress bar, and `error` keeps the last failure. Resuming (`python -m frontier_lens.eval_job RUN_ID`) samples only the (model, question, sample) triples the run does not have yet.

## op_jobs

Background jobs other than evaluation runs: `kind` is `rescue`, `augment`, `calibrate` or `seed`. Same status, heartbeat and resume rules as `eval_runs`. `planned` / `done` count units of work (questions, candidates, topics). Each finished unit is appended to `op_job_items` (`job_id`, `skipped`, `item_json`) as it completes, so the Generate and Seed pages show results after a refresh and resuming skips finished work. Calibration writes `data/level_knobs.json` once, after every selected topic is measured. Run or resume from the CLI with `python -m frontier_lens.jobs JOB_ID`; logs are `logs/op_JOB_ID.stdout.log` (formatted) and `logs/op_JOB_ID.{txt,jsonl}` (events).

## candidates / reviews / audit_batches / suspicious_samples / trajectories

Candidates: executed key, `policy_pass_rate`, `pass_at_k`, `rl_split`, status (`pending` / `auto_rejected` / `approved` / `rejected`). One-shot train items outside the learnable band are `auto_rejected` with reason `out_of_band`.

Reviews: `label`, `note`, `reasons` (JSON list of codes from `feedback.REASONS`, e.g. `ambiguous`, `too_easy`, `answer_leak`) and `original_stem` (the generated stem when the SME edited it before approving). `feedback.context_for(topic)` turns recent human reviews (`kind` `sme_queue` / `audit`) into a prompt section: same-seed rejections, most common reasons for the topic, recent rejections, SME rewrites and approved examples with comments. It is added to every train-side generator prompt and to the AI reviewer in Generate + Review mode; feedback on rescue hints goes only to the hint prompt. Held-out eval generation does not use it, because reviewers see policy pass rates.

Audit: a random sample (default 50) is labeled; Wilson 95% CI on the defect rate; accept the batch if the upper bound is below 10%.

Suspicious samples: policy traces that are correct but look like lucky guesses (conflicting `FINAL:` lines, hedge then correct).

## Exports

Written under `data/exports/` (gitignored data dir; copy numbers into `results/experiment.md`):

- `rl_tasks.jsonl` — accepted train items, `split: "train"`, includes `policy_pass_rate` / `pass_at_k`.
- `eval_tasks.jsonl` — same columns except **no policy-feedback rates**.
- `sft_rejection_sampled.jsonl` — `{prompt, completion, model, item_id}` from correct policy and reference samples.
