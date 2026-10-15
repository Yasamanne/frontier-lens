# Data model

SQLite file: `data/frontier_lens.db` (gitignored). Created on first seed.

## questions

| Column | Notes |
|---|---|
| id | SHA-256 of template + params + code + inputs (stable across reseeds of the same spec) |
| topic | `shell_pipelines`, `python_state`, `multi_step_units`, `constraint_counting` |
| level | 1, 2, 3 |
| band | `harness` (L1), `standard` (L2), `hard` (L3) |
| split | `harness`, `selection`, `baseline`, `generated` |
| template, params_json | provenance |
| stem, code, lang, input_files_json, key | key from sandbox, run twice |
| source | `seed` or `generated` |
| generator_model, generator_mode, source_fail_ids | generated only |

L2/L3: 12 items per cell, split 6 selection / 6 baseline with RNG seed 42. Only selection chooses the weakest cell. New items compare to baseline.

## models

Registry from `models.yaml`: id, role (`evaluated`/`generator`), family, provider, base_url, model, quant, thinking, license, api_key_env. Loader rejects missing license and overlapping families between roles.

## eval_runs

id, created_at, item_set_hash, status (`running`/`complete`/`failed`), k, notes. Resume from last unanswered item. One live writer at a time.

## attempts

run_id, model_id, question_id, sample_index (1..k), raw_output, parsed_answer, is_correct, unparsable, latency_ms, tokens_in, tokens_out.

## candidates / reviews / trajectories

Candidates: executed key, status (`pending`/`auto_rejected`/`approved`/`rejected`), auto_reject_reason, generator_mode. Reviews: rubric + label, reviewer, decided_at. Trajectories: agent steps. Rejected and auto-rejected rows never copy into `questions`.
