# Evaluate tab: a learning guide

Page: `app/pages/2_Evaluate.py`. Engine: `src/frontier_lens/contest.py` (sampling and grading), `eval_job.py` (background job), `runstate.py` (job state), `reward.py` and `parse.py` (grading).

## 1. What the tab is for

It answers one question: **how well does each model solve each question in the database?**

For every (model, question) pair it asks the model `k` times, grades each answer against the gold key, and stores every sample. Everything downstream depends on these samples:

- **Weaknesses** turns per-question pass rates into a histogram and heatmap.
- **Generate** picks rescue targets (pass@k = 0) and augmentation seeds (in-band items).
- **Dataset / export** builds RL and SFT files from correct and incorrect samples.

The tab also flags questions that *every* model failed, because those are either too hard or broken (wrong key, ambiguous stem). A human (SME) decides which.

## 2. Inputs

### Questions (from SQLite `questions`)

| Source | How it got there | Notes |
|---|---|---|
| `seed` | Seed page (procedural templates) | Key comes from executing verifier code |
| `generated` | Generate page, approved candidates | Key comes from executing verifier code |
| `uploaded` | Seed page, CSV/JSONL upload | Key is trusted as given; each row has its own split |

### Controls on the page

| Control | Meaning | Default |
|---|---|---|
| **Primary models** | Policy and/or reference models to score. The page only lists Gemma-family models with role `policy` or `reference`. | first one |
| **Verifier model** (optional) | A second model run *only* on questions where the primary got 0/k. Cannot be the generator or an already selected primary. | none |
| **LLM judge** checkbox | If exact match fails, asks the judge model whether the answers mean the same thing. Needs a `role: judge` entry in `models.yaml`. | on |
| **Override k** | Samples per question. Without it: policy k=8, reference k=3 (`POLICY_K`, `REFERENCE_K`). | off |
| **Limit number of items** | Score only the first N questions (smoke test). | off |
| **Source filter** | All questions, or one uploaded file. | all |
| **Train split only** | Excludes `eval` items. Ignored for uploaded files. | on |
| **Role** (top right) | Organizer can run and stop; SME can review; Viewer is read-only. A UI switch, not authentication. | Viewer |

The run button needs: a model selected, role = Organizer, `OPENROUTER_API_KEY` set, and no run already active.

### Cost preview

`questions x k x models = API calls` (plus up to `questions x k` verifier calls). The minute estimate assumes 15 s per call and is rough; with concurrency it is usually faster.

## 3. The pipeline, end to end

```
Click Evaluate
   │
   ▼
Resolve question ids  ──►  overlap check against completed runs
   │                          │
   │              no overlap  │  overlap found
   │                          ▼
   │                  choose: "Score only what's missing"  or  "Re-evaluate everything"
   ▼
eval_job.start(run_config)         # row in eval_runs, status=queued
   │
   ▼  detached process: python -m frontier_lens.eval_job RUN_ID
claim run (status=running, pid, heartbeat)
   │
   ▼
Phase "primary"      run_contest: for each model × question × sample
   │                    prompt  = contest_prompt(stem)
   │                    model   = provider.complete(...)    (+ one correction turn if no FINAL:)
   │                    grade   = reward.grade(...)         (+ optional judge)
   │                    INSERT into attempts, write logs/RUN_ID.jsonl
   ▼
Phase "verifier"     (only if chosen) re-run the verifier k times on questions where every primary sample was wrong
   ▼
Phase "flagging"     questions with 0 correct across primary + verifier → questions.sme_flagged = 1
   ▼
mark_complete        status=complete
```

### Stage details

1. **Resolve ids** (`_resolve_question_ids`). Uploaded-only: all rows of that file. Otherwise: seed, generated and uploaded questions, optionally train split only, ordered by `level, id`, then cut to the limit.
2. **Overlap check** (`_already_evaluated`). Counts samples per (question, model) in runs with status `complete`. A question is "done" for a model if it has at least `k` samples. A previous k=8 satisfies a new k=4 request, but k=4 does not satisfy k=8.
3. **Run config** (`contest.run_config`) is stored as JSON in `eval_runs.config_json`. This is what makes resume possible.
4. **Prompt** (`prompt.py`). A fixed prefix: reason step by step, no tools, end with exactly one `FINAL: <answer>` line, then the stem.
5. **Provider call** (`providers/openai_compatible.py`). Retries transient errors. If the text has no `FINAL:`, sends **one correction turn** asking only for the `FINAL:` line and appends it to the raw output.
6. **Grade** (`reward.grade`, id `exact_match_v1`). See section 5.
7. **Persist** (main thread only). One row per sample in `attempts`, plus an event in `logs/RUN_ID.jsonl`. Insert is `ON CONFLICT DO NOTHING`, so re-inserting is harmless.
8. **Concurrency** (`parallel.map_unordered`). Samples run in parallel up to `MAX_CONCURRENCY` (default 16); results arrive out of order and are written by one thread.

## 4. Data stores and files

| Where | What | Written by |
|---|---|---|
| `eval_runs` | One row per run: status, k, notes (batch label), `config_json`, `pid`, `heartbeat_at`, `phase`, `planned_primary`, `planned_verifier`, `error` | `runstate`, `contest.mark_*` |
| `attempts` | One row per sample: `raw_output`, `parsed_answer`, `is_correct`, `unparsable`, `latency_ms`, tokens. Unique on `(run_id, model_id, question_id, sample_index)`. Verifier samples live here too, under the verifier's `model_id`. | `run_contest`, `run_verifier` |
| `questions.sme_flagged` | 0 or NULL = fine, 1 = awaiting SME, 2 = marked broken | `eval_job.flag_for_sme`, SME buttons |
| `questions.sme_source` | `evaluate` or generate-side origin of the flag | `flag_for_sme` |
| `logs/RUN_ID.jsonl` and `.txt` | Per-sample log: prompt, output, gold key, reward | `runlog.write_event` |
| `logs/RUN_ID.stdout.log` | Process stdout, tailed in the live panel | detached job |

## 5. How a sample is graded

`reward.grade(raw, key, numeric, judge_fn)` returns `(reward, parsed_answer, unparsable)`.

1. Take the **last** `FINAL:` line. None found → reward 0, **unparsable**.
2. **Degenerate answer** → reward 0 (not unparsable). That means: empty, contains `or` / `either` / `/`, contains `,` or ` and `.
3. Normalize both answer and key: collapse whitespace, unwrap `\boxed{}`, strip `$`; if numeric, compare as numbers (`42.0` == `42`).
4. Equal → reward 1.
5. Not equal and a judge is configured → ask the judge "YES/NO"; `YES` → reward 1. Judge errors count as NO.
6. Otherwise reward 0.

Things worth knowing:

- Only the last `FINAL:` counts. Writing it twice with different answers is graded on the second.
- `parse_final` has extra fallbacks ("The answer is 42") but they are **display only**. The score never uses them, so a missing `FINAL:` stays unparsable.
- Because step 2 runs before normalization, a correct answer like `1,234` or `3/4` scores 0 (known limitation in the README).
- The judge makes the reward non-deterministic and adds roughly one call per wrong sample. The verifier is graded the same way as the primary, with the same judge.
- Calibration samples with thinking on; this page follows each model's `thinking` setting in `models.yaml`.

## 6. Outcomes

### Per sample

| Outcome | `is_correct` | `unparsable` | Typical cause |
|---|---|---|---|
| Correct | 1 | 0 | Exact match, or judge said YES |
| Wrong | 0 | 0 | Different answer |
| Hedge / degenerate | 0 | 0 | "42 or 43", comma lists, empty |
| Unparsable | 0 | 1 | No `FINAL:` even after the correction turn, or empty output |
| Provider error | 0 | 1 | Text is `[provider_error] ...`; counts as unparsable |

The live panel shows Correct / Wrong / Unparsable / Samples. Wrong = done - correct - unparsable.

### Per question (pass rate = correct / k, per model)

| Pass rate | Meaning | Where it goes next |
|---|---|---|
| 0 for all models | Too hard or broken | SME queue (`sme_flagged=1`); rescue candidate |
| 0 < p < 1 for the policy | **Learnable band**, useful for RL | Augmentation seeds, RL export |
| 1 | Too easy for this model | Mostly ignored |

### Per run (`eval_runs.status`)

| Status | Meaning | What you can do |
|---|---|---|
| 🟡 `running` / ⏳ `queued` | In progress with a fresh heartbeat | Watch, or Stop (Organizer) |
| 🟢 `complete` | All phases finished | Download, review SME queue, go to Weaknesses |
| 🟠 `interrupted` | Stopped by user, Ctrl+C, container restart, or heartbeat silent for 90 s | **Resume** (only the missing samples run) |
| 🔴 `failed` | Exception, e.g. unknown model, no questions, or 3 provider failures in a row | Fix the cause, then Resume |

Summary banner after a completed run:
- `N/M correct (x%)` and unparsable count for the primary models.
- Verifier line: how many verifier samples were correct on primary-all-wrong questions, or "no verifier needed".
- A warning with the number of questions flagged for SME review, or a success message if none.

### SME review outcomes (bottom of the page)

| Action | DB effect | Consequence |
|---|---|---|
| ✅ Mark valid | `sme_flagged = 0`, `sme_validated_key = key` | Key stands; the question stays in the pool (it is simply hard). Later runs will not re-flag it while the key is unchanged |
| 🗑 Mark broken | `sme_flagged = 2` | Excluded from future Evaluate runs (the run's question selection skips it), from re-flagging, from SFT export and from the Dataset/Weaknesses analyses |
| 💾 Save key | `key = new`, `sme_flagged = 0`, stored attempts re-graded | Gold key corrected and the question's attempts are re-graded against it with exact match (the judge is not re-run). The success message shows how many attempts changed |


## 7. Data flow variants

1. **Fresh run, no overlap.** Straight to a background job.
2. **Overlap, "Score only what's missing".** Sets `reuse_prev_runs=True`. For each (model, question, sample) the job skips it if any *completed* run (or this run) already has that sample. Mixed new and old data therefore appears across runs, and downstream pages read pooled attempts.
3. **Overlap, "Re-evaluate everything".** `reuse_prev_runs=False`, all samples re-run into the new run id.
4. **With verifier.** Extra phase; verifier attempts share the `attempts` table, and the CSV puts them in separate columns.
5. **Resume.** Same run id, same config. Samples already stored under that run id are skipped, and `planned_primary` is recomputed as stored + remaining. Resume works from the page or with `uv run python -m frontier_lens.eval_job RUN_ID`.
6. **Replay mode** (`replay: true` in config, CLI only). Answers come from `fixtures/recorded_run.json`; no API key or cost.
7. **CLI-only runs.** `uv run python -m frontier_lens.contest --live --models ...` runs in the foreground, without the heartbeat-based Stop.

## 8. Job lifecycle (why refresh is safe)

- The page never samples models itself. It creates the run (`eval_job.create`), then spawns a **detached process** (`runstate.spawn_detached`, own session).
- The process writes a **heartbeat** every 5 s. `reap_stale` (called on every page load) marks any queued/running run silent for more than 90 s as `interrupted`.
- **Ownership:** the job's `pid` must match the row. If you press Stop, or another process resumes the run, the old process sees it no longer owns the run and exits.
- **Progress** is derived from `attempts` counts versus `planned_primary`, so it survives page reloads.
- Only one run is active at a time from the UI (SQLite allows one writer).

## 9. Downloads

| File | Contents |
|---|---|
| `.txt` log | Human-readable per-sample trace |
| `.jsonl` log | Machine-readable events: prompt, raw output, parsed answer, gold key, reward, latency, tokens |
| `results.csv` | One row per question. Columns: run metadata, stem (300 chars), topic, band, key, then per model `response_i`, `final_answer_i`, `correct_i`, `pass_rate`, then `needs_sme_review` (True if every model got 0). Keys and answers are written as `="..."` so spreadsheets do not turn `4/5` into a date; responses are truncated to ~400 chars. |

## 10. Failure modes and how to read them

| Symptom | Likely cause | Check |
|---|---|---|
| Many unparsable | Model ignores `FINAL:`; reasoning cut off; provider errors | Run's `stdout.log`, `raw_output` in the jsonl log |
| `provider failed 3 samples in a row; stopping` | Bad key, rate limit, model slug retired | `.env`, provider error text in the log, then Resume |
| Everything wrong on one topic | Possibly a wrong gold key | SME queue, compare `key` with the model's consistent answer |
| Policy near 100% | Question too easy, or the harness (L1) items | Expected for L1 |
| Run stuck as running after closing Docker | Heartbeat stops; reaped to interrupted within ~90 s | Reload page, then Resume |
| `database disk image is malformed` | Host and container sharing one SQLite file in WAL mode | Use one environment per `data/` folder |
| Run button disabled | Viewer role, no key, nothing selected, or another run active | Sidebar status and role selector |

## 11. Limitations to keep in mind

- k=8 samples gives wide uncertainty; a question with true pass rate 0.05 still lands "in band" often.
- Pass rates pool across runs for downstream pages, even when runs used different k or judge settings.
- Answers containing `,` or `/` score 0 even when correct.
- The judge and the verifier can be the same model family as a reference.
- A question the primary failed but the verifier solved is not flagged for SME review. It stays pass@k = 0 for the primary, so it goes to hint generation (coded questions). Verifier samples are never exported as SFT data.
- Key re-grading is exact match only; a sample the judge had credited under the old key loses that credit.
- Roles are not security; anyone can switch to Organizer.

## 12. Hands-on exercises

1. **Smoke test:** tick *Limit number of items* = 3, k = 2, one model. Watch the progress panel and `logs/RUN_ID.stdout.log`.
2. **Read the evidence:** open the `.jsonl` log and find one sample where `reward` is 0 and `unparsable` is false. Is the model wrong, or the key?
3. **Overlap:** repeat the same run. You should see the per-model breakdown table and the two choices.
4. **Resume:** start a larger run, press Stop, then pick it under *Download results* and press Resume. Count that `attempts` rows do not duplicate.
5. **Verifier:** pick a verifier and compare `needs_sme_review` with and without it.
6. **Judge:** toggle the judge and compare accuracy on a question set with MCQ-style answers.

Useful SQL (read-only, on a copy of the db):

```sql
-- samples per run and model
SELECT run_id, model_id, COUNT(*) n, SUM(is_correct) correct, SUM(unparsable) unparsable
FROM attempts GROUP BY run_id, model_id;

-- per-question pass rate for one model across all runs
SELECT question_id, AVG(is_correct) pass_rate, COUNT(*) k
FROM attempts WHERE model_id = 'gemma4-26b-a4b' GROUP BY question_id;

-- questions waiting for SME review
SELECT id, topic, band, key FROM questions WHERE sme_flagged = 1;
```
