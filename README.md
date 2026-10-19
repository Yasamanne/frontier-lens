# Frontier Lens

A small **RL data engine** for open-weight models. It measures a policy model on execution-verified tasks, finds where the policy has gap, generates or repairs tasks at the edge of what it can do, gates them with human review whose feedback flows back into the generator, and exports RL-ready data.

Models answer tasks in plain text and must not run code. The gold key is the stdout of the task's verifier code, run in a sandbox. Answers end with `FINAL: <answer>`. The reward is `exact_match_v2`: it reads the last `FINAL:` line only (case-insensitive, so `Final: 42` counts; v1 was case-sensitive) and gives 0 to hedges.

**Claim:** the pipeline produces **RL-ready** task files (prompt, executed gold key, pinned reward id) under human review.
**Not claimed:** that RL training on them improves the policy. That would make them **RL-proven**, and no training run was done.

## The problem

- **Signal.** RL post-training (e.g. GRPO) learns only from prompts the policy *sometimes* solves. If all sampled completions get the same reward, the group advantage is zero and there is no gradient. Useful tasks sit in the **learnable band** `0 < pass_rate < 1`.
- **Static benchmarks are a poor source.** They saturate, leak into pretraining, give one score instead of a map of weaknesses, and often have keys you cannot verify.
- **Rewards are easy to game.** A reward that accepts hedges ("42 or 43") or the first answer line teaches the policy to game it.
- **Human review is the expensive part.** Reviewing every item does not scale, and reviewer judgements are usually thrown away instead of improving the next batch.

Frontier Lens targets one policy: measure it, find its weak cells, make tasks that land in its band, keep the keys verifiable, and turn reviewer feedback into reusable guidance for the generator.

## Who it is for

Research engineers and data teams building RL or fine-tuning data for a specific open-weight model, who need diagnostic, execution-verified tasks at that model's learnable band and a held-out eval the generator was not fitted to.

## How it works

```
Seed (procedural templates → sandbox keys)  ──►  Evaluate (policy k=8, references k=3)
        ▲                                                │
        │                                                ▼
   Upload CSV/JSONL                        Gap Analysis (pass-rate histogram, topic×band
                                           heatmap with Wilson CIs, failure modes)
                                                         │
                                                         ▼
     Dataset / Headroom  ◄──  SME review  ◄──  Generate (dead-zone rescue with hints,
     export JSONL             (reasons +        augmentation: one-shot / ReAct agent + Review,
                               notes)           each candidate calibrated against the policy)
                                  │                      ▲
                                  └── feedback.context_for ┘  (fed into generator prompts)
```

| Page | What it does |
|---|---|
| **Seed** | Builds the procedural bank (7 topics × 26 = 182 items: L1 harness, L2 standard, L3 hard). Calibrates the L2/L3 difficulty knobs against the policy. Uploads CSV/JSONL question sets, with exact and near-duplicate detection. |
| **Evaluate** | Scores the policy and reference models k times per item. Optional LLM answer-equivalence judge, and a verifier model on items every model failed. Runs in the background with a live progress panel, Stop and Resume. Downloads CSV and logs. |
| **Gap Analysis** | Histogram of per-item policy pass rates; topic × band heatmap (a cell is "weak" only if its Wilson upper bound is below the model's mean); failure modes (unparsable, off-by-one, consistent-wrong, high-variance, hedge). |
| **Generate** | **Rescue** of pass@k=0 items with progressive hints (L1 framing → L2 technique → L3 common mistake), recalibrating after each. **Augmentation** of in-band items in three modes. The **SME queue** for approve/reject with reasons, notes and edits. |
| **Dataset** | RL-ready pool, SFT pool (unrescuable items with their failed responses), and an SME review tab. |
| **Headroom** | Baseline RL-suitable items vs approved new items: yield per generation mode, rescue rate, pass-rate distribution. |

Roles (Organizer / SME / Viewer) are a UI switch, not authentication.

## Setup

Pick one: **local with uv** (best for development) or **Docker** (nothing to install but Docker). Both were tested from a fresh clone.

You need an [OpenRouter](https://openrouter.ai/keys) API key for anything that calls a model (Evaluate, Generate, rescue, calibration). Seeding, the test suite and browsing existing data work without one.

### Option A: local (uv)

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs the pinned Python 3.11 for you) and git.

```bash
git clone https://github.com/Yasamanne/frontier-lens.git
cd frontier-lens

uv sync --group dev                  # create .venv and install dependencies
cp .env.example .env                 # then set OPENROUTER_API_KEY in .env
uv run python -m frontier_lens.seed  # build the 182-item seed bank in data/frontier_lens.db
uv run streamlit run app/Home.py     # open http://localhost:8501
```

Check the install with `uv run pytest` (about 4 minutes; model calls are faked, so it costs nothing and needs no key) and `uv run ruff check src tests`.

### Option B: Docker

Prerequisites: Docker.

```bash
git clone https://github.com/Yasamanne/frontier-lens.git
cd frontier-lens

cp .env.example .env                 # then set OPENROUTER_API_KEY in .env
mkdir -p data logs
docker build -t frontier-lens .
docker run -d --name frontier-lens -p 8501:8501 --env-file .env \
  -v "$PWD/data:/app/data" -v "$PWD/logs:/app/logs" frontier-lens
docker exec frontier-lens python -m frontier_lens.seed   # first run only: build the seed bank
```

Open http://localhost:8501. Stop and remove with `docker rm -f frontier-lens`; your database and logs stay in `./data` and `./logs`.

- Mount `data/` and `logs/` as shown, or the database is lost when the container is removed.
- Do not run the app or tests on the host against the same `data/` while a container uses it. SQLite in WAL mode cannot share its lock memory across the Docker Desktop VM boundary.
- Run CLI commands in the container with `docker exec frontier-lens python -m frontier_lens.<module>`.
- `railway.toml` deploys the same image to Railway; attach a volume for `data/`, or the database is lost on redeploy.

### First run

1. Open **Seed** and confirm the bank has 182 items (or upload your own CSV/JSONL).
2. Open **Evaluate**, pick the policy model and start a run.
3. Read the results in **Gap Analysis**, then use **Generate** to rescue or augment questions.
4. Review candidates in the Generate tab's SME queue (switch **Role** to Organizer or SME, top right), then export from **Dataset**.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `401` or "missing env OPENROUTER_API_KEY" | Set the key in `.env`. In Docker, pass `--env-file .env` and restart the container. |
| Pages are empty | Run the seed command (step 3 of Option A, last line of Option B). |
| `database is locked` or `disk I/O error` | Only one writer at a time. Stop other processes using `data/` (see the Docker note above). |
| Changes under `src/` do not show up | Restart Streamlit, or run it with `FRONTIER_LENS_DEV_RELOAD=1` for local development only. |
| Port 8501 is taken | Use `--server.port 8502` (local) or `-p 8502:8501` (Docker). |

**CLI.** Every stage works without the UI:

```bash
uv run python -m frontier_lens.calibrate --live --k 8      # measure policy pass rates on the knob grid
uv run python -m frontier_lens.seed --force                # rebuild the seed bank with the new knobs
uv run python -m frontier_lens.contest --live --models gemma4-26b-a4b
uv run python -m frontier_lens.eval_job RUN_ID             # resume an interrupted or failed run
uv run python -m frontier_lens.jobs JOB_ID                 # resume a rescue/augment/calibrate/seed job
uv run python -m frontier_lens.generate --run RUN_ID --model gemma4-26b-a4b --mode one_shot --split train
uv run python -m frontier_lens.export                      # JSONL into data/exports/
```

Each evaluation sample is logged to `logs/RUN_ID.{jsonl,txt}` (prompt, output, key, grade).

Settings live in `.env` / environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `POLICY_K` | 8 | Samples per item for the policy |
| `REFERENCE_K` | 3 | Samples per item for each reference model |
| `MAX_CONCURRENCY` | 16 | Model calls in flight at once |
| `AUDIT_SAMPLE_N` | 50 | Audit sample size |
| `AUDIT_MAX_DEFECT_HI` | 0.10 | Audit threshold on the defect-rate upper bound |
| `FRONTIER_LENS_REQUIRE_LIVE_KNOBS` | — | Refuse to seed from synthetic difficulty knobs |

`FRONTIER_LENS_DEV_RELOAD=1` is for local development only (see `docs/07-ops.md`).

## Models

All open-weight, called through [OpenRouter](https://openrouter.ai) with one OpenAI-compatible adapter (`models.yaml`).

| Role | Model |
|---|---|
| Policy | Gemma 4 26B-A4B (`google/gemma-4-26b-a4b-it`) |
| References | Gemma 4 31B, Gemma 2 27B, Qwen 3.8 27B, DeepSeek V4.1 Flash, Llama 4 Scout |
| Generator | Gemma 4 31B, thinking on |
| Judge | DeepSeek V4.1 Flash |

The generator shares a family with the policy, and the judge with one reference. This is allowed with a warning; see Limitations.

## Key design decisions

- **Keys come from executing code.** Every seed and generated item has verifier code. Its stdout, which must be identical on two runs, is the gold key. Uploaded sets are the exception: their keys are trusted as given.
- **The reward is hard to game.** Only the last `FINAL:` line counts. Hedges, multiple answers and empty answers score 0. The reward id is pinned in exports.
- **Selection by measured pass rate.**
  - The policy is sampled k=8 times per item.
  - Items in `0 < pass_rate < 1` are RL-suitable.
  - pass@k uses the unbiased estimator.
  - Heatmap and audit intervals are Wilson 95% intervals, chosen because samples are small.
- **Difficulty is procedural, not trick questions.**
  - 7 seeded generators: Python state with aliasing, shell pipelines, multi-step unit chains, constraint counting, FSM simulation, minimax, logic grids.
  - Knobs such as step count and state size control difficulty.
  - Large answer spaces (checksums, hashes) make guessing useless.
- **The held-out eval is a disjoint seed range.**
  - Each family's eval items come from its own `{topic}_eval` template: seeds 10000–10003, vs 0–7 for train.
  - Eval generation never sees policy results.
  - The eval export drops policy pass rates.
- **Human review is sampled and reused.**
  - An audit samples the pending pool and accepts it only if the Wilson upper bound of the defect rate is below 10%.
  - Every SME review records reasons (ambiguous, wrong key, too easy, answer leak, …), a note and any edit.
  - `feedback.context_for` turns recent reviews into a prompt section for every train-side generator and for the AI reviewer, which may not pass a question that repeats an SME-flagged problem.
  - Eval generation does not get this feedback, because reviewers see policy pass rates.
- **Long work runs in the background.**
  - Evaluations, calibration, seeding, rescue and augmentation run as detached processes.
  - They report through SQLite and send a heartbeat every 5 s; a job silent for 90 s is marked interrupted.
  - A job is owned by exactly one process.
  - Resume does only the missing work.

## Changes of direction

- **Local Ollama (Q4 GGUF) → hosted OpenRouter.** Consistent results across machines and no local GPU. Gemma 4 e4b is not hosted, so the policy became Gemma 4 26B-A4B.
- **GLM generator → DeepSeek V4.1 Flash → Gemma 4 31B (thinking on).** The rule that the generator family must be disjoint from the evaluated models was relaxed to a warning.
- **Review every item → sampled audit.** The Wilson upper bound decides whether a batch passes.
- **Item-level holdout → template-level holdout.** Item-level holdout leaked templates into eval.
- **Hand-written Python "gotchas" → procedural generators.** Difficulty now comes from step length and state, not tricks (`docs/known-gotchas.md`). The topics grew from 4 to 7.
- **Contest/Results pages → today's six pages.** The current pages are Seed, Evaluate, Gap Analysis, Generate, Dataset and Headroom.
- **In-page runs → resumable background jobs.** A page refresh used to kill an evaluation.
- **Stems of rejected variants → structured SME feedback.** Reasons, notes and rewrites now feed the generator and the AI reviewer.

## Results so far

From the local database (Oct 15–17, 36 evaluation runs with samples, 700 graded samples). These are small, exploratory numbers, not a benchmark.

| Policy (Gemma 4 26B-A4B), samples pooled across runs | Items | Samples | Accuracy | In band | Never solved | Always solved |
|---|---|---|---|---|---|---|
| Seed harness (L1) | 10 | 146 | 99% | 1 | 0 | 9 |
| Seed train (L2/L3) | 6 | 58 | 50% | 2 | 2 | 2 |
| Seed eval (held out) | 3 | 19 | 58% | 2 | 0 | 1 |
| Uploaded math/MCQ sets | 39 | 261 | 50% | 15 | 16 | 8 |

- The harness passing at 99% means the prompt/`FINAL:` format and grading work end to end.
- On uploaded sets, 38% of evaluated items were in the learnable band and 41% were never solved. The never-solved items are the targets for rescue.
- Generation: three augmentation candidates were approved by SME review, and two rescue attempts ended with no hint level reaching the band.
- **No training run was done. Coverage of the seed bank is small.** The next step is a full seed evaluation, then a GRPO run (see Next steps).

## Testing and validation

- **90 pytest tests** (`uv run pytest`) cover:
  - the reward (last FINAL only, hedges) and parsing;
  - sandbox bans;
  - pass@k, Wilson and learnable-band edges;
  - split leakage, and that **every topic has held-out eval items**;
  - re-executing every seed key;
  - procedural determinism;
  - provider retry and request shape;
  - export schemas (eval rows reject policy fields; SFT excludes eval and harness items);
  - the audit and promotion gates;
  - failure analysis;
  - background jobs (reap, claim, resume, ownership);
  - the SME feedback loop.

  Model calls are faked, so tests cost nothing.
- **CI** (GitHub Actions): `uv sync --frozen`, ruff, pytest, and a failure if `.env` is tracked.
- **Manual and end-to-end checks:**
  - Background jobs were exercised with real detached processes (kill -9, reaping, takeover, Stop).
  - Every page was rendered for each role with Streamlit's `AppTest`.
  - A concurrency race in dev hot-reload was reproduced (6 of 44 concurrent page loads failed) and confirmed fixed (0 of 48).

## Limitations

- **Evidence is proxy only.** Pass rates, band share and review results, not a training delta.
- **Small samples.** k=8 per item, about 8 items per heatmap cell, one probe item per calibration point: intervals are wide. With k=8, an item whose true pass rate is 0.05 still lands in the band about a third of the time.
- **Eval is near-distribution.** It comes from the same generators on held-out seeds, not unseen task types.
- **Judge bias.** The generator, the AI reviewer and the optional answer judge are the same model as one reference.
- **Reward edge cases.**
  - Answers containing `,` or `/` (e.g. `1,234`, `3/4`) score 0.
  - The optional LLM judge makes the reward non-deterministic.
  - Calibration samples the policy with thinking on, evaluation with thinking off.
- **The sandbox is a validator, not isolation.** It uses an AST and shlex allowlist, a 2 s timeout, and no CPU, memory or network limits.
- **Human data.** One annotator (the builder), no agreement statistics. The audit flow is in code and tests but not yet in the UI.
- **Ops.** No authentication, and `SPEND_CAP_USD` is read but not enforced: a public deployment can spend the API key. SQLite allows a single writer. Model calls have no hard total deadline (provider keep-alives can stall a call). Some older tests use the real `data/` database.
- **Known bug.** One rescue candidate stored a list-shaped generator response as its stem; generator output is not validated for type.

## Next steps

1. Evaluate the full seed bank (all 7 topics, train and eval), then run a small **GRPO/LoRA training run with TRL** on `rl_tasks.jsonl`. Measure on `eval_tasks.jsonl` against a random-selection baseline, and report a null result if that is what comes out.
2. Fix the reward edge cases (commas, fractions) and match the thinking mode between calibration and evaluation.
3. Use adaptive k: keep sampling until the interval clears 0 and 1. Use larger calibration probes.
4. Add a second annotator and Cohen's κ. Wire the audit into the UI. Measure whether SME feedback lowers rejection rates over time.
5. Use a judge from a different model family, and measure judge–SME agreement.
6. Add container isolation for the sandbox, Postgres, authentication and a job queue for multi-user use.

## Tools used, including AI

- **Libraries:** Streamlit, SQLAlchemy, the OpenAI Python SDK (pointed at OpenRouter), Pydantic, python-dotenv, PyYAML; uv, pytest and ruff for development; Docker and Railway for deployment.
- **AI coding assistants:**
  - **Cursor (agent mode)** was used for the initial scaffold and much of the Oct 14–16 feature work; those commits carry `Co-authored-by: Cursor`.
  - **Claude Code** was used on Oct 17 for:
    - the resumable background-job system;
    - diagnosing the Streamlit `sys.modules` race and the SQLite/Docker read errors;
    - the SME feedback loop;
    - independent code reviews;
    - this README refresh.
  - I chose the problem and the design direction, reviewed and tested the changes, ran the system, and decided the tradeoffs. AI-written code was checked with tests, end-to-end runs and separate review passes, several of which found bugs that were then fixed. ⟨fill in: your own account of what you did and how you directed the tools⟩
- **Models inside the pipeline** are open-weight only (see Models).

## Prior work and attribution

- Related work: [Dynabench](https://dynabench.org/) (adversarial human data), refreshing/live benchmarks, [Terminal-Bench](https://www.tbench.ai/) (execution-graded tasks), [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) (the trainer the exports target).
  - **What is different here:** generation targets one policy's learnable band; keys are executed; review is a statistical audit; and reviewer feedback is fed back into generation.
- Seed tasks are original procedural templates in `src/frontier_lens/templates/`. No benchmark items are included in the repository.
- The uploaded sets used in local experiments (MMLU, DeepMath, UltraData-Math CSVs) are **not redistributed here**. Anyone uploading third-party data should follow its licence.
- Model licences are listed in `models.yaml`. The project is MIT-licensed (`LICENSE`).

## Time spent

Approximately 25 hours over Oct 14–17, 2026.

## More documentation

- `docs/00-vision.md` — vision and claims.
- `docs/02-data-model.md` — tables, job states, SME feedback.
- `docs/06-evaluation.md` — evaluation plan.
- `docs/07-ops.md` — environment and operations.
