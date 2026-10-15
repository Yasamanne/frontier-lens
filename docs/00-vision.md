# Vision

Frontier Lens is a lens on where open-weight models fail, plus a human-gated way to grow an eval set aimed at those failures.

## Who it is for

Eval engineers and subject-matter reviewers who need diagnostic items, not another static leaderboard. Organizers run contests. SMEs approve or reject generated candidates. Viewers inspect heatmaps and results.

## The problem

Static eval sets go stale and do not say *where* a model fails. Generated items that skip review are often broken (wrong keys, duplicates, ambiguous stems). Closed eval APIs make reruns hard to pin.

## What we build

1. Models predict the output of **execution-verified** snippets (they must not run code).
2. Failures become a weakness map with **n** and **Wilson 95% CIs**.
3. A generator from a **family not under test** proposes new items aimed at the weakest cell (one-shot or a small agent that checks its own work).
4. Every candidate is **executed** for its key, then an SME approves, edits, or rejects it (`broken_item` vs `hard_item` vs `trivial`).
5. We measure whether approved items are harder, discriminating, or junk, and whether the agent produces less junk than one-shot.

## Why not seed from an existing benchmark

Numbers below should be re-checked before quoting in the README.

- **ARC-AGI-3** — interactive game environments, not questions.
- **Humanity's Last Exam** — copying breaks originality; contamination risk.
- **FrontierMath Tier 4** — private by design.
- **Terminal-Bench 2.0** — Docker sandboxes and agent loops; too few tasks in two days for filled heatmap cells.

Lessons: static sets saturate; Terminal-Bench's real strength is **ground truth from execution**, which we borrow in a **single-turn** format.

**Rule:** all items are original templates. No questions, assets, or code copied from other benchmarks or projects.

## Prior art

Name these in the README:

- **Dynabench** — humans write examples that fool models.
- Refreshing / live benchmarks.
- Adversarial data collection.
- **Terminal-Bench** — execution-graded tasks.

**What is different:** generation targets a *measured cell*; keys come from execution, not the generator's claim; a human gate decides fairness and novelty; every item carries provenance (seed vs generated, template, source failure ids, generator model, generator mode).

## Claims

- **Claim:** the system grows a diagnostic set under human control.
- **Do not claim:** that the generator improves models, or that the items measure general coding ability.

Open-weight only: no closed-source model evaluates, generates, or assists in the pipeline. Tradeoff: no closed frontier reference. Several open models still sit near the top of current open leaderboards.
