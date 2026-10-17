# Evaluation

## Unit of analysis

The **item**, not the sample. k samples of the same item are correlated.

Policy: k=8. Per item `pass_rate = correct/k`. Unbiased `pass@k = 1 − C(n−c, k)/C(n, k)`. Headroom = `pass@8 − pass@1`. Learnable band: `0 < pass_rate < 1`. Report the share of items in the band, per template and per generator mode.

Reference models: k=3. Heatmap cells still use Wilson 95% CIs over items; do not color a cell weak if its CI overlaps the model's overall mean.

## Holdout

Each generator family holds out a disjoint seed range as its own template (`shell_pipelines_eval`, `python_state_eval`, `constraint_counting_eval`, `multi_step_units_eval`, origins 10000–10003). Train uses the same knobs on origins 0–7 under `{topic}_train`. The same template and seed range never appears in both splits. Eval items — seed and generated — are produced **without policy feedback**. Train generation may call `estimate_pass_rate`.

**L2/L3 knobs (production):** `uv run python -m frontier_lens.calibrate --live` measures the policy on the monotone grid (k=8) and writes `data/level_knobs.json` with policy id, k, timestamp, per-point rates, and chosen indices. L2 is the grid point in pass-rate `[0.2, 0.8]` closest to 0.5 that still has a harder lower point; L3 is that harder point (prefer rate `< 0.2`). Then `seed --force`. Each L2/L3 seed row stores `knob_source` / `knob_policy` / `knob_grid_index` in `params_json`. CI uses a synthetic probe curve unless `FRONTIER_LENS_REQUIRE_LIVE_KNOBS=1`.

L1 harness is neither train nor eval for RL export.

## L1 harness

If a model fails L1, fix prompt, parser, or keys before reading L2/L3.

## Generator comparison

Headline metric: **learnable-band hit rate** per mode (one-shot vs agent). One-shot is post-filtered by the same k=8 estimate. Also report junk by auto-reject reason.

## Human QA

Draw a random sample (default 50) from a generated batch. SME labels each, blind to generator model and mode. Defect rate gets a Wilson 95% CI; accept the batch if the upper bound is below 10%. A second queue holds correct-but-suspicious policy samples (right answer, inconsistent reasoning) to catch lucky guesses and reward hacks.

## Tests

Parser; last-FINAL reward (hedge / multi-answer → 0); sandbox bans; pass@k known values; learnable-band edges 0 and 1; template-split leakage; Wilson audit acceptance; promotion gate; export Pydantic schema (eval rows omit policy-feedback fields); registry license + family split; **re-execute every seed item and match stored keys**.

## Honest negatives

"The agent did not raise band hit rate" and "GRPO did not move eval" are valid findings. Do not claim RL-proven without a training run.
