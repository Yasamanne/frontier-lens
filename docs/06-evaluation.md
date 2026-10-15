# Evaluation

## Unit of analysis

The **item**, not the sample. k samples of the same item are correlated. Per-item accuracy = mean over k (or majority vote for the cell CI). Then Wilson 95% CI over items in the cell. Stability = share of items where all k samples agree.

## Holdout

Selection items (6 per L2/L3 cell) choose the weakest cell. Baseline items (6) are the comparison for new approved items. This avoids regression-to-the-mean from using the same items that looked worst by chance.

Do not color a cell weak if its CI overlaps the model's overall mean. Expect wide CIs at n=6 or 12.

## L1 harness

If a model fails L1, fix prompt, parser, or keys before reading L2/L3.

## Generator comparison

Same cell, same failures, same generator model, 10 candidates each, SME blind to mode. Metrics: junk rate by reason, SME approval rate, accuracy of approved items vs baseline, discrimination (weak minus strong error rates on the approved set).

## Tests

Parser; sandbox rejections for each banned construct; Wilson math; promotion gate; adapter contract (plain text, tools off); registry license + family split; **re-execute every seed item and match stored keys** locally and in CI.

## Honest negatives

"The agent did not reduce SME rejections" is a valid finding. Do not claim the generator improves models.
