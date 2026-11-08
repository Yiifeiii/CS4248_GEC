# Two curriculum-training comparisons

These experiments answer different questions. Both use BART-base and T5-base,
with no GRECO, GECToR or longer-trained T5 checkpoint involved.

| Comparison | What changes? | dev_test curriculum gain | Full results |
|---|---|---|---|
| Original recipe vs mixed single-stage recipe | Data order, LR schedule, batching, optimizer resets and checkpoint handoff | BART +4.19; T5 +2.71 | [Recipe comparison](mixed-control/results.md) |
| Matched curriculum vs shuffled stream | Only training-example order within each model pair | BART +2.69; T5 +2.35 | [Ordering comparison](matched-order/results.md) |

The first comparison uses the selected best-dev_tune checkpoints from the two
recipes and nominally matches example exposure. It favors Dasha's complete recipe
but cannot attribute the gain to ordering alone.

The second comparison reruns both sides with matching initialization, exact
per-example multiplicities, batch size, one continuous optimizer/LR schedule,
evaluation points and total updates. Its primary endpoint is final weights after
all 1,853,729 examples, including the last incomplete batch. It favors curriculum
ordering under the matched recipe for both model families.

The eight rounds in the second comparison are matched-size blocks of a fixed
exposure stream, not eight complete passes through the combined dataset. Trainer
executes one physical pass so examples cannot be resampled or silently repeated.

Both comparisons use the same dev_tune/dev_test splits and five-beam ERRANT
correction F0.5. Each configuration has one seed; the reported gains are observed
differences, not statistical significance claims. Do not mix endpoints or subtract
across the two experiment designs to estimate the contribution of another setting.
