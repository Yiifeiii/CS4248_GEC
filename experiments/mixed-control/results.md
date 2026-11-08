# Completed single-stage controls

CPU preflight 1094809 and BART/T5 array 1094810 completed with exit code 0 on
8 November 2026. Both models completed 14,482 optimizer updates over the same
1,853,729-pair mixed dataset (nominal presentations). Exact mixed-data hashes
match between runs. These are real full-data runs, not smoke-test scores.

## F0.5 comparison

ERRANT correction F0.5 percentages, five-beam final evaluation. Best checkpoints
are selected on dev_tune. Curriculum references are the original BART/T5 3+5
recipes (`bart-base.s1s2`, `t5-base.s1s2`), not the longer T5 recipe.

| Model | Split | Single stage | Two-stage curriculum | Curriculum gain (points) |
|---|---|---:|---:|---:|
| BART | dev_tune (2,000) | 49.74 | 54.02 | +4.28 |
| BART | dev_test (2,384) | 48.07 | 52.26 | +4.19 |
| T5 | dev_tune (2,000) | 51.84 | 55.21 | +3.37 |
| T5 | dev_test (2,384) | 49.79 | 52.50 | +2.71 |

| Model | Single-stage dev_test P / R | Curriculum dev_test P / R | Selected control step | Selected epoch | Slurm elapsed |
|---|---|---|---:|---:|---|
| BART | 54.53 / 32.62 | 57.73 / 37.90 | 9,000 | 0.621 | 1h27m08s |
| T5 | 57.68 / 32.18 | 57.44 / 39.06 | 13,000 | 0.898 | 1h22m36s |

The curriculum improves BART precision and recall; T5's gain comes from recall
while precision is slightly lower. These results favor the complete two-stage
recipe over this single-stage recipe for both models. They do not isolate data
ordering: stage-specific LR/batching, optimizer/scheduler resets and intermediate
checkpoint selection also differ. One seed per model provides no significance test.

## Provenance and reporting

- Same original pretrained initialization, hashed stage1/stage2 training data,
  stage1/stage2 example weights 3:5, dropout/weight decay/smoothing and seed as
  documented in the [design](README.md).
- Software matches the historical reference: torch 2.11.0+cu128 and transformers
  4.46.3. No dev/test examples were added to training. Dev_tune selects checkpoints;
  dev_test is final evaluation.
- Dataset length counts nominal exposure, not exact processed samples. Trainer
  final-batch/accumulation rounding applies. The selected checkpoints precede the
  end of the full run; checkpoint selection is part of both recipes.
- Tables consistently use the saved final score JSONs. BART's training-time best
  callback scored 49.79 on dev_tune; the saved checkpoint's final evaluation is
  49.74. This small decoding difference is retained rather than mixing score routes.
- Slurm elapsed includes data setup and final evaluation, not only optimizer time.
- Original failed array 1094732 never trained; the dependency repair is recorded
  in the design. Completed source: `39b9029`.

[Machine-readable comparison](results.json) includes all edit counts and P/R/F0.5.
Per-run scores, configs, selected-checkpoint and mixed-data provenance are under
[`bart-base.mixed-control`](../training/bart-base.mixed-control/) and
[`t5-base.mixed-control`](../training/t5-base.mixed-control/). Raw predictions,
training histories and model weights remain on the cluster, outside this PR.
