# All twelve BART stage-2 ablations

ERRANT correction F0.5 (%), 5 beams. Checkpoints were selected on dev_tune; final scores below are from saved evaluation JSONs. Delta columns are percentage-point differences from the rerun baseline.

| Run | Learning rate | Weight decay | Dropout | Label smoothing | dev_tune | Δ tune | dev_test | Δ test | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1e-05 | 0.01 | 0.1 | 0.1 | 54.02 | +0.00 | 52.26 | +0.00 | 5 |
| lr-low | 5e-06 | 0.01 | 0.1 | 0.1 | 53.39 | -0.63 | 51.90 | -0.36 | 3 |
| lr-high | 2e-05 | 0.01 | 0.1 | 0.1 | 53.61 | -0.41 | 50.84 | -1.42 | 4 |
| wd-zero | 1e-05 | 0 | 0.1 | 0.1 | 53.99 | -0.03 | 51.86 | -0.40 | 5 |
| wd-mid | 1e-05 | 0.05 | 0.1 | 0.1 | 54.04 | +0.02 | 52.04 | -0.22 | 4 |
| wd-high | 1e-05 | 0.1 | 0.1 | 0.1 | 54.20 | +0.18 | 52.07 | -0.19 | 4 |
| dropout-zero | 1e-05 | 0.01 | 0 | 0.1 | 52.86 | -1.16 | 51.11 | -1.15 | 1 |
| dropout-low | 1e-05 | 0.01 | 0.05 | 0.1 | 53.51 | -0.51 | 51.10 | -1.16 | 4 |
| dropout-high | 1e-05 | 0.01 | 0.3 | 0.1 | 41.55 | -12.47 | 40.92 | -11.34 | 5 |
| smoothing-zero | 1e-05 | 0.01 | 0.1 | 0 | 53.16 | -0.86 | 50.96 | -1.30 | 4 |
| smoothing-low | 1e-05 | 0.01 | 0.1 | 0.05 | 53.87 | -0.15 | 52.01 | -0.25 | 5 |
| smoothing-high | 1e-05 | 0.01 | 0.1 | 0.2 | 53.94 | -0.08 | 51.74 | -0.52 | 4 |

## Findings

- Weight decay 0.1 gives the highest dev_tune score (54.20), just 0.18 points above baseline. This is a small, single-seed difference.
- The baseline has the highest dev_test score (52.26); no tested variant improves it. Do not select hyperparameters by repeatedly inspecting this split.
- Removing or reducing dropout lowers both scores; dropout 0.3 is much worse (41.55 tune / 40.92 test). This experiment changes residual, attention, and activation dropout together.
- Neither tested learning-rate change improves the baseline on dev_tune or dev_test.
- Label smoothing 0.1 remains strongest among the tested smoothing values on both splits.
- The 0.05 follow-ups were chosen after inspecting the original results. Their held-out comparisons are exploratory; small differences need additional seeds.

## Provenance and interpretation

All 12 result folders were retrieved on 6 November 2026. Each was checked against its committed config: 33,943 training pairs, identical recorded torch/transformers versions, correct effective dropout fields, 2,000 dev_tune predictions, 2,384 dev_test predictions, and 5-beam final decoding. All have a training-finished timestamp and saved best-checkpoint metadata.

Machine-readable P/R/F0.5, M2 scores, training durations, and selected epochs are in [summary-all-12.json](summary-all-12.json). Raw inputs to this summary are each run's `dev_tune.scores.json`, `dev_test.scores.json`, `run_info.json`, and `best_checkpoint.json`.

Use [the experiment log](README.md) for the original submission and fixed procedure, and [the follow-up log](followup-005.md) for the three additional runs. The configs and hashes are controlled, but this is one seed per configuration, not a statistical significance study. Final evaluation scores can differ slightly from training-time callback scores; this table consistently uses the final saved evaluations.
