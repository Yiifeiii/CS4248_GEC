# Curriculum baseline versus four controls

The same two-stage curriculum baseline is compared with four single-stage
shuffled controls for each model.

| Setup | Training exposure | Optimization settings | BART dev_test F0.5 | T5 dev_test F0.5 |
|---|---|---|---:|---:|
| **Curriculum baseline** | Stage-1 data ×3, then stage-2 data ×5 | Stage 1, then stage 2 | **52.26** | **52.50** |
| 1A | Matched baseline exposure | Stage 1 throughout | 49.01 | 50.74 |
| 1B | Matched baseline exposure | Stage 2 throughout | 46.49 | 50.31 |
| 2A | Every unique training pair ×3 | Stage 1 throughout | 46.10 | 46.13 |
| 2B | Every unique training pair ×5 | Stage 2 throughout | 44.65 | 46.46 |

All scores use best-dev_tune checkpoints and five-beam ERRANT correction F0.5.
All controls start from the same original pretrained model, use seed42 and
exclude dev/test from training. Each configuration has one seed.

[Full results and provenance](four-controls/results.md) ·
[Exact settings, data construction and commands](four-controls/README.md).
