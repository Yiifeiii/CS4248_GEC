# Completed curriculum controls

All eight tasks in array **1097382** completed successfully (exit code 0). The CPU preflight **1097381** passed.

Each control is compared with the same original two-stage curriculum baseline for its model family.

## Primary results

ERRANT correction F0.5 (%), five beams, best-dev_tune checkpoints. Dev_tune has 2,000 sentences; dev_test has 2,384.

| Setup | Exposure | Optimization settings | BART dev_tune | BART dev_test | T5 dev_tune | T5 dev_test |
|---|---|---|---:|---:|---:|---:|
| **Curriculum baseline** | Stage-1 data ×3, then stage-2 data ×5 | Stage 1, then stage 2 | **54.02** | **52.26** | **55.21** | **52.50** |
| 1A | Matched baseline exposure | Stage 1 throughout | 50.77 | 49.01 | 53.15 | 50.74 |
| 1B | Matched baseline exposure | Stage 2 throughout | 49.04 | 46.49 | 52.43 | 50.31 |
| 2A | Every unique pair ×3 | Stage 1 throughout | 46.88 | 46.10 | 49.43 | 46.13 |
| 2B | Every unique pair ×5 | Stage 2 throughout | 46.42 | 44.65 | 49.03 | 46.46 |

The curriculum baseline beats all four controls for both models. The strongest control is 1A: the baseline leads it by **3.25 points for BART** and **1.76 for T5** on dev_test.

## Exposure and validation

- Matched exposure: **1,853,729** presentations; 1A uses **14,483** updates and 1B **57,930**.
- Uniform dataset: **569,499 unique source/target pairs**, with exact duplicates and stage overlap counted once.
- Three uniform passes: **1,708,497** presentations, **13,348** updates (2A).
- Five uniform passes: **2,847,495** presentations, **88,985** updates (2B).
- All eight recorded configs match the executed runs. Final source/target hashes and pretrained initialization records agree across the corresponding controls.
- Complete example consumption was checked against recorded update/sample counts; accumulation keeps the final partial update without padding or dropping examples.
- One seed per config. These compare recipes and exposure policies, not data order alone. The historical baseline is unchanged; its exposure counts are nominal, subject to its original batching/early-stopping procedure.

Primary tables use saved **best-checkpoint** score JSONs. Final-checkpoint scores are retained separately in [results.json](results.json); do not mix the endpoints.

## Artifacts

Executed source commit: `1a4c872`. See [design/configs/commands](README.md). Curated scores, run/config metadata, exposure/initialization manifests and selected-checkpoint metadata are under `experiments/training/four-controls-20261108/<model>-<control>/`. Weights, full histories and raw predictions stay on the cluster.
