# Completed matched-order results

Preflight 1095333 and all four tasks in array 1095334 completed with exit code0
on 8 November 2026. Both model families favor curriculum ordering in this run.

ERRANT correction F0.5 percentages, five-beam evaluation of **final weights**:

| Model | Split | Shuffled order | Curriculum order | Curriculum gain (points) |
|---|---|---:|---:|---:|
| BART | dev_tune (2,000) | 49.29 | 52.24 | +2.95 |
| BART | dev_test (2,384) | 47.82 | 50.51 | +2.69 |
| T5 | dev_tune (2,000) | 52.22 | 54.66 | +2.44 |
| T5 | dev_test (2,384) | 50.27 | 52.62 | +2.35 |

| Model | Order | dev_test precision | dev_test recall | Slurm elapsed |
|---|---|---:|---:|---|
| BART | Curriculum | 56.12 | 36.06 | 1h33m45s |
| BART | Shuffled | 52.71 | 34.88 | 1h33m22s |
| T5 | Curriculum | 58.20 | 38.04 | 1h17m18s |
| T5 | Shuffled | 56.90 | 34.28 | 1h16m53s |

Both precision and recall improve for both model families. Each run consumed
1,853,729 training presentations in 28,965 updates. Configs match within each
family except stream order and artifact paths. Exposure manifests and original
checkpoint initialization hashes match across all runs. Sequential loading retains
the final partial batch; there is no early stopping, optimizer reset, global
reshuffling or length grouping. Eight logical rounds annotate the exposure stream.

This supports an ordering benefit under this shared optimization recipe. It is
one seed per model, not a significance test or a claim that ordering helps every
recipe. Unlike the earlier [complete-recipe comparison](../mixed-control/results.md),
this experiment does not change the optimization schedule between paired runs.
Do not compare these final-checkpoint scores directly with the old comparison's
independently selected best checkpoints to estimate an ordering effect.

The evaluated T5 family is google-t5/t5-base with newly trained weights. Dasha's
longer-trained T5 checkpoint used in separate GRECO studies is not involved here.

## Evidence

Executed source: `2c3d817`; full [design and commands](README.md).
[Machine-readable results](results.json) include ERRANT/M2 scores and update counts.
Each run under `experiments/training/matched-order-20261108/<family>-<order>/`
includes final score JSONs, recorded config/run metadata, exposure and initialization
manifests, and best-checkpoint metadata for diagnostics. Tables use final saved
scores consistently, not best-checkpoint or training-time callback scores.
Raw predictions, full training histories and weights stay on the cluster.
