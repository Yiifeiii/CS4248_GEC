# Four controls against one curriculum baseline

**Completed:** all eight controls succeeded. [Results](results.md) · [Numerical results](results.json).

Each model family has one reference: the original BART-base or T5-base two-stage
curriculum checkpoint. Four new single-stage shuffled controls are compared with
that same reference. BART starts from facebook/bart-base; T5 starts from
google-t5/t5-base. Stage-2 optimization controls also start from those raw
pretrained weights, not a stage-1 checkpoint.

Training data definitions:
- **Stage-1 data:** broad errorful sentence pairs.
- **Stage-2 data:** W&I sentence pairs, including correct sentences.

| Control | Training exposure | Optimization settings used throughout |
|---|---|---|
| 1A | Match the reference's planned per-example exposures: 3 copies of stage-1 data + 5 copies of stage-2 data | Stage-1 settings |
| 1B | Exactly the same exposure stream as 1A | Stage-2 settings |
| 2A | One combined dataset of unique source/target pairs, each seen exactly 3 times | Stage-1 settings |
| 2B | The same combined dataset of unique source/target pairs, each seen exactly 5 times | Stage-2 settings |

Controls 1A/1B mix all presentations throughout a single continuous run. They
retain original duplicates and stage overlap: their source/target multiplicity
is exactly `3 * Counter(stage1) + 5 * Counter(stage2)`.

Controls 2A/2B remove exact source/target duplicates both within and across
stages, so every unique training pair really has equal exposure. Different
targets for the same source remain different pairs. Each logical pass shuffles
the entire unique dataset independently; there are 3 or 5 passes, respectively.
No dev/test examples are added to training. Input hashes are checked against
the original curriculum manifest. Unique-pair counts and stream hashes are
recorded by preflight before GPU jobs become eligible.

## Shared settings and stage-specific optimization

| Setting | Stage-1 settings (1A/2A) | Stage-2 settings (1B/2B) |
|---|---:|---:|
| BART learning rate | 5e-5 | 1e-5 |
| T5 learning rate | 1e-4 | 3e-5 |
| Microbatch size | 64 | 32 |
| Gradient accumulation | 2 | 1 |
| Effective full-update batch | 128 | 32 |
| Warmup updates | 1,000 | 100 |
| Optimizer | AdamW | AdamW |
| Learning-rate schedule | One continuous linear schedule | One continuous linear schedule |
| Weight decay / dropout / label smoothing | 0.01 / 0.1 / 0.1 | Same |
| Seed | 42 | 42 |

Epoch/exposure budget follows the first table, not the copied stage config's
original epoch field. Files explicitly contain the full exposure stream, and
Trainer makes one physical pass. Thus 2A/2B execute 3/5 logical full-dataset
passes without accidentally multiplying an already repeated dataset again.

Controls use a sequential loader over the pre-shuffled files, with no second
shuffle or length grouping. All examples are retained. The final incomplete
update is split into nonempty microbatches so stage-1 accumulation remains 2
without dropping or duplicating examples. Full updates retain the configured
microbatch/effective batch; only the final incomplete update may be smaller.
The rare one-example tail borrows the preceding full update and repartitions
those examples across its two accumulation groups. Tests verify real training
consumes every original index once. The historical baseline has nominal 3+5
exposures; its original accumulation rounding is not retroactively changed.

## Evaluation and interpretation

- No early stopping: the configured exposure budget is always completed.
- All controls use full dev_tune, five beams, with ERRANT checkpoint selection
  every 128,000 examples and at the end (1,000 stage-1-setting updates or 4,000
  stage-2-setting updates). Loss/checkpoint saves share those update intervals.
- **Primary reporting:** best-dev_tune checkpoint on dev_tune and dev_test,
  matching the historical reference's selection rule and final scorer.
- Final-checkpoint scores are retained separately as a fixed-budget diagnostic.
  Never mix best and final endpoints in one result table. The reference's
  historical checkpoint-evaluation cadence is not being rerun or changed.
- Reference dev_test F0.5: BART 52.26; T5 52.50 (standard 3+5 recipe, not long-T5).

1A/1B ask whether the complete curriculum recipe beats shuffled training with
matched nominal exposures under either stage's optimization settings. 2A/2B
ask whether it beats equal-exposure ordinary training. These are recipe-level
comparisons, not ordering-only tests. This is one seed per configuration; no significance claims are made.

## Run and provenance

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
bash cluster/submit_four_controls.sh --dry-run
# Configured SoC cluster:
bash cluster/submit_four_controls.sh --submit
```

Array order: BART 1A/1B/2A/2B, then T5 1A/1B/2A/2B. CPU preflight verifies
dependencies, exact historical software versions, tests and source data, then
builds the streams and records pretrained weight hashes. Eight GPU jobs depend
on its success, at most two concurrent, each requesting one H100-47 for up to12h.

Configs: `configs/training/four-controls/`. Data/checkpoints live on scratch.
Results: `experiments/training/four-controls-20261108/<family>-<control>/`, with
separate `best/` and `final/` scores and exposure/initialization provenance.
No model training is performed on login nodes. All eight runs are now complete.

## Submission — 8 November 2026

CPU preflight **1097381**, dependent GPU array **1097382**, at most two tasks
concurrently. Executed source commit: `1a4c872`. Isolated source snapshot:
`~/CS4248_GEC_experiments/four-controls/`. Local validation: 34 tests, 33 passed
and one environment-dependent GRECO integration test skipped; shell syntax and
whitespace checks passed. All tasks subsequently completed with exit0.

| Array index | Model | Control |
|---|---|---|
| 0 | BART | 1A |
| 1 | BART | 1B |
| 2 | BART | 2A |
| 3 | BART | 2B |
| 4 | T5 | 1A |
| 5 | T5 | 1B |
| 6 | T5 | 2A |
| 7 | T5 | 2B |
