# Single-stage BART and T5 curriculum controls

Question: how does a stage-free training recipe compare with Dasha's curriculum?
Both models start from their original pretrained checkpoint, not stage-1 weights.

## Data and exposure control

Use the exact hashed curriculum training files. Mix three copies of stage1
(561,338 pairs each) and five copies of stage2 (33,943 each), for 1,853,729 nominal
pair presentations. Shuffle with seed 42. Do not deduplicate: W&I errorful pairs
already overlap between stages, and their repeated exposure is intentional.
Correct sentences from stage2 are mixed throughout. Neither dev_tune nor dev_test
is added to training. The builder verifies all four source/target hashes against
the original data manifest before writing anything.

Each model trains once over this mixture using one optimizer and one continuous
linear learning-rate schedule; there is no stage boundary, optimizer reset or
checkpoint handoff. The trainer also shuffles/groups by length as in the original
recipe. Nominal examples match the prescribed 3+5 curriculum epochs; final-batch
rounding and any original early stopping mean actual processed counts can differ.

## Fixed settings

| Setting | BART | T5 |
|---|---|---|
| Initialization | facebook/bart-base | google-t5/t5-base |
| Peak learning rate | 5e-5 | 1e-4 |
| Warmup updates | 1,000 | 1,000 |
| Batch × accumulation | 32 × 4 = 128 | 32 × 4 = 128 |
| Epochs over mixed data | 1 | 1 |
| Weight decay / dropout / label smoothing | 0.01 / 0.1 / 0.1 | same |
| Seed | 42 | 42 |
| Checkpoint selection | full dev_tune, beam5 every 1,000 updates and at end | same |
| Early stopping | off, to retain the planned exposure | off |
| Final reporting | best-checkpoint dev_tune and dev_test, beam5 ERRANT + M2 | same |

These settings inherit the original stage-1 optimizer recipe. Smaller microbatches
retain its effective batch 128. Compared with Dasha's complete curriculum, the
stage-2 lower LR, effective batch 32, schedule/optimizer reset and intermediate
best-checkpoint selection are absent. This is therefore an **exposure-matched
single-stage recipe control**, not a causal isolation of data order. A cleaner
order-only experiment needs a matched curriculum rerun with identical optimization
and selection rules. Do not conclude that every difference is caused by ordering.

Historical dev_test reference scores: original BART stage1 43.84, stage2 52.26;
original T5 stage1 46.36, stage2 52.50. The long-T5 checkpoint is a separate recipe
and is not the reference here. These are one-seed comparisons.

## Run

```bash
python -m unittest discover -s tests -v
bash cluster/submit_mixed_control.sh --dry-run
# Configured cluster only:
bash cluster/submit_mixed_control.sh --submit
```

Array 0 is BART, 1 is T5; at most two jobs run at once. Each requests one H100-47
for up to 12h via the existing wrapper. Data and checkpoints live on scratch;
the two jobs build separate deterministic mixtures to avoid write races. Source
code must be synced into a dedicated cluster checkout/snapshot before submission.
Results go to `experiments/training/{bart,t5}-base.mixed-control/`. Retain
run_info.json, trainer_state.json, best_checkpoint.json, score JSONs and the mixed
data manifest. Report actual optimizer steps and selected checkpoint alongside F0.5.

Status: submitted; results pending.

## Submission — 8 November 2026

- Slurm job/array: **1094732**.
- Executed source commit: `41225d37af54a2d7105b7368751488ac72105eea`.
- Local validation: 24 tests: 23 passed, one GRECO environment integration test skipped.
- Dedicated cluster snapshot: `~/CS4248_GEC_experiments/mixed-control/`.
- This records submission, not a completed training or F-score result.

Snapshot dependencies `data/`, `external/`, `models/` and `m2scorer/` link to
the original cluster project. Training outputs and code remain isolated.

## Repair after initial submission

Array 1094732 failed before training (both tasks, exit 1) because the isolated
snapshot omitted the original root `m2_to_parallel.py` dependency. It has been
restored. The launcher now runs a CPU compute-node preflight covering source
imports, scorer files, exact historical software versions, tests and full data
validation. The GPU array depends on its success, avoiding repeated setup failures
on allocated GPUs. This changes validation/packaging, not the training recipe.

Replacement submission on 8 November 2026: CPU preflight **1094809**, dependent
BART/T5 array **1094810** (index0 BART, index1 T5). Executed source commit:
`39b9029`. Initial failed array 1094732 did not train either model. Results
remain pending; the training recipe and exposure counts have not changed.
