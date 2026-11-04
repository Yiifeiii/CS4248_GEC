# BART stage-2 hyperparameter ablations

Creating these files, running tests, and previewing the submission does not submit
cluster jobs. Actual submission requires `--submit`.

Stage 1 teaches general GEC from the broad BEA training mixture. Stage 2 adapts
the best stage-1 model to W&I training essays, including already-correct sentences.
Each run below independently starts from Dasha's completed stage-1 checkpoint,
`CS4248-GEC/gec-bart-base-s1`. Do not start these runs from the completed stage-2
checkpoint or from our 200-step pilot weights.

## Nine runs

| Array index | Config | Learning rate | Weight decay | Dropout | Label smoothing |
|---|---|---:|---:|---:|---:|
| 0 | `baseline.json` | 1e-5 | 0.01 | 0.1 | 0.1 |
| 1 | `lr-low.json` | 5e-6 | 0.01 | 0.1 | 0.1 |
| 2 | `lr-high.json` | 2e-5 | 0.01 | 0.1 | 0.1 |
| 3 | `wd-zero.json` | 1e-5 | 0 | 0.1 | 0.1 |
| 4 | `wd-high.json` | 1e-5 | 0.1 | 0.1 | 0.1 |
| 5 | `dropout-zero.json` | 1e-5 | 0.01 | 0 | 0.1 |
| 6 | `dropout-high.json` | 1e-5 | 0.01 | 0.3 | 0.1 |
| 7 | `smoothing-zero.json` | 1e-5 | 0.01 | 0.1 | 0 |
| 8 | `smoothing-high.json` | 1e-5 | 0.01 | 0.1 | 0.2 |

Everything else is fixed: stage-2 training pairs, batch 32, seed 42, AdamW,
100 warmup steps followed by linear learning-rate decay, up to 5 epochs,
evaluation each epoch on all 2,000 dev_tune sentences with 5 beams, and early
stopping after 2 evaluations without improvement. The training budget is a common
maximum; early stopping can give different runs different actual lengths.

`dropout_scope: all` applies the chosen dropout probability to BART's residual,
attention, and activation dropout. Setting only `config.dropout` would leave
attention and activation dropout active in the no-dropout run. LayerDrop is a
separate setting and stays disabled. Classifier-head dropout is unused for
conditional generation. Old configs without `dropout_scope` retain their original
single-field behaviour. The preflight checks that the stage-1 checkpoint has 0.1
for all three active fields, so the baseline preserves Dasha's effective recipe.

This isolates effects during stage-2 adaptation. It does not establish optimal
hyperparameters for stage 1 or for every writing domain. Multiple seeds would be
needed to check whether small improvements are reliable.

## Local tests and preview

Python 3.11 or 3.12 with `requirements-train.txt` installed:

```bash
HF_HOME=/tmp/gec-ablation-test-cache HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
bash cluster/submit_bart_ablations.sh --dry-run
```

The training smoke test constructs a tiny random BART locally, runs two updates
for each recipe on synthetic token sequences, and saves/reloads each model.
It does not download model weights, update the real checkpoints, or measure GEC
quality. Warmup is disabled only in the two-step test, so both updates can be
checked. Tests also verify single-factor changes, unique checkpoint paths,
dropout propagation to actual model layers, and that the default submission
command does not call Slurm.

## Cluster preparation and later submission

Use the code from this branch on the cluster. Our existing cluster directory is
a source snapshot, not a Git checkout: copy the new configs, the three
`cluster/*bart_ablations*.sh` scripts, `training/dropout.py`, the updated
`training/finetune.py`, and `tests/test_bart_ablations.py` before submitting.
Use LF line endings for shell scripts and committed data. Follow
`cluster/README.md` for the existing `gec-train` environment, scratch HF cache,
authentication, and stage-1 data preparation.

When explicitly ready to start jobs, run from the cluster repository root:

```bash
bash cluster/submit_bart_ablations.sh --submit
```

This submits a CPU preflight (no GPU), then an array of nine independent GPU jobs,
with at most two array tasks running at once. Each training task requests one
H100-47 GPU and a four-hour limit on `gpu-long`. Queue delays and actual runtimes
determine whether the whole sweep finishes overnight.

The preflight matches the baseline against Dasha's recorded `run_info.json` and
requires her recorded PyTorch 2.11.0+cu128 and Transformers 4.46.3 versions.
It downloads Dasha's private stage-1 checkpoint to the scratch cache,
verifies its SHA-256 against `experiments/training/checkpoints.sha256`, checks
baseline dropout, validates data, and runs the tests on the compute node. Training
starts only if preflight succeeds; jobs run offline and auto-resume their own
checkpoints if resubmitted. `runs.txt` is the array-index mapping.

Outputs:

- Checkpoints: `$CKPT_ROOT/bart-ablations-20261105/<config name>/`.
- Results: `experiments/training/bart-ablations-20261105/<config name>/`.
- Source-checkpoint provenance: `experiments/training/bart-ablations-20261105/preflight.json`.
- Logs: `logs/slurm/bart-ablations-<task job ID>.out`.

Each task records the config, effective dropout settings, training history, best
checkpoint, and dev_tune/dev_test predictions and ERRANT/M2 scores, matching Dasha's
final reporting. Only dev_tune drives checkpoint selection and early stopping.
Choose hyperparameters using dev_tune; use dev_test to report the final comparison,
not to keep selecting new settings. Small hardware-dependent numerical differences
are possible even with the same recipe; the rerun baseline controls for the shared
environment of this sweep.

Do not change configs in place and resume old checkpoints: give a revised
experiment a new run/output path. The original Dasha checkpoints and prior pilot
outputs have separate paths.
