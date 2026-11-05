# Exploratory follow-up: 0.05 regularisation values

Three additional BART stage-2 runs extend the completed nine-run sweep. Each starts
independently from the same verified `CS4248-GEC/gec-bart-base-s1` weights and uses
the same data, software, seed, optimiser, warmup/linear decay, five-epoch maximum,
early stopping, and ERRANT/M2 evaluation as the original runs.

| Follow-up array index | Config | Learning rate | Weight decay | Dropout | Label smoothing |
|---|---|---:|---:|---:|---:|
| 0 | `wd-mid` | 1e-5 | 0.05 | 0.1 | 0.1 |
| 1 | `dropout-low` | 1e-5 | 0.01 | 0.05 | 0.1 |
| 2 | `smoothing-low` | 1e-5 | 0.01 | 0.1 | 0.05 |

The original `baseline` is reused as the control. New run names have distinct
checkpoint/result paths; the original nine-run list is unchanged. `0.05` is
between the tested weight-decay values 0.01 and 0.1, and below the baseline 0.1
for dropout and label smoothing. These three values are separate interventions,
not one run changing all three together.

This extension was chosen after the original results had been inspected. Compare
settings using dev_tune and describe further dev_test comparisons as exploratory;
do not claim the extension was a preplanned held-out experiment.

The [configuration README](../../../training/configs/bart-ablations-20261105/README.md#follow-up-three-005-values)
contains the preview/submission commands. Selecting `runs-005.txt` submits only
these three tasks, with a fresh preflight dependency and at most two concurrent
GPU jobs.

## Submission

- Submitted on 5 November 2026.
- Executed source commit: `d3fc43a66c2d1b850d716e9b970f85311b026700`.
- Preflight job: **1078869**.
- Training array: **1078870**, indices 0–2, at most two concurrent tasks.
- Mapping: `1078870_0` = `wd-mid`, `1078870_1` = `dropout-low`,
  `1078870_2` = `smoothing-low`.
- Each training task requests one H100-47 GPU and the same four-hour limit.
- At submission, preflight was queued; training was gated on its success.

```bash
squeue -j 1078869,1078870
sacct -j 1078869,1078870 --format=JobID,State,ExitCode,Elapsed -X
```

No follow-up results have been recorded yet. Result/checkpoint folders use the
three new config names alongside the completed original nine folders.
