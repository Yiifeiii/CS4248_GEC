# BART stage-2 hyperparameter ablations — 5 November 2026

**Completed:** all nine original jobs and all three 0.05 follow-up jobs finished
successfully (exit code 0). The [consolidated twelve-run results](results-all-12.md)
were retrieved and validated on 6 November 2026. Curated per-run scores are
versioned; raw predictions and training histories remain local/on the cluster.

Nine independent BART-only runs test learning rate, weight decay, dropout, and
label smoothing during stage-2 adaptation. The reference is Dasha's recorded
`bart-base.s1s2` recipe, rerun alongside the variants in the same environment.
Each variant changes one hyperparameter; these are not cumulative changes.

The complete numerical grid and launch instructions are in the
[ablation index](../../ablations/README.md).
The [ordered run list](../../../training/configs/bart-ablations-20261105/runs.txt)
maps each Slurm array index to a config.

## Questions and runs

| Array task | Config | Difference from baseline | Question |
|---|---|---|---|
| `1075239_0` | `baseline` | None | Reproduce Dasha's recipe in this sweep's environment |
| `1075239_1` | `lr-low` | LR 1e-5 → 5e-6 | Do smaller updates improve adaptation? |
| `1075239_2` | `lr-high` | LR 1e-5 → 2e-5 | Do larger updates improve adaptation? |
| `1075239_3` | `wd-zero` | Weight decay 0.01 → 0 | Is weight decay helping? |
| `1075239_4` | `wd-high` | Weight decay 0.01 → 0.1 | Does stronger weight decay help? |
| `1075239_5` | `dropout-zero` | Dropout 0.1 → 0 | Is dropout helping? |
| `1075239_6` | `dropout-high` | Dropout 0.1 → 0.3 | Does stronger dropout help? |
| `1075239_7` | `smoothing-zero` | Label smoothing 0.1 → 0 | Is label smoothing helping? |
| `1075239_8` | `smoothing-high` | Label smoothing 0.1 → 0.2 | Does stronger smoothing help? |

Baseline: LR 1e-5, weight decay 0.01, dropout 0.1, label smoothing 0.1.
Dropout covers residual, attention, and activation dropout together. LayerDrop
stays off; classifier-head dropout is unused by seq2seq generation. The baseline
checkpoint was checked to use 0.1 for all three active dropout fields. This is
why configuring all three preserves the baseline while making "no dropout" real.

## Fixed training and evaluation procedure

- **Starting weights:** every run independently loads
  `CS4248-GEC/gec-bart-base-s1`, Dasha's completed stage-1 BART model. No run starts
  from the finished stage-2 model, another ablation, or the 200-step pilot.
- **Training data:** the same 33,943 W&I stage-2 pairs, including 11,262 unchanged
  pairs, with the committed preprocessing, token-length limit, order, and hashes.
- **Optimisation:** AdamW, batch 32, accumulation 1, seed/data seed 42,
  100 warmup steps, then linear learning-rate decay; up to 5 epochs.
- **Precision/batching:** bf16 on a supported CUDA GPU and grouping by length,
  using the original training implementation.
- **Monitoring:** loss evaluation and checkpoint saves every 500 steps; keep
  the last two resumable checkpoints; log every 50 steps.
- **Model selection:** ERRANT correction F0.5 on all 2,000 dev_tune sentences
  every epoch, decoded with 5 beams. Save the best checkpoint; stop after two
  evaluations without improvement. Early stopping can give runs different actual
  training lengths despite the common five-epoch maximum.
- **Final reporting:** decode the best checkpoint with 5 beams on dev_tune and
  dev_test (2,384 sentences), reporting ERRANT P/R/F0.5 and M2 P/R/F0.5. This
  matches Dasha's reporting pipeline. Only dev_tune informs checkpoint and
  hyperparameter selection; dev_test is for the final comparison.

No generator beam/group sweep, GRECO training, GECToR changes, or T5 training is
part of this experiment.

## Submission and provenance

- Submitted **5 November 2026, 00:35:02 Asia/Singapore**.
- Source branch: `raeeda/bart-stage2-ablations`.
- Executed source commit: `ed9cb03774ed56f8a68231e944fa539e010d04bd`, also in
  [source_commit.txt](source_commit.txt). Later documentation-only commits do not
  change the code already deployed for these jobs.
- Preflight job: **1075238**, completed successfully in **3m 36s**.
- Training array: **1075239**, indices 0–8, at most two running concurrently.
- Per training task: one H100-47 request, eight CPUs, 64,000 MB total requested
  host memory, `gpu-long`, four-hour time limit. Actual GPU assignment is recorded
  in each task's `run_info.json`; hardware can differ from Dasha's historical run.
- Software checked against Dasha's recorded versions: PyTorch **2.11.0+cu128**,
  Transformers **4.46.3**.
- Stage-1 weights checked against the recorded SHA-256:
  `40b41277a8bf99c76009d5b6b7939fbdb3d118593bc87d64381e72da507e8986`.
- The downloaded HF snapshot and check results are in [preflight.json](preflight.json).

The cluster code directory is a source snapshot without `.git`. Its automatic
`run_info.json` commit field may therefore be null; use the explicit source commit
above and the run configs for provenance. Original checkpoints and earlier pilot
outputs have separate paths.

## Validation before training

Both local validation and cluster preflight ran **29 tests: 28 passed and one
GRECO integration test skipped** because it requires its separate environment.
The tests check the grid against Dasha's recorded config, single-factor changes,
noncolliding output paths, backward-compatible dropout behaviour, actual BART
dropout propagation, and preview-only submission by default. Tiny offline BART
tests also run two CPU updates per recipe and save/reload each model; those are
synthetic smoke tests, not the real GEC runs.

Cluster preflight additionally verified the source weight hash, software versions,
training-data hashes/provenance, dev partition, sequence lengths, and tokenizer
round trips. Training depended on that job succeeding. Shell syntax and preview
checks passed before submission.

## Monitoring and output collection

```bash
squeue -j 1075239
sacct -j 1075239 --format=JobID,State,ExitCode,Elapsed -X
```

`cluster/job.sbatch` logs to `logs/slurm/bart-ablations-<individual task job ID>.out`.
The task ID in this filename is Slurm's individual job ID, which can differ from
the displayed `1075239_<array index>` notation. Use `scontrol show job
1075239_<index>` to inspect a task. The preflight log is
`logs/slurm/bart-ablation-check-1075238.out`.

Each completed task copies its config, run metadata, loss/ERRANT history, selected
checkpoint metadata, predictions, and scores into `<config name>/` in this folder.
Best/final weights and resumable checkpoints stay under
`$CKPT_ROOT/bart-ablations-20261105/<config name>/`, outside Git.

Copy completed result folders back with the rsync procedure in
[cluster/README.md](../../../cluster/README.md#copying-results-back). Retain raw
predictions and histories locally or on the cluster. Commit curated score JSONs
and summaries with reported results. Never resume a
changed config into the same checkpoint folder or label the two-update smoke-test
outputs as ablation results.

## Results status

**As of 5 November 2026, 00:42:31 Asia/Singapore:** preflight passed, baseline
task 0 was running, and tasks 1–8 were pending resources. There were no completed
ablation results to report at that check. This is a timestamped snapshot, not a
live status indicator.

When results are collected, compare each variant against the rerun baseline using
dev_tune, reporting P/R/F0.5, differences in F0.5 points, selected epoch/step, and
runtime. Report dev_test as the held-out comparison. Dasha's historical 52.3
dev_test F0.5 is a reference rather than a guaranteed bit-for-bit target. These
are single-seed stage-2 experiments; small gains need replication, and conclusions
do not establish optimal settings for stage 1 or every writing domain.
