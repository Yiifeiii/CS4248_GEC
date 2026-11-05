# T5 stage-2 tuning

Tunes the learning rate, weight decay, dropout and label smoothing for T5 stage 2, starting from the frozen stage-1 model. Decoding is not part of this folder.

## Constraints

- **No existing repo file is edited.** The one exception is the `training/README.md` edit (Cleaning section), which was approved and is still uncommitted.
- **New files go in `t5_tuning/`**, and the results and logs of runs are new files too. Checkpoints go to `$CKPT_ROOT` on scratch, because each one is a few GB.
- **Existing code is reused, not copied.** Runs call `training.finetune` and `training.evaluate` as external processes, through `cluster/job.sbatch`.
- **`dev_test` is not touched.** `run_one.sh` scores `dev_tune` only. `cluster/train.sh` would also score `dev_test` for every run, so it is not used.

## How the existing code takes its parameters

Checked against `training/finetune.py` and `training/evaluate.py`.

**Set in the JSON config, not by a flag.** Each run gets its own config file in `configs/`.

| Parameter | Config key | Used at |
|---|---|---|
| learning rate | `learning_rate` | `finetune.py:168` |
| weight decay | `weight_decay` | `finetune.py:168` |
| dropout | `dropout` | `finetune.py:141`, sets `config.dropout_rate` for T5. This overrides the value saved in the checkpoint. |
| label smoothing | `label_smoothing` | `finetune.py:170` |
| seed | `seed` | `finetune.py:164`, sets both `seed` and `data_seed` (shuffling and dropout randomness) |
| starting weights | `init` | `finetune.py:136–143`, passed to `from_pretrained`. The Hugging Face repo name works, and the offline job reads it from the cache. |
| output folder | `output_dir` | `finetune.py:134` |

**Command-line flags.**
- `finetune.py`: `--config` (required), `--output-dir`, `--resume {auto,no}` (default `auto`), `--max-steps`, `--train-limit`, `--dev-limit`, `--errant-steps`. These control the run, not the hyperparameters.
- `evaluate.py`: `--model` (required), `--out` (required), `--split` (default `dev_tune dev_test`; `run_one.sh` passes `dev_tune`), `--beams` (default 5), `--batch-size`, `--limit`, `--prefix`, `--retokenize`, `--device`.

**Invocation.** `finetune.py` imports `training.common`, so it must run as a module from the repo root: `python -m training.finetune --config ...`. The form `python ../finetune.py` does not work.

**Hardcoded, not configurable** (none of these needed a change for this study):
- **Dev paths.** `finetune.py:149, 155, 158` read `data/curriculum/dev_tune.*`. `dev_tune` drives the loss evaluation, the ERRANT evaluation, best-checkpoint selection and early stopping. `finetune.py` never references `dev_test`.
- **Optimizer and schedule.** AdamW (`adamw_torch`, line 168), linear scheduler (line 169), eval batch twice the train batch (line 166), `group_by_length=True` (line 171).
- **Gradient clipping.** Not set, so the Trainer default applies. I believe it's a maximum norm of 1.0, but I haven't checked the installed version.
- **Truncation.** `MAX_TOKENS = 128` (`training/common.py:26`), applied at `finetune.py:49`.

**Precision.** bf16 is automatic (`finetune.py:171`, `bf16 = cuda and torch.cuda.is_bf16_supported()`). There is no fp16 path. On a CUDA GPU without bf16 it falls back to fp32 silently, and nothing records that. Decoding also runs under bf16 autocast on CUDA (`training/common.py:138`). `run_info.json` records the GPU name, not the precision.

**What the score is.** The sweep metric is correction-level ERRANT F0.5 (`errant.correction.f0_5`) on `dev_tune`. A run's score is its best checkpoint's score, and that checkpoint was chosen using `dev_tune`, so the number is optimistic. The bias is the same for every run, so comparisons between runs still hold. `dev_test` is the unbiased number, and it is scored once, at the end.

## Decisions

- **Baseline learning rate is 3e-5**, the group's stage-2 default (`training/configs/t5-base.stage2.json`). This matches the recorded `t5-base.s1s2` run (dev_tune F0.5 55.21), so the baseline is comparable to it.
- **Grid.** Each knob's default sits in the middle, with one step down and one step up:
  - learning rate `{1e-5, 3e-5, 5e-5}`
  - weight decay `{0, 0.01, 0.1}`
  - dropout `{0, 0.1, 0.3}`
  - label smoothing `{0, 0.1, 0.2}`
- **Order.** The learning rate is swept first. Its winner is `lr*`. Then weight decay, dropout and label smoothing are each varied alone at `lr*`, against the baseline at `lr*`.
- **Seeds.** The noise floor uses seeds `42, 123, 7` at the default settings. Sweep runs use seed 42 only.
- **Noise band.** The band is the range of the three baseline seeds. The range is simple, and with three seeds it is only a rough estimate. The standard deviation is reported too.
- **Judging a difference.** A candidate's difference from its reference is inside the band if its absolute size is at most the band. A value is chosen only if it beats the reference by more than the band. A smaller gain keeps the default and calls for a second seed before it is reported. This matches the rule in the spec.
- **Endpoints.** If a chosen value is the first or last point of its grid, the report flags it, and the grid should be extended in that direction.
- **Stability.** The collector does not check for NaN or loss spikes. A run that diverges usually scores low, so it loses the comparison anyway. Check the loss in `trainer_state.json` for any run you rely on.
- **Compute.** Stage-2 runs take about half an hour each. Phase 1 is 5 runs, and phase 2 is 7 runs (the baseline at `lr*` plus 6 variants). The runs in a phase are independent, so they run in parallel.

## Files

| File | Purpose |
|---|---|
| `tune.py` | Plans the runs, writes their configs and manifest, submits them, and collects the results |
| `run_one.sh` | Runs one job: train, then score `dev_tune` only. Passed to `cluster/job.sbatch`. |
| `fetch_stage1.sh` | Downloads the frozen stage-1 model and checks its sha256 |
| `configs/` | One JSON config per run, written by `tune.py plan` |
| `manifest.json`, `manifest.md` | The run manifest: every run, its settings, and its config file |
| `runs/<run>/` | Per-run results: `dev_tune.scores.json` and `dev_tune.pred` |
| `results.md` | The table written by `tune.py collect` |
| `README.md` | This file |

## How to run

From the repo root.

**1. Cache and check the stage-1 model**, once, on a compute node:

```bash
salloc -t 30 && srun --pty bash
conda activate gec-train && bash t5_tuning/fetch_stage1.sh
exit
```

**2. Phase 1: noise band and learning-rate sweep**, on xlogin:

```bash
python3 t5_tuning/tune.py plan --phase 1           # writes the configs and manifest
python3 t5_tuning/tune.py submit --phase 1         # prints the sbatch commands
python3 t5_tuning/tune.py submit --phase 1 --run   # submits them
squeue --me                                         # wait for the jobs to finish
python3 t5_tuning/tune.py collect                  # band, learning-rate table, chosen value
```

Submit each phase once. `submit` skips runs that already have a `dev_tune` score, but a run that is still queued would be submitted again.

**3. Phase 2: weight decay, dropout and label smoothing**, at the learning rate chosen in phase 1:

```bash
python3 t5_tuning/tune.py plan --phase 2 --lr <lr*>
python3 t5_tuning/tune.py submit --phase 2 --run
python3 t5_tuning/tune.py collect
```

**4. Second seeds, for any candidate that beats the default by less than the band.** `tune.py` doesn't plan these yet. Each one needs a config with a new seed and the same settings, which can be copied from the candidate's config in `configs/`.

**5. The final check**, after all choices are made: run the chosen setting with the seeds above and report the mean and the spread. Then score `dev_test` once. Keep that step separate from tuning.

Logs for each job go to `logs/slurm/` in the repo, which `cluster/job.sbatch` creates.

## Run manifest

Generated by `tune.py plan`, in `manifest.md`. Phase 1 has five runs: the three baseline seeds at 3e-5, and seed 42 at 1e-5 and 5e-5. Phase 2 adds the rest, depending on `lr*`.

All phase-1 and phase-2 runs have finished, and every score is `dev_tune` ERRANT F0.5 (correction level, percent). The table is also written by `tune.py collect` to `results.md`.

**Noise band.** Three seeds at the default settings: 55.61, 55.44 and 55.29. Mean 55.45, sd 0.16, range 0.32. The range is the band.

**Learning rate** (seed 42; reference is the default 3e-5 run):

| Learning rate | dev_tune F0.5 | Change vs reference | Verdict |
|---|---|---|---|
| 1e-5 | 54.28 | −1.33 | outside band, worse |
| **3e-5** (default) | 55.61 | — | reference |
| 5e-5 | 55.72 | +0.11 | inside band; top of the grid |

Chosen: **3e-5**. The best candidate, 5e-5, beats the default by less than the band, so it isn't a reported gain. Because 5e-5 is the top point of the grid, it's flagged as an endpoint. Its gain is within noise, so the grid was not extended upward.

**Phase 2 at 3e-5** (seed 42; reference is the default run):

| Setting | dev_tune F0.5 | Change vs reference | Verdict |
|---|---|---|---|
| Weight decay 0 | 55.13 | −0.48 | outside band, worse |
| **Weight decay 0.01** (default) | 55.61 | — | reference |
| Weight decay 0.1 | 55.25 | −0.36 | outside band, worse |
| Dropout 0 | 54.56 | −1.05 | outside band, worse |
| **Dropout 0.1** (default) | 55.61 | — | reference |
| Dropout 0.3 | 50.73 | −4.88 | outside band, worse |
| Label smoothing 0 | 54.87 | −0.74 | outside band, worse |
| **Label smoothing 0.1** (default) | 55.61 | — | reference |
| Label smoothing 0.2 | 55.03 | −0.58 | outside band, worse |

Chosen: **all three defaults**. Every variant scores below the reference by more than the band.

Two caveats for the write-up:
- The reference is seed 42, the highest of the three default seeds. Against the three-seed mean (55.45), weight decay 0.1 is −0.20, which falls inside the band. The verdict is the same, but the gap is small.
- Dropout 0.3 was still improving at the last step (step 5305), so the fixed five-epoch budget may have cut it short. Its recall stayed around 0.29, against about 0.40 for the default.

**Chosen configuration:** learning rate 3e-5, weight decay 0.01, dropout 0.1, label smoothing 0.1, with stage 1 frozen at `CS4248-GEC/gec-t5-base-s1`.

**Endpoint flags:** learning rate 5e-5 (top of the grid, inside the band). Nothing else was at an endpoint.

**Hardcoded parameters that needed a change:** none.

**Final `dev_test` score** for the chosen configuration, scored once per seed checkpoint (the `best/` model from each seed run), with the scores in `runs/final-dev_test/seed<N>/`:

| Seed | dev_test ERRANT F0.5 |
|---|---|
| 42 | 52.5 |
| 123 | 52.9 |
| 7 | 52.2 |

Mean 52.53, range 0.70, sd 0.35. For comparison, the group's recorded `t5-base.s1s2` run scored 52.5 on `dev_test`. The `dev_test` mean is about three points below the `dev_tune` mean of 55.45, which is the selection optimism described above.

## Status

- `tune.py` was tested locally on fake scores in a temporary folder, and its logic is checked for a clear win, an endpoint, a near-tie and a default that is kept.
- Phase 1 and phase 2 have run on the cluster, and the results above are from those runs.
- Stage 1 was not tuned. It stays frozen as `CS4248-GEC/gec-t5-base-s1`.
