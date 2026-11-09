# T5 tuning methodology

This records how the T5 hyperparameters were tuned for the GEC project. Every number comes from the run logs and results tables in `t5_tuning/` and `experiments/`. Anything that wasn't run or wasn't recorded is marked as such.

## 0. Handover summary

**Outcome.** Tuning confirmed the group's default T5 settings. 21 runs were made: two noise bands, a stage-2 learning-rate sweep, weight decay, dropout and label smoothing sweeps, a stage-1 learning-rate sweep, and a stage-2 re-run on the best stage-1 model. Once both stages were trained, no change beat the defaults by more than the noise band. The final pipeline is unchanged: the group's stage-1 model `CS4248-GEC/gec-t5-base-s1`, then stage 2 at learning rate 3e-5, weight decay 0.01, dropout 0.1 and label smoothing 0.1.

| Result | Value | Where |
|---|---|---|
| Stage-2 noise band, dev_tune (3 seeds) | mean 55.45, range 0.32 | section 4, `results.md` |
| Best single alternative, stage 2 | lr 5e-5: +0.11 (inside band) | section 5.1 |
| Stage-1 lr 3e-4 on stage 1 alone | +1.11 on dev_tune | section 5.5, `results-stage1.md` |
| The same 3e-4 model after stage 2 | mean 54.84, 0.61 *below* the default pipeline | section 5.5, `results-s1lr3e-4.md` |
| **Final configuration, dev_test (3 seeds)** | **mean 52.53** (52.49 / 52.92 / 52.18) | section 8, `runs/final-dev_test/` |

**Quick check (laptop, about a second, standard-library Python 3 only):**

```bash
python3 t5_tuning/tune.py collect && python3 -m unittest tests.test_t5_tuning
```

This rebuilds `results.md` from the committed scores and prints the band and the verdict for each sweep. Retraining anything needs the SoC cluster setup in `cluster/README.md`; then follow section 7.

**What's in this folder**

| Path | Contents |
|---|---|
| `README.md` | This document: method, results, code map, how to reproduce |
| `tune.py` | One CLI for every step: `plan`, `submit`, `preflight`, `verify`, `collect`, `test` (`python3 t5_tuning/tune.py -h`) |
| `configs/` | The training config of each of the 21 runs (input to `training/finetune.py`) |
| `manifest.json`, `manifest.md` | Every planned run with its settings. The configs are checked against it. |
| `runs/<run>/dev_tune.scores.json` | ERRANT and M² scores of each run's best checkpoint on dev_tune |
| `runs/final-dev_test/seed<N>/dev_test.scores.json` | dev_test scores of the final configuration, one per seed |
| `results.md`, `results-stage1.md`, `results-s1lr3e-4.md` | Results tables written by `tune.py collect` |
| `run_one.sh`, `run_test.sh`, `preflight.sh`, `fetch_stage1.sh` | Cluster job scripts (section 6) |
| `../tests/test_t5_tuning.py` | Tests for the configs and the run checks |

**Not in the repo.** The checkpoints, predictions and Slurm logs stay on cluster scratch (`$CKPT_ROOT/<run>/`, `logs/slurm/`). Only the scores are committed. The group's T5 models are on Hugging Face as `CS4248-GEC/gec-t5-base-s1` and `CS4248-GEC/gec-t5-base-s1s2` (see `experiments/training/README.md`).

**Open items for whoever picks this up**

1. Add references at the `[CITATION]` markers in section 5 before this goes into the report.
2. Confirm whether anyone looked at dev_test before the final scoring (see the end of section 5.5).
3. Optional: stage-1 learning rate 1e-3 was never run, since 3e-4 is the top of the grid. Stage-2 learning rate 5e-5 was run with only one seed.

## 1. Glossary

- **Default settings.** The values already in the group's config files: `training/configs/t5-base.stage1.json` for stage 1 and `training/configs/t5-base.stage2.json` for stage 2. They are fixed numbers, not random values, and they are not the base model. Each sweep changes one of them and keeps the rest at these values. The group chose them before this project, and the repo doesn't record how.
- **Default run.** A training run with the default settings. Its seed is 42 unless stated otherwise.
- **Reference.** The run a candidate is compared with. For the learning-rate and regularization sweeps, it's the seed-42 default run.
- **Base model.** The pretrained `google-t5/t5-base` weights from Hugging Face. Stage 1 starts from it. It's a different thing from the default settings.
- **Stage 1 and stage 2.** Stage 1 trains on the BEA-2019 corpora, errorful sentences only. Stage 2 trains on the W&I+LOCNESS training data, which includes correct sentences.
- **Frozen stage-1 model.** The group's stage-1 checkpoint, `CS4248-GEC/gec-t5-base-s1`. Every stage-2 run in the main experiments starts from it, so they share one starting point.
- **Noise band.** The spread of scores across seeds at the default settings. A difference smaller than the band is treated as noise.
- **dev_tune.** The 2,000-sentence dev split used for every choice.
- **dev_test.** The 2,384-sentence dev split held out for reporting. It's scored once per configuration, after the choices are made.
- **Tag.** A name for stage-2 runs that start from a different stage-1 model, such as `s1lr3e-4`. Tagged runs and their results are kept apart from the untagged runs.
- **F0.5.** Correction-level ERRANT F0.5, as a percentage. Every decision uses it.

Every F0.5 value in this document is read from `errant.correction.f0_5` in a `scores.json` file, multiplied by 100.

## 2. Data and models

- **Stage 1** trains on the BEA-2019 corpora, errorful sentences only. It has 561,338 pairs after cleaning, and it starts from the base model. Source: `data/curriculum/manifest.json`.
- **Stage 2** trains on the W&I+LOCNESS training data, including correct sentences. It has 33,943 pairs, and it starts from the frozen stage-1 model. Source: `data/curriculum/manifest.json`.
- **Dev splits** come from the W&I+LOCNESS dev set. `dev_tune` (2,000 sentences) is used for every choice. `dev_test` (2,384 sentences) is held out for reporting.
- **Metric:** correction-level ERRANT F0.5. Each run's dev_tune score is that of its best checkpoint, which was chosen using dev_tune. Those scores are optimistic, in the same way for every run.
- **Frozen stage-1 checkpoint:** `CS4248-GEC/gec-t5-base-s1`. Its sha256 matches `experiments/training/checkpoints.sha256`.

These are the default settings the sweeps start from:

| Setting | Stage 1 | Stage 2 |
|---|---|---|
| Learning rate | 1e-4 | 3e-5 |
| Weight decay | 0.01 | 0.01 |
| Dropout | 0.1 | 0.1 |
| Label smoothing | 0.1 | 0.1 |
| Batch size × gradient accumulation | 64 × 2 | 32 × 1 |
| Warmup steps | 1,000 | 100 |
| Epochs | 3 | Up to 5, with early stopping (patience 2) |
| Seed | 42 | 42 |

Source: `training/configs/t5-base.stage1.json` and `training/configs/t5-base.stage2.json`.

## 3. Process

1. **Cache the frozen stage-1 model and check it.** `fetch_stage1.sh` downloads `CS4248-GEC/gec-t5-base-s1` and checks its sha256. The check passed.
2. **Noise band for stage 2.** Run the default settings with seeds 42, 123 and 7. The spread of these runs is the noise band.
3. **Stage-2 learning-rate sweep.** Vary the learning rate with seed 42, and keep everything else at default. Compare each value with the default run at seed 42.
4. **Stage-2 regularization sweeps** at the chosen learning rate, one hyperparameter at a time, with seed 42. Each one is compared with the same default run.
5. **dev_test for the stage-2 default**, once per seed.
6. **Stage-1 learning-rate sweep.** Vary the stage-1 learning rate, with a noise band from three seeds at the default.
7. **Stage-2 re-run on the 3e-4 stage-1 model.** Done. Its noise band and learning-rate sweep were run, and the result is lower than the original pipeline. See section 5.5.

A value is chosen only if it beats the reference by more than the noise band. Otherwise the default is kept. The rule is in section 4.

## 4. Noise band

**Stage 2** (seeds 42, 123, 7; default settings), dev_tune F0.5:

| Seed | F0.5 |
|---|---|
| 42 | 55.61 |
| 123 | 55.44 |
| 7 | 55.29 |

Source: `t5_tuning/runs/t5-tune.base.lr3e-5.seed42/`, `seed123/` and `seed7/`, each `dev_tune.scores.json`.

- **Mean** 55.45. **Range** 0.32, which is the band used for every decision. **Sample standard deviation** 0.16, which I computed from these three values.
- **How precise that is:** with three values, the true standard deviation could be about half to six times 0.16 (a 95% interval), which is roughly 0.08 to 1.0.
- **Rule:** a difference from the reference is noise when its absolute size is at most 0.32 F0.5. A candidate is chosen only if it beats the reference by more than 0.32. A smaller gain keeps the default and calls for a second seed before it's reported.
- **Caveat:** each candidate is compared with the seed-42 default run (55.61), the highest of the three seeds. Against the three-seed mean (55.45), every delta shifts by 0.16. Negative deltas get smaller and positive deltas get larger. For example, weight decay 0.1 is −0.36 against the seed-42 run but −0.20 against the mean. It's outside the band in one comparison and inside it in the other.

**Stage 1** (seeds 42, 123, 7; default learning rate 1e-4), dev_tune F0.5: mean 46.69, sample standard deviation 0.19, range 0.35.
Source: `t5_tuning/runs/t5-tune.s1.base.lr1e-4.seed42/`, `seed123/` and `seed7/`. The table is also in `t5_tuning/results-stage1.md`.

**Stage-2 re-run on the 3e-4 stage-1 model**, dev_tune F0.5: mean 54.84, sample standard deviation 0.21, range 0.38. Seed 42 scored 54.95.
Source: `t5_tuning/runs/t5-tune.s1lr3e-4.base.lr3e-5.seed42/`, `seed123/` and `seed7/`. The table is also in `t5_tuning/results-s1lr3e-4.md`.

## 5. Hyperparameters

Each subsection gives why the setting was varied, the values, the results and the chosen value. All learning-rate and regularization sweeps use seed 42, and everything else is at default.

### 5.1 Stage-2 learning rate

- **Why vary it:** the learning rate sets the size of each weight update. Stage 2 has 33,943 pairs and starts from a model that has already learned the task. A rate that's too high can overwrite that learning, and one that's too low can leave the model undertrained. [CITATION]
- **Values:** `{1e-5, 3e-5, 5e-5}`, with 3e-5 as the default in the middle. The grid isn't evenly spaced on a log scale: 1e-5 is a factor of 3 below the default, and 5e-5 is a factor of 1.7 above it.
- **Results** (dev_tune F0.5):

| Learning rate | F0.5 | Change vs reference (55.61) | Verdict |
|---|---|---|---|
| 1e-5 | 54.28 | −1.33 | outside band, worse |
| **3e-5** (default) | 55.61 | — | reference |
| 5e-5 | 55.72 | +0.11 | inside band |

Source: `t5_tuning/runs/t5-tune.base.lr1e-5.seed42/`, `t5-tune.base.lr3e-5.seed42/` and `t5-tune.base.lr5e-5.seed42/`, each `dev_tune.scores.json`.

- **Chosen:** 3e-5, the default. 5e-5 is inside the band, so there's no significant change and the default is kept. 5e-5 is the top of the grid, so the best rate may be higher. The gain is within noise, so the grid wasn't extended.

### 5.2 Stage-2 weight decay

- **Why vary it:** weight decay pulls each weight slightly toward zero on every update, which can limit overfitting. [CITATION]
- **Values:** `{0, 0.01, 0.1}`, with 0.01 as the default. The grid includes no decay (0) and a tenfold larger value (0.1).
- **Results** (dev_tune F0.5):

| Weight decay | F0.5 | Change vs reference | Verdict |
|---|---|---|---|
| 0 | 55.13 | −0.48 | outside band, worse |
| **0.01** (default) | 55.61 | — | reference |
| 0.1 | 55.25 | −0.36 | outside band vs seed-42 reference; inside band vs the three-seed mean (−0.20) |

Source: `t5_tuning/runs/t5-tune.wd0.lr3e-5.seed42/`, `t5-tune.base.lr3e-5.seed42/` and `t5-tune.wd0.1.lr3e-5.seed42/`, each `dev_tune.scores.json`.

- **Chosen:** 0.01, the default. Neither alternative beats it. The 0.1 result is borderline depending on the reference, and it's lower in both comparisons.

### 5.3 Stage-2 dropout

- **Why vary it:** dropout randomly switches off part of the network during training, which regularizes the model. Too much dropout slows learning. [CITATION]
- **Values:** `{0, 0.1, 0.3}`, with 0.1 as the default.
- **Results** (dev_tune F0.5):

| Dropout | F0.5 | Change vs reference | Verdict |
|---|---|---|---|
| 0 | 54.56 | −1.05 | outside band, worse |
| **0.1** (default) | 55.61 | — | reference |
| 0.3 | 50.73 | −4.88 | outside band, much worse |

Source: `t5_tuning/runs/t5-tune.dropout0.lr3e-5.seed42/`, `t5-tune.base.lr3e-5.seed42/` and `t5-tune.dropout0.3.lr3e-5.seed42/`, each `dev_tune.scores.json`.

- **Chosen:** 0.1, the default.
- **Observation (not a result):** the dropout 0.3 run reached its best score at the last evaluation (step 5,305). Its final recall was 0.29, against 0.41 for the default run with seed 123. Both recall values are from the `[errant]` lines in the Slurm logs on the cluster: `logs/slurm/t5-tune.dropout0.3.lr3e-5.seed42-1079920.out` and `logs/slurm/t5-tune.base.lr3e-5.seed123-1076161.out`. The five-epoch cap and early stopping were held fixed, so a longer run might change this. That wasn't tested.

### 5.4 Stage-2 label smoothing

- **Why vary it:** label smoothing replaces the single correct target with a softer distribution, so the model isn't trained to be fully certain. [CITATION]
- **Values:** `{0, 0.1, 0.2}`, with 0.1 as the default.
- **Results** (dev_tune F0.5):

| Label smoothing | F0.5 | Change vs reference | Verdict |
|---|---|---|---|
| 0 | 54.87 | −0.74 | outside band, worse |
| **0.1** (default) | 55.61 | — | reference |
| 0.2 | 55.03 | −0.58 | outside band, worse |

Source: `t5_tuning/runs/t5-tune.ls0.lr3e-5.seed42/`, `t5-tune.base.lr3e-5.seed42/` and `t5-tune.ls0.2.lr3e-5.seed42/`, each `dev_tune.scores.json`.

- **Chosen:** 0.1, the default.

### 5.5 Stage-1 learning rate

- **Why vary it:** the hypothesis was that stage 1, which starts from the base model, could take a higher rate without losing anything it had learned, and that the higher rate would improve the final result. The default is 1e-4.
- **Values:** `{3e-5, 1e-4, 3e-4}`, with 1e-4 as the default. That's a factor of 3 on each side. The sweep used seed 42. The noise band is from three seeds at 1e-4.
- **Results** (dev_tune F0.5, seed 42):

| Learning rate | F0.5 | Change vs reference (46.82) | Verdict |
|---|---|---|---|
| 3e-5 | 46.03 | −0.79 | outside band, worse |
| **1e-4** (default) | 46.82 | — | reference |
| 3e-4 | 47.93 | +1.11 | outside band, better |

Source: `t5_tuning/runs/t5-tune.s1.base.lr3e-5.seed42/`, `t5-tune.s1.base.lr1e-4.seed42/` and `t5-tune.s1.base.lr3e-4.seed42/`, each `dev_tune.scores.json`.

- **Effect on stage 2** (the test that matters for the pipeline). The higher stage-1 score didn't carry through. Stage 2 was rerun from the 3e-4 stage-1 model with the default settings, and it scored lower.

  | Stage-2 learning rate (seed 42, on the 3e-4 stage-1 model) | F0.5 | Change vs reference (54.95) | Verdict |
  |---|---|---|---|
  | 1e-5 | 53.97 | −0.98 | outside band, worse |
  | **3e-5** (default) | 54.95 | — | reference |
  | 5e-5 | 55.26 | +0.31 | inside band |

  Source: `t5_tuning/runs/t5-tune.s1lr3e-4.base.lr1e-5.seed42/`, `t5-tune.s1lr3e-4.base.lr3e-5.seed42/` and `t5-tune.s1lr3e-4.base.lr5e-5.seed42/`, each `dev_tune.scores.json`.

  - Noise band from three seeds at the default: mean 54.84, sample standard deviation 0.21, range 0.38. Source: section 4.
  - The stage-2 learning rate stays at 3e-5. 5e-5 is inside the band, so it isn't a reported gain.
  - Against the original pipeline, the mean fell from 55.45 to 54.84, a drop of 0.61. The seed-42 run fell from 55.61 to 54.95, a drop of 0.66. Both are larger than either band, so the drop isn't noise.
- **Chosen for the final pipeline:** none. The group's stage-1 model at 1e-4 is kept. On stage 1 alone, 3e-4 is the best value in the grid. It's the top of the grid, so the best rate may be higher, and 1e-3 **has not been run**.
- **Note:** the 3e-4 gain is from a single seed. It's larger than the stage-1 band of 0.35, but that band was measured at 1e-4, not at 3e-4.

The final pipeline is the original one. dev_test was not run for the 3e-4 pipeline.

**Not recorded:** whether dev_test was looked at before the final scoring. Confirm this before the write-up.

## 8. Final result on dev_test

The final configuration (the default settings, on the frozen stage-1 model) was scored once per seed on dev_test with `tune.py test`. The values are correction-level ERRANT percentages.

| Seed | Precision | Recall | F0.5 |
|---|---|---|---|
| 42 | 58.07 | 37.92 | 52.49 |
| 123 | 57.94 | 39.31 | 52.92 |
| 7 | 57.00 | 38.99 | 52.18 |
| **Mean** | 57.67 | 38.74 | **52.53** |

Source: `t5_tuning/runs/final-dev_test/seed42/`, `seed123/` and `seed7/`, each `dev_test.scores.json`.

For comparison, the group's shared model `CS4248-GEC/gec-t5-base-s1s2` scores 52.50 on dev_test (`experiments/training/t5-base.s1s2/dev_test.scores.json`). That's within the seed spread above.

## 6. Where the code is

| Step | File | What it does |
|---|---|---|
| Fetch and check the stage-1 model | `t5_tuning/fetch_stage1.sh` | Downloads `CS4248-GEC/gec-t5-base-s1` and checks its sha256 against `experiments/training/checkpoints.sha256` |
| Plan runs and write configs | `t5_tuning/tune.py plan` | Writes one JSON config per run into `t5_tuning/configs/`, and updates `manifest.json` and `manifest.md` |
| Submit runs | `t5_tuning/tune.py submit` | Submits the preflight as a CPU job, then each unscored run with `sbatch cluster/job.sbatch t5_tuning/run_one.sh <config>`, depending on the preflight passing |
| Check before any GPU run | `t5_tuning/preflight.sh`, `t5_tuning/tune.py preflight` | CPU job. Runs `training.check_data` and the tests, then checks every config against `manifest.json`, the torch and transformers versions against the recorded runs (`experiments/training/t5-base.s1s2/run_info.json`), that each starting model is cached, and the stage-1 model's sha256. Writes `preflight.json`. |
| Train and score one run | `t5_tuning/run_one.sh` | Calls `training/finetune.py` for training, then `training/evaluate.py --split dev_tune` for scoring, then `tune.py verify` on the run. It doesn't score dev_test. |
| Check a finished run | `t5_tuning/tune.py verify` | Compares the run with its config: the settings it trained with, the training-pair count and data manifest hash, the software versions, 2,000 dev_tune predictions, and the stability check below. Writes `runs/<run>/verified.json`; exits non-zero on any failure. |
| Stability check | Part of `tune.py verify`, on `trainer_state.json` | Fails on a non-finite loss or gradient norm. Warns on a loss spike above twice the lowest loss so far, after the first tenth of the run, or a loss that doesn't fall. Runs on every run. |
| Collect and judge results | `t5_tuning/tune.py collect` | Reads `t5_tuning/runs/<run>/dev_tune.scores.json`, computes the band, applies the rule in section 4, and writes `results.md`, `results-stage1.md` or `results-<tag>.md` |
| Final dev_test scoring | `t5_tuning/tune.py test`, `t5_tuning/run_test.sh` | Submits dev_test scoring for the final configuration's three seed runs. Only runs that passed `verify` are scored, and a seed that already has dev_test scores is skipped, so dev_test is scored once. |
| Tests | `tests/test_t5_tuning.py` | Every committed config matches its manifest row; the run and stability checks catch changed settings, versions, short predictions, unfinished runs, non-finite losses and spikes |

Earlier helper scripts (`band.py`, `lr_sweep.py`, `submit_band.sh`) were deleted and replaced by `tune.py`.

Results files: `t5_tuning/results.md` (stage 2, untagged), `t5_tuning/results-stage1.md` (stage 1), and `t5_tuning/results-<tag>.md` for tagged runs, such as `results-s1lr3e-4.md` for the stage-2 re-run on the 3e-4 stage-1 model.

`manifest.json`, `manifest.md` and `configs/` cover all 21 runs. They were rebuilt with the `plan` commands in section 7, which reproduce the five previously committed configs byte for byte. `collect` on the committed scores reproduces every number in sections 4 and 5.

## 7. How to reproduce

Run these from the repo root. Steps 1 to 3 are one-time setup.

**Without the cluster**, the results tables can be rebuilt from the committed scores:

```bash
python3 t5_tuning/tune.py collect                   # stage 2: band 55.45 / 0.16 / 0.32 and each sweep
python3 t5_tuning/tune.py collect --stage 1         # stage 1: band 46.69 / 0.19 / 0.35
python3 t5_tuning/tune.py collect --tag s1lr3e-4    # stage 2 on the 3e-4 stage-1 model: band 54.84 / 0.21 / 0.38
python3 -m unittest tests.test_t5_tuning
```

**On the cluster**, each `submit --run` below first submits a CPU preflight job, `t5-tune-preflight`, and every training run waits for it to pass (`afterok`). If the preflight fails, the training runs are cancelled before they start; its log in `logs/slurm/` names each failed check. Each training run then verifies itself at the end (`tune.py verify`); a failed check fails the job, and `runs/<run>/verified.json` lists what failed. Without `--run`, `submit` and `test` only print the `sbatch` commands.

1. **Build and copy the stage-1 data** (laptop, then cluster). `python3 -m training.build_data` builds `data/curriculum/stage1.*`. It needs `git lfs fetch origin 3e50997` first; see the "From scratch" and "Data" sections of `training/README.md`. Copy the three files to `data/curriculum/` on the cluster with `rsync`. The stage-2 data and dev splits are committed.
2. **Cache the base models** (compute node): `bash cluster/download_models.sh`. Stage 1 needs `google-t5/t5-base`, and jobs run offline.
3. **Cache the frozen stage-1 model** (compute node): `hf auth login` first, with a token from an account in the `CS4248-GEC` organization (see `experiments/training/README.md`), then `bash t5_tuning/fetch_stage1.sh`. It must print "sha256 matches".
4. **Noise band and stage-2 learning-rate sweep:**
   ```bash
   python3 t5_tuning/tune.py plan --phase 1
   python3 t5_tuning/tune.py submit --phase 1 --run
   python3 t5_tuning/tune.py collect
   ```
5. **Stage-2 regularization sweeps** at the chosen learning rate (3e-5 here):
   ```bash
   python3 t5_tuning/tune.py plan --phase 2 --lr 3e-5
   python3 t5_tuning/tune.py submit --phase 2 --run
   python3 t5_tuning/tune.py collect
   ```
6. **Stage-1 learning-rate sweep:**
   ```bash
   python3 t5_tuning/tune.py plan --stage 1 --phase 1
   python3 t5_tuning/tune.py submit --stage 1 --phase 1 --run
   python3 t5_tuning/tune.py collect --stage 1
   ```
7. **Stage-2 on a different stage-1 model:**
   ```bash
   python3 t5_tuning/tune.py plan --phase 1 --tag s1lr3e-4 --init '${CKPT_ROOT}/t5-tune.s1.base.lr3e-4.seed42/best'
   python3 t5_tuning/tune.py submit --phase 1 --tag s1lr3e-4 --run
   python3 t5_tuning/tune.py collect --tag s1lr3e-4
   ```
8. **dev_test** for the final configuration, once per seed, after its three seed runs have passed `verify`:
   ```bash
   python3 t5_tuning/tune.py test --run                # stage 2, default settings at lr 3e-5
   ```
   For another stage, tag or learning rate, add `--stage`, `--tag` or `--lr`. Scores go to `runs/final-dev_test/seed<seed>/` (with `-s1` or `-<tag>` for other stages and tags). `run_test.sh` refuses a folder that already has dev_test scores and checks there are 2,384 predictions.

To check runs that finished before these checks existed, run `python3 t5_tuning/tune.py verify --stage 2` on a compute node with `CKPT_ROOT` set (as `cluster/job.sbatch` does). It needs the runs' `run_info.json` and `trainer_state.json` from scratch, and their `dev_tune.pred` in `t5_tuning/runs/`.
