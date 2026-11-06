# T5 tuning methodology

This records how the T5 hyperparameters were tuned for the GEC project. Every number comes from the run logs and results tables in `t5_tuning/` and `experiments/`. Anything that wasn't run or wasn't recorded is marked as such.

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

## 2. Data and models

- **Stage 1** trains on the BEA-2019 corpora, errorful sentences only. It has 561,338 pairs after cleaning, and it starts from the base model.
- **Stage 2** trains on the W&I+LOCNESS training data, including correct sentences. It has 33,943 pairs, and it starts from the frozen stage-1 model.
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

- **Mean** 55.45. **Range** 0.32, which is the band used for every decision. **Sample standard deviation** 0.16, which I computed from these three values.
- **How precise that is:** with three values, the true standard deviation could be about half to six times 0.16 (a 95% interval), which is roughly 0.08 to 1.0.
- **Rule:** a difference from the reference is noise when its absolute size is at most 0.32 F0.5. A candidate is chosen only if it beats the reference by more than 0.32. A smaller gain keeps the default and calls for a second seed before it's reported.
- **Caveat:** each candidate is compared with the seed-42 default run (55.61), the highest of the three seeds. Against the three-seed mean (55.45), every delta shifts by 0.16. Negative deltas get smaller and positive deltas get larger. For example, weight decay 0.1 is −0.36 against the seed-42 run but −0.20 against the mean. It's outside the band in one comparison and inside it in the other.

**Stage 1** (seeds 42, 123, 7; default learning rate 1e-4), dev_tune F0.5: mean 46.69, sample standard deviation 0.19, range 0.35. The values for seeds 123 and 7 are in `t5_tuning/results-stage1.md` on the cluster, not in this document.

**Stage-2 re-run on the 3e-4 stage-1 model**, dev_tune F0.5: mean 54.84, sample standard deviation 0.21, range 0.38. Seed 42 scored 54.95. The values for seeds 123 and 7 are in `t5_tuning/results-s1lr3e-4.md` on the cluster.

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

- **Chosen:** 0.1, the default.
- **Observation (not a result):** the dropout 0.3 run reached its best score at the last evaluation (step 5,305). Its final recall was 0.29, against 0.41 for the default run with seed 123. The five-epoch cap and early stopping were held fixed, so a longer run might change this. That wasn't tested.

### 5.4 Stage-2 label smoothing

- **Why vary it:** label smoothing replaces the single correct target with a softer distribution, so the model isn't trained to be fully certain. [CITATION]
- **Values:** `{0, 0.1, 0.2}`, with 0.1 as the default.
- **Results** (dev_tune F0.5):

| Label smoothing | F0.5 | Change vs reference | Verdict |
|---|---|---|---|
| 0 | 54.87 | −0.74 | outside band, worse |
| **0.1** (default) | 55.61 | — | reference |
| 0.2 | 55.03 | −0.58 | outside band, worse |

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

- **Effect on stage 2** (the test that matters for the pipeline). The higher stage-1 score didn't carry through. Stage 2 was rerun from the 3e-4 stage-1 model with the default settings, and it scored lower.

  | Stage-2 learning rate (seed 42, on the 3e-4 stage-1 model) | F0.5 | Change vs reference (54.95) | Verdict |
  |---|---|---|---|
  | 1e-5 | 53.97 | −0.98 | outside band, worse |
  | **3e-5** (default) | 54.95 | — | reference |
  | 5e-5 | 55.26 | +0.31 | inside band |

  - Noise band from three seeds at the default: mean 54.84, sample standard deviation 0.21, range 0.38.
  - The stage-2 learning rate stays at 3e-5. 5e-5 is inside the band, so it isn't a reported gain.
  - Against the original pipeline, the mean fell from 55.45 to 54.84, a drop of 0.61. The seed-42 run fell from 55.61 to 54.95, a drop of 0.66. Both are larger than either band, so the drop isn't noise.
- **Chosen for the final pipeline:** none. The group's stage-1 model at 1e-4 is kept. On stage 1 alone, 3e-4 is the best value in the grid. It's the top of the grid, so the best rate may be higher, and 1e-3 **has not been run**.
- **Note:** the 3e-4 gain is from a single seed. It's larger than the stage-1 band of 0.35, but that band was measured at 1e-4, not at 3e-4.

The final pipeline is the original one. dev_test was not run for the 3e-4 pipeline.

**Not recorded:** whether dev_test was looked at before the final scoring. Confirm this before the write-up.

## 6. Where the code is

| Step | File | What it does |
|---|---|---|
| Fetch and check the stage-1 model | `t5_tuning/fetch_stage1.sh` | Downloads `CS4248-GEC/gec-t5-base-s1` and checks its sha256 against `experiments/training/checkpoints.sha256` |
| Plan runs and write configs | `t5_tuning/tune.py plan` | Writes one JSON config per run into `t5_tuning/configs/`, and updates `manifest.json` and `manifest.md` |
| Submit runs | `t5_tuning/tune.py submit` | Calls `sbatch cluster/job.sbatch t5_tuning/run_one.sh <config>` |
| Train and score one run | `t5_tuning/run_one.sh` | Calls `training/finetune.py` for training, then `training/evaluate.py --split dev_tune` for scoring. It doesn't score dev_test. |
| Collect and judge results | `t5_tuning/tune.py collect` | Reads `t5_tuning/runs/<run>/dev_tune.scores.json`, computes the band, applies the rule in section 4, and writes `results.md`, `results-stage1.md` or `results-<tag>.md` |
| Final dev_test scoring | A direct call to `training/evaluate.py --split dev_test`, as in section 7 | Not wrapped in `tune.py`, so dev_test is read only when it's run deliberately |
| Stability check | A short Python check of `trainer_state.json` | Not in the repo. Run once, on the stage-1 3e-4 run. |

Earlier helper scripts (`band.py`, `lr_sweep.py`, `submit_band.sh`) were deleted and replaced by `tune.py`.

Results files: `t5_tuning/results.md` (stage 2, untagged), `t5_tuning/results-stage1.md` (stage 1), and `t5_tuning/results-<tag>.md` for tagged runs, such as `results-s1lr3e-4.md` for the stage-2 re-run on the 3e-4 stage-1 model.

## 7. How to reproduce

Run these from the repo root. Steps 1 to 3 are one-time setup.

1. **Build and copy the stage-1 data** (laptop, then cluster). `python3 -m training.build_data` builds `data/curriculum/stage1.*`. Copy the three files to `data/curriculum/` on the cluster with `rsync`. The stage-2 data and dev splits are committed.
2. **Cache the base models** (compute node): `bash cluster/download_models.sh`. Stage 1 needs `google-t5/t5-base`, and jobs run offline.
3. **Cache the frozen stage-1 model** (compute node): `bash t5_tuning/fetch_stage1.sh`. It must print "sha256 matches".
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
8. **dev_test** for a chosen configuration, once per seed:
   ```bash
   for s in 42 123 7; do
     python -m training.evaluate --model $CKPT_ROOT/<run>/best --split dev_test \
       --out t5_tuning/runs/final-dev_test/seed$s
   done
   ```
   Run this on a GPU node after `conda activate gec-train`, and set `CKPT_ROOT` to the scratch checkpoint folder.
