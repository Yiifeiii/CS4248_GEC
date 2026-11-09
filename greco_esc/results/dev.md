# Full dev set results (BEA-2019 dev)

All 4,384 sentences of BEA-2019 dev (`data/dev.src` = `dev_tune` + `dev_test`), scored with ERRANT F0.5, the BEA-2019 metric. M2 is also given. The outputs come from jobs 1098818, 1099227 and 1101538. The last one reran the six runs whose α changed when the full-dev α were adopted. Scores are from `python -m greco_esc.evaluate dev`, copied from `runs/dev/results.{csv,md}` into [`dev.csv`](dev.csv). Every run uses the same settings as on the test sets.

**This is the set the best dev model is chosen on: the run with the highest full-dev ERRANT F0.5, picked without looking at any test score.**

## Models to report

| Model | Run | Full dev (ERRANT) | BEA-2019 test (ERRANT) | CoNLL-2014 test (M2, official) |
|---|---|---|---|---|
| **Best dev model** | GRECO + ESC, k = 3, `rank` 0.4 | **60.35** (P 71.88 / R 36.76) | 77.24 | 66.14 |
| **Best BEA-2019 model** | GRECO + ESC, k = 2, `system` 0.1 | 60.22 | **77.45** | 66.13 |
| **Best CoNLL-2014 model** | GRECO alone, k = 2, `rank` 0.2 | 58.59 | 74.83 | **67.98** |
| Best single system | GECToR | 54.77 | 71.57 | 64.03 |

The best dev model's test scores are the unbiased, held-out result. The best BEA-2019 and CoNLL-2014 models were picked on their own test scores, so they show the best each setup reaches there, not held-out estimates.

**How ESC is kept off its own training sentences (cross-fitting).** ESC learns from gold edits, so no sentence is scored by an ESC trained on it. Each `dev_test` sentence (2,384) is scored with ESC trained on `dev_tune` (2,000), and each `dev_tune` sentence with ESC trained on `dev_test`. The two halves of each output are then merged back into `dev.src` order and scored together.

**α in this table.** The α come from tuning on this same dev set without ESC (`main`, `outputs/greco/tuning/dev.errant.beam/best.json`), so they are in-sample here: `hypothesis` 0.1 and `system` 0.1 at every k, and `rank` 0.1 / 0.2 / 0.4 / 0.6 at k = 1 / 2 / 3 / 5.

## Every run

k is the number of hypotheses taken from each of BART and T5. GECToR always gives its 1-best, so there are 1 + 2k inputs. At k = 1 the three vote modes give the same output, so only `hypothesis` is run.

| k | Inputs | System | ERRANT P | ERRANT R | ERRANT F0.5 | M2 P | M2 R | M2 F0.5 |
|---|---|---|---|---|---|---|---|---|
| - | 1 | bart-1 (single system) | 58.61 | 38.49 | 53.07 | 66.59 | 41.52 | 59.42 |
| - | 1 | t5-1 (single system) | 58.87 | 39.83 | 53.74 | 67.12 | 43.08 | 60.38 |
| - | 1 | gector-1 (single system) | 63.81 | 34.96 | 54.77 | 70.24 | 36.49 | 59.28 |
| 1 | 3 | ESC alone | 71.74 | 34.67 | 59.10 | 75.97 | 35.38 | 61.79 |
| 1 | 3 | GRECO, no vote | 66.80 | 40.61 | 59.17 | 74.36 | 43.20 | 64.98 |
| 1 | 3 | GRECO, `hypothesis` 0.1 | 67.84 | 40.68 | 59.85 | 74.76 | 43.12 | 65.19 |
| 1 | 3 | GRECO + ESC, no vote | 72.20 | 36.35 | 60.31 | 77.12 | 37.41 | 63.61 |
| 1 | 3 | GRECO + ESC, `hypothesis` 0.1 | 72.18 | 36.20 | 60.21 | 77.10 | 37.33 | 63.56 |
| 2 | 5 | ESC alone | 73.15 | 33.27 | 59.00 | 77.57 | 34.03 | 61.76 |
| 2 | 5 | GRECO, no vote | 62.42 | 43.63 | 57.47 | 69.83 | 46.51 | 63.47 |
| 2 | 5 | GRECO, `hypothesis` 0.1 | 63.20 | 43.53 | 57.96 | 70.02 | 46.20 | 63.47 |
| 2 | 5 | GRECO, `rank` 0.2 *(best CoNLL-2014)* | 64.64 | 42.65 | 58.59 | 71.11 | 45.09 | 63.75 |
| 2 | 5 | GRECO, `system` 0.1 | 63.37 | 44.16 | 58.30 | 70.26 | 46.83 | 63.87 |
| 2 | 5 | GRECO + ESC, no vote | 72.93 | 35.21 | 60.06 | 78.03 | 36.48 | 63.55 |
| 2 | 5 | GRECO + ESC, `hypothesis` 0.1 | 72.34 | 35.69 | 60.02 | 77.51 | 37.07 | 63.62 |
| 2 | 5 | GRECO + ESC, `rank` 0.2 | 72.43 | 35.85 | 60.16 | 77.56 | 37.26 | 63.77 |
| 2 | 5 | GRECO + ESC, `system` 0.1 *(best BEA-2019)* | 72.72 | 35.69 | 60.22 | 77.78 | 36.98 | 63.72 |
| 3 | 7 | ESC alone | 72.95 | 32.97 | 58.71 | 77.62 | 33.95 | 61.74 |
| 3 | 7 | GRECO, no vote | 60.00 | 45.03 | 56.26 | 67.61 | 48.32 | 62.61 |
| 3 | 7 | GRECO, `hypothesis` 0.1 | 60.79 | 45.03 | 56.82 | 67.92 | 48.20 | 62.78 |
| 3 | 7 | GRECO, `rank` 0.4 | 63.98 | 42.33 | 58.04 | 70.54 | 44.68 | 63.22 |
| 3 | 7 | GRECO, `system` 0.1 | 61.00 | 45.87 | 57.22 | 68.00 | 48.81 | 63.04 |
| 3 | 7 | GRECO + ESC, no vote | 72.97 | 35.46 | 60.23 | 78.04 | 36.83 | 63.77 |
| 3 | 7 | GRECO + ESC, `hypothesis` 0.1 | 72.09 | 36.24 | 60.18 | 77.24 | 37.71 | 63.85 |
| **3** | **7** | **GRECO + ESC, `rank` 0.4 (best dev model)** | **71.88** | **36.76** | **60.35** | **76.90** | **38.22** | **63.96** |
| 3 | 7 | GRECO + ESC, `system` 0.1 | 72.56 | 36.00 | 60.31 | 77.55 | 37.34 | 63.81 |
| 5 | 11 | ESC alone | 72.81 | 32.98 | 58.65 | 77.67 | 34.04 | 61.82 |
| 5 | 11 | GRECO, no vote | 56.23 | 45.88 | 53.81 | 64.37 | 49.57 | 60.74 |
| 5 | 11 | GRECO, `hypothesis` 0.1 | 57.37 | 46.45 | 54.80 | 64.99 | 49.80 | 61.25 |
| 5 | 11 | GRECO, `rank` 0.6 | 63.25 | 42.17 | 57.50 | 69.91 | 44.51 | 62.75 |
| 5 | 11 | GRECO, `system` 0.1 | 57.14 | 46.92 | 54.76 | 64.86 | 50.33 | 61.32 |
| 5 | 11 | GRECO + ESC, no vote | 71.97 | 36.00 | 59.98 | 77.27 | 37.54 | 63.77 |
| 5 | 11 | GRECO + ESC, `hypothesis` 0.1 | 70.39 | 37.43 | 59.85 | 75.62 | 39.10 | 63.72 |
| 5 | 11 | GRECO + ESC, `rank` 0.6 | 69.76 | 37.64 | 59.59 | 74.95 | 39.32 | 63.45 |
| 5 | 11 | GRECO + ESC, `system` 0.1 | 70.96 | 36.74 | 59.81 | 76.12 | 38.30 | 63.57 |

## The α rerun (job 1101538)

The six runs whose α changed, before (earlier `dev_tune`-only α) and after (full-dev α). Every other run reproduced its earlier score exactly.

| Run | Old α → new α | Old score | New score | Change |
|---|---|---|---|---|
| GRECO, k1, vote | 0.4 → 0.1 | 59.86 | 59.85 | −0.01 |
| GRECO + ESC, k1, vote | 0.4 → 0.1 | 59.73 | 60.21 | +0.48 |
| GRECO, k2, `rank` | 0.4 → 0.2 | 58.54 | 58.59 | +0.05 |
| GRECO + ESC, k2, `rank` | 0.4 → 0.2 | 59.99 | 60.16 | +0.17 |
| GRECO, k5, `rank` | 0.4 → 0.6 | 57.30 | 57.50 | +0.20 |
| GRECO + ESC, k5, `rank` | 0.4 → 0.6 | 59.84 | 59.59 | −0.25 |

None of them overtook the best dev model, which has the same α (0.4) under both tunings.

## Findings on dev

1. **The best dev model is GRECO + ESC, k = 3, `rank` 0.4, at 60.35** (P 71.88 / R 36.76), 5.6 points above GECToR (54.77).
2. **GRECO + ESC is flat across k**, at 59.59–60.35 for all 14 GRECO + ESC rows. GRECO alone falls as k grows, from 59.85 at k = 1 to 53.81–57.50 at k = 5. More hypotheses give GRECO more wrong edits to choose from, and ESC filters them out.
3. **ESC's gain grows with k**: +0.4 to +1.1 at k = 1, +1.6 to +2.6 at k = 2, +2.3 to +4.0 at k = 3 and +2.1 to +6.2 at k = 5.
4. **All GRECO + ESC rows beat ESC alone** (58.65–59.10) and every single system (BART 53.07, T5 53.74, GECToR 54.77).
5. **On M2, ESC does not help on dev either.** The best M2 score is GRECO alone, k = 1, vote 0.1 (65.19), against 63.96 for the best dev model. This agrees with the CoNLL-2014 M2 results.
6. **The top runs are within noise of each other.** The best dev model is the highest, but five other GRECO + ESC rows are within 0.15 of it: k = 1 no vote (60.31), k = 3 `system` (60.31), k = 3 no vote (60.23), k = 2 `system` (60.22) and k = 1 vote 0.1 (60.21).

## Dev against the test sets

Every run, on all three sets. BEA-2019 test is ERRANT F0.5 from Codabench. For CoNLL-2014, M2 is official, and ERRANT is given for comparison with dev.

| Run | Full dev (ERRANT) | BEA-2019 test (ERRANT) | CoNLL-2014 (M2) | CoNLL-2014 (ERRANT) |
|---|---|---|---|---|
| BART (single system) | 53.07 | 69.24 | 63.23 | 58.05 |
| T5 (single system) | 53.74 | 69.91 | 63.12 | 57.89 |
| GECToR (single system) | 54.77 | 71.57 | 64.03 | 59.94 |
| ESC alone, k1 | 59.10 | 76.12 | 64.02 | 62.13 |
| ESC alone, k2 | 59.00 | 76.33 | 63.78 | 61.85 |
| ESC alone, k3 | 58.71 | 75.88 | 62.48 | 60.69 |
| ESC alone, k5 | 58.65 | 75.57 | 63.38 | 61.41 |
| GRECO, k1, no vote | 59.17 | 76.21 | 67.70 | 63.43 |
| GRECO, k1, vote 0.1 | 59.85 | 76.82 | 67.76 | 63.78 |
| GRECO + ESC, k1, no vote | 60.31 | 77.17 | 65.43 | 63.73 |
| GRECO + ESC, k1, vote 0.1 | 60.21 | 77.16 | 65.75 | 63.85 |
| GRECO, k2, no vote | 57.47 | 72.84 | 66.75 | 62.48 |
| GRECO, k2, `hypothesis` 0.1 | 57.96 | 73.88 | 67.60 | 63.92 |
| GRECO, k2, `system` 0.1 | 58.30 | 73.95 | 67.62 | 63.61 |
| **GRECO, k2, `rank` 0.2 (best CoNLL-2014)** | 58.59 | 74.83 | **67.98** | 64.02 |
| GRECO + ESC, k2, no vote | 60.06 | 77.11 | 65.73 | 63.64 |
| GRECO + ESC, k2, `hypothesis` 0.1 | 60.02 | 77.33 | 66.06 | 63.87 |
| **GRECO + ESC, k2, `system` 0.1 (best BEA-2019)** | 60.22 | **77.45** | 66.13 | 63.90 |
| GRECO + ESC, k2, `rank` 0.2 | 60.16 | 77.28 | 66.33 | 64.16 |
| GRECO, k3, no vote | 56.26 | 70.22 | 65.78 | 61.01 |
| GRECO, k3, `hypothesis` 0.1 | 56.82 | 71.92 | 66.91 | 62.74 |
| GRECO, k3, `system` 0.1 | 57.22 | 71.79 | 66.10 | 61.77 |
| GRECO, k3, `rank` 0.4 | 58.04 | 74.56 | 67.52 | 63.60 |
| GRECO + ESC, k3, no vote | 60.23 | 76.77 | 65.26 | 63.51 |
| GRECO + ESC, k3, `hypothesis` 0.1 | 60.18 | 76.89 | 65.48 | 63.62 |
| GRECO + ESC, k3, `system` 0.1 | 60.31 | 77.09 | 65.38 | 63.63 |
| **GRECO + ESC, k3, `rank` 0.4 (best dev model)** | **60.35** | 77.24 | 66.14 | 64.47 |
| GRECO, k5, no vote | 53.81 | 67.26 | 64.90 | 59.50 |
| GRECO, k5, `hypothesis` 0.1 | 54.80 | 69.58 | 65.81 | 61.04 |
| GRECO, k5, `system` 0.1 | 54.76 | 68.97 | 65.01 | 60.37 |
| GRECO, k5, `rank` 0.6 | 57.50 | pending | 67.28 | 63.35 |
| GRECO + ESC, k5, no vote | 59.98 | 76.74 | 65.88 | 63.96 |
| GRECO + ESC, k5, `hypothesis` 0.1 | 59.85 | 76.80 | 66.94 | 64.64 |
| GRECO + ESC, k5, `system` 0.1 | 59.81 | pending | 66.96 | 64.84 |
| GRECO + ESC, k5, `rank` 0.6 | 59.59 | pending | 66.91 | 64.74 |

**What carries over from dev to BEA-2019 test.** The patterns all carry over:

- GRECO alone falls as k grows.
- GRECO + ESC stays flat and above GRECO alone at every k.
- `rank` voting helps GRECO alone most.
- Every GRECO + ESC run beats ESC alone and the single systems.

Only the order of the near-tied top runs changes. k = 2 `system` 0.1 was 0.13 behind the best dev model on dev and is 0.21 ahead of it on test.

**What carries over to CoNLL-2014.** On ERRANT the pattern partly holds: ESC helps in 13 of 14 pairs. On M2, the official metric, GRECO alone is best and ESC lowers the score at k ≤ 3. Dev M2 showed the same (finding 5).

## Why dev and test results can differ

None of this means a run was set up wrongly. The unchanged runs reproduced their dev scores exactly in the rerun.

1. **Different sentences and references.** Dev and BEA-2019 test are both W&I+LOCNESS, but they come from different writers. The test references are hidden on Codabench. CoNLL-2014 is a different corpus (NUCLE essays) with different annotators, guidelines and an M2 metric. Absolute scores are not comparable across sets: every BEA-2019 test row is 13.5–17.5 points above its dev row, single systems, ESC alone and GRECO alone included, so the gap comes from the data and its references, not from the combination. Compare how runs rank within a set, not raw numbers across sets.
2. **The choices were made on dev, so dev flatters them.** α was tuned on the full dev set, and the best dev model was the highest of 28 combinations on it. The best of many close runs on a set is usually a little lucky on that set (the "winner's curse"), so its dev score overstates what it gets elsewhere. The test sets are the unbiased check.
3. **Differences between the top runs are within noise.** The leading GRECO + ESC rows differ by 0.1–0.3 on both dev and test. On a few thousand sentences, F0.5 moves by more than that when the sentences are resampled, so the order among them can flip between sets, as k = 2 and k = 3 did.
4. **ESC was trained on more data for the test sets.** For dev scores (cross-fitting), each ESC learned from one half, 2,000 or 2,384 sentences. For the test sets, ESC learned from all 4,384. A better-trained ESC is one plausible reason ESC's gain on BEA-2019 test is larger than on dev at k ≥ 2 (k = 3 no vote: +3.97 on dev, +6.55 on test).
5. **ESC is fitted to BEA's annotation and ERRANT's edits.** That fits BEA-2019 test, which shares both. It does not fit CoNLL-2014's annotation or the M2 scorer's edit matching, so ESC's effect on CoNLL-2014 M2 differs from dev ERRANT (see [`conll14.md`](conll14.md)).
