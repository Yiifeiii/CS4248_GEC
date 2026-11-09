# BEA-2019 test results (Codabench)

ERRANT span-correction F0.5 on the 4,477 BEA-2019 test sentences, as reported by the BEA-2019 Codabench task. Codabench shows only F0.5, so precision and recall are not given. Each row is one submission of `runs/bea19/k<k>/<run>/pred.txt` (or `runs/bea19/text/<system>-1.txt` for the single systems), from test jobs 1099945 and 1100522. The second job reran the runs whose α changed when the full-dev α were adopted. All were submitted on 9 Nov 2026.

Nothing was tuned or trained on the test set. ESC was trained once per k on the full dev set (BEA-2019 dev, 4,384 sentences). The vote coefficients α are the ones tuned on the full dev set (`settings.VOTES`).

## Models to report

| Model | Run | BEA-2019 test | How it was picked |
|---|---|---|---|
| **Best BEA-2019 model (so far)** | GRECO + ESC, k = 2, `system` 0.1 | **77.45** | highest test score; two GRECO + ESC k = 5 runs are still pending |
| **Best dev model** | GRECO + ESC, k = 3, `rank` 0.4 | **77.24** | highest full-dev ERRANT F0.5 (60.35), chosen without seeing test scores |
| Best GRECO alone | GRECO, k = 1, vote 0.1 | 76.82 | highest test score without ESC |
| Best ESC alone | ESC, k = 2 | 76.33 | |
| Best single system | GECToR | 71.57 | |

The best BEA-2019 model was picked by looking at test scores, so 77.45 shows the best this setup reaches, not a held-out estimate. The best dev model, chosen without seeing any test score, is the unbiased one.

## Every submission

k is the number of hypotheses taken from each of BART and T5. GECToR always contributes its 1-best, so GRECO combines 1 + 2k inputs. At k = 1 the three vote modes give the same output.

| k | Inputs | System | Voting | BEA-2019 test F0.5 |
|---|---|---|---|---|
| – | 1 | BART (`bart-1`) | – | 69.24 |
| – | 1 | T5 (`t5-1`) | – | 69.91 |
| – | 1 | GECToR (`gector-1`) | – | 71.57 |
| 1 | 3 | ESC alone | – | 76.12 |
| 1 | 3 | GRECO | no vote | 76.21 |
| 1 | 3 | GRECO | vote 0.1 | 76.82 *(best GRECO alone)* |
| 1 | 3 | GRECO + ESC | no vote | 77.17 |
| 1 | 3 | GRECO + ESC | vote 0.1 | 77.16 |
| 2 | 5 | ESC alone | – | 76.33 |
| 2 | 5 | GRECO | no vote | 72.84 |
| 2 | 5 | GRECO | `hypothesis` 0.1 | 73.88 |
| 2 | 5 | GRECO | `system` 0.1 | 73.95 |
| 2 | 5 | GRECO | `rank` 0.2 | 74.83 |
| 2 | 5 | GRECO + ESC | no vote | 77.11 |
| 2 | 5 | GRECO + ESC | `hypothesis` 0.1 | 77.33 |
| **2** | **5** | **GRECO + ESC** | **`system` 0.1** | **77.45** *(best BEA-2019 so far)* |
| 2 | 5 | GRECO + ESC | `rank` 0.2 | 77.28 |
| 3 | 7 | ESC alone | – | 75.88 |
| 3 | 7 | GRECO | no vote | 70.22 |
| 3 | 7 | GRECO | `hypothesis` 0.1 | 71.92 |
| 3 | 7 | GRECO | `system` 0.1 | 71.79 |
| 3 | 7 | GRECO | `rank` 0.4 | 74.56 |
| 3 | 7 | GRECO + ESC | no vote | 76.77 |
| 3 | 7 | GRECO + ESC | `hypothesis` 0.1 | 76.89 |
| 3 | 7 | GRECO + ESC | `system` 0.1 | 77.09 |
| 3 | 7 | GRECO + ESC | `rank` 0.4 | **77.24** *(best dev model)* |
| 5 | 11 | ESC alone | – | 75.57 |
| 5 | 11 | GRECO | no vote | 67.26 |
| 5 | 11 | GRECO | `hypothesis` 0.1 | 69.58 |
| 5 | 11 | GRECO | `system` 0.1 | 68.97 |
| 5 | 11 | GRECO | `rank` 0.6 | pending |
| 5 | 11 | GRECO + ESC | no vote | 76.74 |
| 5 | 11 | GRECO + ESC | `hypothesis` 0.1 | 76.80 |
| 5 | 11 | GRECO + ESC | `system` 0.1 | pending |
| 5 | 11 | GRECO + ESC | `rank` 0.6 | pending |

## GRECO alone vs GRECO + ESC

| k | Voting | GRECO alone | GRECO + ESC | ESC's effect |
|---|---|---|---|---|
| 1 | no vote | 76.21 | 77.17 | +0.96 |
| 1 | vote 0.1 | 76.82 | 77.16 | +0.34 |
| 2 | no vote | 72.84 | 77.11 | +4.27 |
| 2 | `hypothesis` 0.1 | 73.88 | 77.33 | +3.45 |
| 2 | `system` 0.1 | 73.95 | 77.45 | +3.50 |
| 2 | `rank` 0.2 | 74.83 | 77.28 | +2.45 |
| 3 | no vote | 70.22 | 76.77 | +6.55 |
| 3 | `hypothesis` 0.1 | 71.92 | 76.89 | +4.97 |
| 3 | `system` 0.1 | 71.79 | 77.09 | +5.30 |
| 3 | `rank` 0.4 | 74.56 | 77.24 | +2.68 |
| 5 | no vote | 67.26 | 76.74 | +9.48 |
| 5 | `hypothesis` 0.1 | 69.58 | 76.80 | +7.22 |
| 5 | `system` 0.1 | 68.97 | pending | |
| 5 | `rank` 0.6 | pending | pending | |

## Findings

1. **Every GRECO + ESC run beats every single system by more than 5 points.** All twelve scored so far are between 76.74 and 77.45, against 71.57 for GECToR, 69.91 for T5 and 69.24 for BART. Even ESC alone (75.57–76.33) is 4.0–4.8 above GECToR.
2. **ESC is what makes the combination work on BEA-2019.** At every k and with every voting setting, GRECO + ESC beats GRECO alone, by +0.3–1.0 at k = 1, +2.4–4.3 at k = 2, +2.7–6.6 at k = 3 and +7.2–9.5 at k = 5 (so far). The more hypotheses GRECO sees, the more ESC adds. Every GRECO + ESC run also beats ESC alone.
3. **GRECO alone gets worse as k grows, and drops below GECToR.** Its best score is 76.82 at k = 1, 74.83 at k = 2 and 74.56 at k = 3, and at most 69.58 at k = 5 so far. At k = 3 with no vote (70.22) and at every k = 5 run so far, GRECO alone is below GECToR on its own. More hypotheses give the beam search more wrong edits to choose from. With ESC, the score stays flat across k: 76.74–76.80 at k = 5 so far, against 77.16–77.17 at k = 1.
4. **`rank` voting is the best voting for GRECO alone at larger k.** At k = 3 it scores 74.56, against 70.22 without voting and about 71.9 for `hypothesis` and `system`. It trusts each system's 1-best more than its lower-ranked hypotheses. With ESC on, the vote modes are within 0.5 of each other.
5. **The top GRECO + ESC runs are within noise.** The best (k = 2 `system`, 77.45) and the best dev model (k = 3 `rank`, 77.24) are 0.21 apart. On dev the order was reversed by 0.13. See [`dev.md`](dev.md) for why dev and test can differ.
6. **The test set ranks the runs as the full dev set did.** On dev, GRECO + ESC was flat across k, GRECO alone fell with k, and rank voting helped most without ESC. All three hold here.

## Reproducing

```bash
sbatch -J greco-test --time=12:00:00 cluster/job.sbatch greco_esc/test.sh bea19
```

Then zip `runs/bea19/k<k>/<run>/pred.txt` (one `.txt` of 4,477 lines per zip) and upload it under **My Submissions** on the BEA-2019 Codabench task. The zips submitted here are in `codabench/` (not tracked).
