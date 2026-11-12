# BEA-2019 test results (Codabench)

ERRANT span-correction F0.5 on the 4,477 BEA-2019 test sentences, as reported by the BEA-2019 Codabench task. Codabench shows only F0.5, so precision and recall are not given. Each row is one submission of `runs/bea19/k<k>/<run>/pred.txt` (or `runs/bea19/text/<system>-1.txt` for the single systems), from test jobs 1099945 and 1100522. The second job reran the runs whose α changed when the full-dev α were adopted. All were submitted on 9 Nov 2026.

Nothing was tuned or trained on the test set. ESC was trained once per k on the full dev set (BEA-2019 dev, 4,384 sentences). The vote coefficients α are the ones tuned on the full dev set (`settings.VOTES`).

## Models to report

| Model | Run | BEA-2019 test | How it was picked |
|---|---|---|---|
| **Best BEA-2019 model** | GRECO + ESC, k = 2, `system` 0.1 | **77.45** | highest BEA-2019 test score of all 35 submissions |
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
| **2** | **5** | **GRECO + ESC** | **`system` 0.1** | **77.45** *(best BEA-2019)* |
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
| 5 | 11 | GRECO | `rank` 0.6 | 73.67 |
| 5 | 11 | GRECO + ESC | no vote | 76.74 |
| 5 | 11 | GRECO + ESC | `hypothesis` 0.1 | 76.80 |
| 5 | 11 | GRECO + ESC | `system` 0.1 | 76.85 |
| 5 | 11 | GRECO + ESC | `rank` 0.6 | 76.73 |

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
| 5 | `system` 0.1 | 68.97 | 76.85 | +7.88 |
| 5 | `rank` 0.6 | 73.67 | 76.73 | +3.06 |

## Findings

1. **Every GRECO + ESC run beats every single system by more than 5 points.** All 14 are between 76.73 and 77.45, against 71.57 for GECToR, 69.91 for T5 and 69.24 for BART. Even ESC alone (75.57–76.33) is 4.0–4.8 above GECToR.
2. **ESC is what makes the combination work on BEA-2019.** At every k and with every voting setting, GRECO + ESC beats GRECO alone, by +0.3–1.0 at k = 1, +2.4–4.3 at k = 2, +2.7–6.6 at k = 3 and +3.1–9.5 at k = 5. The more hypotheses GRECO sees, the more ESC adds. Every GRECO + ESC run also beats ESC alone.
   **It does this by trading recall for precision**, as on CoNLL-2014. Codabench reports only F0.5, so this test set has no P / R, but on BEA-2019 dev ([`dev.md`](dev.md)) ESC raises precision by 4.3–15.7 points and lowers recall by 4.3–10.2 against GRECO alone with the same voting. On BEA-2019 the precision gain outweighs the recall loss; on CoNLL-2014's M2 it does not ([`conll14.md`](conll14.md)).
3. **GRECO alone gets worse as k grows.** Its best score is 76.82 at k = 1, 74.83 at k = 2, 74.56 at k = 3 and 73.67 at k = 5. Without `rank` voting it drops below GECToR (71.57): k = 3 with no vote (70.22) and the three other k = 5 runs (67.26–69.58). More hypotheses give the beam search more wrong edits to choose from. With ESC, the score stays flat across k: 76.73–76.85 at k = 5, against 77.16–77.17 at k = 1.
4. **`rank` voting is the best voting for GRECO alone at larger k.** At k = 3 it scores 74.56, against 70.22 without voting and about 71.9 for `hypothesis` and `system`. At k = 5 it scores 73.67, against 67.26–69.58. It trusts each system's 1-best more than its lower-ranked hypotheses. With ESC on, the vote modes are within 0.5 of each other.
5. **The top GRECO + ESC runs are close.** The best (k = 2 `system`, 77.45) and the best dev model (k = 3 `rank`, 77.24) are 0.21 apart, and on dev the order was reversed by 0.13. See [Why k = 2 beats k = 3](#why-k--2-beats-k--3-on-bea-2019-test) below.
6. **The test set ranks the runs as the full dev set did.** On dev, GRECO + ESC was flat across k, GRECO alone fell with k, and rank voting helped most without ESC. All three hold here.

## Why k = 2 beats k = 3 on BEA-2019 test

The best dev model (k = 3, `rank` 0.4) and the best BEA-2019 model (k = 2, `system` 0.1) differ by **0.21** on BEA-2019 test (77.24 vs 77.45). On dev the order was the other way round: k = 3 was ahead by 0.13 (60.35 vs 60.22).

**We attribute this to differences in the content of the two sets.** Dev and test were written by different learners. They contain different sentences, a different mix of error types, and different reference annotations; the test references are hidden on Codabench. A setting that suits one set's errors slightly better can rank a little lower on the other. We did not test whether the gap is statistically significant.

The best dev model stays the main result, because it was chosen before any test score was seen. Switching to k = 2 because of a test score would be tuning on the test set.

**What this says about using more than each system's 1-best (k > 1):**

- **With ESC, adding hypotheses does not hurt.** Every k from 1 to 5 scores between 76.73 and 77.45 on BEA-2019 test, and both best models use k > 1.
- **Without ESC, adding hypotheses clearly hurts.** GRECO alone falls from 76.82 at k = 1 to at most 74.83 at k = 2, 74.56 at k = 3 and 73.67 at k = 5.
- **So ESC is what makes the extra hypotheses safe to use.** The best k > 1 runs are only 0.07–0.28 above the best k = 1 run on test (0.04 on dev). The results show that k > 1 is no worse than k = 1 once ESC is on, not that it is better.

## Reproducing

```bash
sbatch -J greco-test --time=12:00:00 cluster/job.sbatch greco_esc/test.sh bea19
```

Then pack every output for Codabench and upload each zip under **My Submissions** on the BEA-2019 Codabench task:

```bash
python3 -m greco_esc.codabench     # writes codabench/bea19_k<k>_<run>.zip, one 4,477-line .txt each
```

The zips submitted here are in `codabench/` (not tracked).
