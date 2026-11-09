# BEA-2019 test results (Codabench)

ERRANT span-correction F0.5, as reported by the BEA-2019 Codabench task, on the 4,477 test sentences. Each row is one submission of `runs/bea19/k<k>/<run>/pred.txt`, made from test jobs 1099945 and 1100522 (released GRECO checkpoint; the second reran the runs whose α changed when the full-dev α were adopted) and submitted on 9 Nov 2026. Codabench shows only F0.5, so precision and recall are not given.

Nothing was tuned or trained on the test set. ESC was trained once per k on the full dev set (BEA-2019 dev, 4,384 sentences). The vote coefficients α are the ones tuned on the full dev set (`settings.VOTES`). **The final system was fixed on the full dev set before any test submission**: GRECO + ESC, k = 3, `rank` vote 0.4 (60.4 ERRANT F0.5 on dev). The other rows are analysis and must not be used to pick a different winner.

## Results

k is the number of hypotheses taken from each of BART and T5. GECToR always contributes its 1-best, so GRECO combines 1 + 2k inputs.

| k | Run | GRECO alone | GRECO + ESC |
|---|---|---|---|
| 1 | no vote | 76.21 | 77.17 |
| 1 | vote 0.1 (all modes are the same at k = 1) | 76.82 | 77.16 |
| 2 | no vote | 72.84 | 77.11 |
| 2 | `hypothesis` 0.1 | 73.88 | 77.33 |
| 2 | `system` 0.1 | 73.95 | **77.45** |
| 2 | `rank` 0.2 | 74.83 | 77.28 |
| 3 | no vote | 70.22 | 76.77 |
| 3 | `hypothesis` 0.1 | 71.92 | 76.89 |
| 3 | `system` 0.1 | 71.79 | 77.09 |
| 3 | `rank` 0.4 | 74.56 | **77.24** (final system) |
| 5 | (not submitted yet) | – | – |

ESC alone (its own edit selection, without GRECO):

| k = 1 | k = 2 | k = 3 | k = 5 |
|---|---|---|---|
| 76.12 | 76.33 | 75.88 | 75.57 |

**Not submitted yet:** the eight k = 5 GRECO runs and the single systems (BART, T5 and GECToR top-1), which give the baseline the combination should beat.

## Findings

1. **The final system scores 77.24.** It is 0.21 behind the best submission (k = 2, GRECO + ESC, `system` 0.1, 77.45). The ten GRECO + ESC runs all fall between 76.77 and 77.45, so a 0.2 gap is within the spread and does not overturn the choice made on dev.
2. **ESC is what makes the combination work.** At every k and with every voting setting, GRECO + ESC beats GRECO alone, by 0.3–1.0 at k = 1 and by 2.7–6.6 at k = 3. Every GRECO + ESC run also beats ESC alone (at most 76.33).
3. **GRECO alone gets worse as k grows**: 76.82 at k = 1, at most 74.83 at k = 2 and at most 74.56 at k = 3. More hypotheses give its beam search more wrong edits to choose from. The full dev set showed the same pattern.
4. **Rank voting is the best voting for GRECO alone at larger k.** At k = 3 it scores 74.56, against 70.22 without voting and about 71.9 for `hypothesis` and `system`. It counts a system's 1-best more than its lower-ranked hypotheses, so lower-ranked edits get less support. With ESC on, the vote modes are within 0.5 of each other.
5. **The test set ranks the runs the same way the full dev set did.** On dev, GRECO + ESC was flat across k (59.7–60.4), GRECO alone fell with k, and rank voting helped most without ESC. All three hold on the test set.

## Reproducing

```bash
sbatch -J greco-test --time=12:00:00 cluster/job.sbatch greco_esc/test.sh bea19
```

Then zip `runs/bea19/k<k>/<run>/pred.txt` and upload it under **My Submissions** on the BEA-2019 Codabench task. The zips submitted here are in `codabench/` (not tracked).
