# CoNLL-2014 test results

All 1,312 CoNLL-2014 test sentences, scored against `data/test.conll14.m2`, the official two-annotator gold. **M2 F0.5 is CoNLL-2014's official metric.** ERRANT F0.5 is also given so the numbers can be compared with dev and BEA-2019. The outputs come from test jobs 1099945 and 1100522. The second job reran every run whose α changed when the full-dev α were adopted, and runs whose settings did not change were reused. Scores were made with `python -m greco_esc.evaluate conll14` and copied from `runs/conll14/results.{csv,md}` into [`conll14.csv`](conll14.csv). The `runs` folder is not tracked.

Nothing was tuned or trained on CoNLL-2014. ESC was trained once per k on the full BEA-2019 dev set (4,384 sentences). The α values are the full-dev ones in `settings.VOTES`. The best dev model, chosen on the full dev set without seeing any test score, is GRECO + ESC, k = 3, `rank` 0.4 (60.35 on dev).

## Models to report

| Model | Run | M2 F0.5 (official) | P / R | ERRANT F0.5 | How it was picked |
|---|---|---|---|---|---|
| **Best CoNLL-2014 model** | GRECO alone, k = 2, `rank` 0.2 | **67.98** | 76.06 / 47.72 | 64.02 | highest test M2 |
| **Best dev model** | GRECO + ESC, k = 3, `rank` 0.4 | **66.14** | 80.43 / 38.67 | 64.47 | highest full-dev ERRANT F0.5 (60.35) |
| Best GRECO + ESC | GRECO + ESC, k = 5, `system` 0.1 | 66.96 | 81.19 / 39.36 | 64.84 | highest test M2 with ESC |
| Best ESC alone | ESC, k = 1 | 64.02 | 78.69 / 36.67 | 62.13 | |
| Best single system | GECToR | 64.03 | 75.02 / 40.37 | 59.94 | |

The best CoNLL-2014 model was picked by looking at test scores, so 67.98 shows the best this setup reaches, not a held-out estimate. The best dev model, chosen without seeing any test score, is the unbiased one.

## Every run

k is the number of hypotheses taken from each of BART and T5. GECToR always gives its 1-best, so there are 1 + 2k inputs. At k = 1 the three vote modes give the same output, so only `hypothesis` is run.

| k | Inputs | System | ERRANT P | ERRANT R | ERRANT F0.5 | M2 P | M2 R | M2 F0.5 |
|---|---|---|---|---|---|---|---|---|
| - | 1 | bart-1 (single system) | 65.17 | 40.40 | 58.05 | 71.43 | 43.33 | 63.23 |
| - | 1 | t5-1 (single system) | 63.65 | 42.52 | 57.89 | 70.10 | 45.15 | 63.12 |
| - | 1 | gector-1 (single system) | 69.54 | 38.61 | 59.94 | 75.02 | 40.37 | 64.03 |
| 1 | 3 | ESC alone | 76.17 | 35.76 | 62.13 | 78.69 | 36.67 | 64.02 |
| 1 | 3 | GRECO, no vote | 71.70 | 43.40 | 63.43 | 76.78 | 45.96 | 67.70 |
| 1 | 3 | GRECO, `hypothesis` 0.1 | 72.44 | 43.14 | 63.78 | 77.20 | 45.52 | 67.76 |
| 1 | 3 | GRECO + ESC, no vote | 77.58 | 37.18 | 63.73 | 79.76 | 38.08 | 65.43 |
| 1 | 3 | GRECO + ESC, `hypothesis` 0.1 | 77.61 | 37.36 | 63.85 | 80.03 | 38.36 | 65.75 |
| 2 | 5 | ESC alone | 76.86 | 34.72 | 61.85 | 79.43 | 35.67 | 63.78 |
| 2 | 5 | GRECO, no vote | 68.52 | 46.19 | 62.48 | 73.42 | 48.96 | 66.75 |
| 2 | 5 | GRECO, `hypothesis` 0.1 | 70.47 | 46.61 | 63.92 | 74.69 | 49.00 | 67.60 |
| **2** | **5** | **GRECO, `rank` 0.2 (best CoNLL-2014)** | **71.47** | **45.18** | **64.02** | **76.06** | **47.72** | **67.98** |
| 2 | 5 | GRECO, `system` 0.1 | 70.00 | 46.60 | 63.61 | 74.63 | 49.15 | 67.62 |
| 2 | 5 | GRECO + ESC, no vote | 77.71 | 36.91 | 63.64 | 80.47 | 37.93 | 65.73 |
| 2 | 5 | GRECO + ESC, `hypothesis` 0.1 | 77.38 | 37.60 | 63.87 | 80.21 | 38.74 | 66.06 |
| 2 | 5 | GRECO + ESC, `rank` 0.2 | 77.64 | 37.87 | 64.16 | 80.39 | 39.03 | 66.33 |
| 2 | 5 | GRECO + ESC, `system` 0.1 | 77.56 | 37.50 | 63.90 | 80.41 | 38.67 | 66.13 |
| 3 | 7 | ESC alone | 78.27 | 31.97 | 60.69 | 80.66 | 32.86 | 62.48 |
| 3 | 7 | GRECO, no vote | 65.32 | 48.28 | 61.01 | 70.77 | 51.32 | 65.78 |
| 3 | 7 | GRECO, `hypothesis` 0.1 | 67.87 | 48.17 | 62.74 | 72.56 | 51.01 | 66.91 |
| 3 | 7 | GRECO, `rank` 0.4 | 71.01 | 44.88 | 63.60 | 75.57 | 47.34 | 67.52 |
| 3 | 7 | GRECO, `system` 0.1 | 66.42 | 48.26 | 61.77 | 71.42 | 50.94 | 66.10 |
| 3 | 7 | GRECO + ESC, no vote | 79.31 | 35.34 | 63.51 | 81.59 | 36.24 | 65.26 |
| 3 | 7 | GRECO + ESC, `hypothesis` 0.1 | 78.16 | 36.48 | 63.62 | 80.51 | 37.50 | 65.48 |
| 3 | 7 | **GRECO + ESC, `rank` 0.4 (best dev model)** | 78.42 | 37.66 | 64.47 | 80.43 | 38.67 | 66.14 |
| 3 | 7 | GRECO + ESC, `system` 0.1 | 78.57 | 36.15 | 63.63 | 80.73 | 37.13 | 65.38 |
| 5 | 11 | ESC alone | 78.04 | 33.15 | 61.41 | 80.78 | 34.04 | 63.38 |
| 5 | 11 | GRECO, no vote | 62.67 | 49.47 | 59.50 | 68.78 | 52.95 | 64.90 |
| 5 | 11 | GRECO, `hypothesis` 0.1 | 64.84 | 49.46 | 61.04 | 70.33 | 52.35 | 65.81 |
| 5 | 11 | GRECO, `rank` 0.6 | 70.72 | 44.71 | 63.35 | 75.37 | 47.08 | 67.28 |
| 5 | 11 | GRECO, `system` 0.1 | 63.73 | 49.84 | 60.37 | 69.03 | 52.71 | 65.01 |
| 5 | 11 | GRECO + ESC, no vote | 78.37 | 36.85 | 63.96 | 80.88 | 37.82 | 65.88 |
| 5 | 11 | GRECO + ESC, `hypothesis` 0.1 | 77.45 | 38.91 | 64.64 | 80.25 | 40.25 | 66.94 |
| 5 | 11 | GRECO + ESC, `rank` 0.6 | 76.85 | 39.72 | 64.74 | 79.45 | 41.02 | 66.91 |
| 5 | 11 | GRECO + ESC, `system` 0.1 | 78.59 | 38.15 | 64.84 | 81.19 | 39.36 | 66.96 |

## GRECO alone vs GRECO + ESC (M2 F0.5, official)

| k | Voting | GRECO alone | GRECO + ESC | ESC's effect |
|---|---|---|---|---|
| 1 | no vote | 67.70 | 65.43 | −2.27 |
| 1 | `hypothesis` 0.1 | 67.76 | 65.75 | −2.01 |
| 2 | no vote | 66.75 | 65.73 | −1.02 |
| 2 | `hypothesis` 0.1 | 67.60 | 66.06 | −1.54 |
| 2 | `system` 0.1 | 67.62 | 66.13 | −1.49 |
| 2 | `rank` 0.2 | **67.98** | 66.33 | −1.65 |
| 3 | no vote | 65.78 | 65.26 | −0.52 |
| 3 | `hypothesis` 0.1 | 66.91 | 65.48 | −1.43 |
| 3 | `system` 0.1 | 66.10 | 65.38 | −0.72 |
| 3 | `rank` 0.4 | 67.52 | 66.14 (best dev model) | −1.38 |
| 5 | no vote | 64.90 | 65.88 | +0.98 |
| 5 | `hypothesis` 0.1 | 65.81 | 66.94 | +1.13 |
| 5 | `system` 0.1 | 65.01 | **66.96** | +1.95 |
| 5 | `rank` 0.6 | 67.28 | 66.91 | −0.37 |

## Findings

1. **The best dev model (GRECO + ESC, k = 3, `rank` 0.4) scores 66.14 M2 F0.5** (P 80.43 / R 38.67) and 64.47 ERRANT F0.5. That is 2.11 above the best single system, GECToR (64.03 M2).
2. **Every GRECO run, with or without ESC, beats every single system on M2.** The lowest, GRECO alone at k = 5 with no vote, scores 64.90, against 64.03 for GECToR, 63.23 for BART and 63.12 for T5. ESC alone is level with GECToR at k = 1 (64.02) and below it at larger k.
3. **The best CoNLL-2014 model is GRECO alone, k = 2, `rank` 0.2, at 67.98 M2** (P 76.06 / R 47.72). It is 1.84 above the best dev model. It was picked by its test score, so it shows the best this setup reaches on CoNLL-2014, not a held-out estimate.
4. **ESC lowers M2 at k = 1 to 3**, by 0.5–2.3 points in every pair, and raises it at k = 5 except with `rank` voting. On ERRANT, ESC raises the score in 13 of the 14 pairs (the one exception is k = 2 `hypothesis`, −0.05). So on CoNLL-2014, whether ESC helps depends on the metric.
5. **ESC trades recall for precision, as on BEA-2019.** Against GRECO alone with the same voting, ERRANT precision rises by 5–16 points and recall falls by 5–13 (on BEA-2019 dev: +4.3–15.7 and −4.3–10.2, [`dev.md`](dev.md)). F0.5 weights precision more, but on CoNLL-2014's M2 the recall loss is larger than the precision gain at k ≤ 3.
6. **Voting matters more without ESC.** GRECO alone at k = 5 goes from 64.90 with no vote to 67.28 with `rank` 0.6. With ESC, the vote modes at a given k stay within 1.1 points of each other.

## Why ESC helps on BEA-2019 but not on CoNLL-2014 M2

- **ESC learned BEA's annotation, not CoNLL's.** It was trained on BEA-2019 dev (W&I+LOCNESS). Its labels say which edits BEA's annotators made, and those annotators work under different guidelines and see a different mix of errors from CoNLL-2014's NUCLE essays. Edits that BEA annotators leave out are learned as "wrong", even where CoNLL's annotators would have accepted them.
- **The labels are ERRANT edits, and M2 matches edits differently.** ESC's training labels come from ERRANT's edit extraction of the BEA gold. The M2 scorer chooses whichever split of a correction best matches the gold edits. A precision gain measured in ERRANT's units does not fully carry over to M2, and the recall loss does.
- **The GRECO paper did the same.** It uses ESC's edit scores only for BEA-2019 and reports GRECO with voting alone on CoNLL-2014. Our results agree with that choice. We report the best dev model on both test sets and the best CoNLL-2014 model (GRECO alone) next to it.
