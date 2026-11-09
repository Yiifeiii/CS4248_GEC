# Test-set results

Nothing is tuned on these sets: the checkpoints were chosen on dev_tune and GRECO's vote coefficients α on the full W&I+LOCNESS dev set (outputs/greco/tuning/dev.errant.beam). F0.5 (P / R). CoNLL-2014 (1,312 sentences) is scored with the M2 scorer, its official metric, and ERRANT; BEA-2019 (4,477) with ERRANT on Codabench. Single BART and T5 rows are beam-5 1-best (training.evaluate); GRECO pools the top-k of 10-beam lists (cluster/nbest_beam.sh). Written by greco_setup/test_report.py.

| Enhancement | Output | CoNLL-14 M2 | CoNLL-14 ERRANT | BEA-19 ERRANT (Codabench) |
|---|---|---|---|---|
| 1. curriculum | BART, stage 1 only (no curriculum) | 59.7 (67.6 / 40.7) | 55.2 (62.2 / 38.2) | – |
| 1. curriculum | BART, stage 1 + 2 (curriculum) | 62.9 (70.9 / 43.4) | 57.9 (64.8 / 40.5) | – |
| 1. curriculum | T5, stage 1 only (no curriculum) | 61.7 (72.2 / 39.0) | 57.3 (66.8 / 36.5) | – |
| 1. curriculum | T5, stage 1 + 2 (curriculum) | 63.0 (69.8 / 45.4) | 57.7 (63.4 / 42.5) | – |
| 2. GRECO k = 1 | GECToR (official RoBERTa checkpoint) | 64.0 (75.0 / 40.4) | 59.9 (69.5 / 38.6) | – |
| 2. GRECO k = 1 | GRECO BART + T5 + GECToR, k = 1, no voting | 67.7 (76.8 / 46.0) | 63.4 (71.7 / 43.4) | – |
| 2. GRECO k = 1 | GRECO BART + T5 + GECToR, k = 1, rank vote, α 0.1 | 67.8 (77.2 / 45.5) | 63.8 (72.4 / 43.1) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 2, no voting | 66.8 (73.4 / 49.0) | 62.5 (68.5 / 46.2) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 2, hypothesis vote, α 0.1 | 67.6 (74.7 / 49.0) | 63.9 (70.5 / 46.6) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 2, system vote, α 0.1 | 67.6 (74.6 / 49.1) | 63.6 (70.0 / 46.6) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 2, rank vote, α 0.2 | 68.0 (76.1 / 47.7) | 64.0 (71.5 / 45.2) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 3, no voting | 65.8 (70.8 / 51.3) | 61.0 (65.3 / 48.3) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 3, hypothesis vote, α 0.1 | 66.9 (72.6 / 51.0) | 62.7 (67.9 / 48.2) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 3, system vote, α 0.1 | 66.1 (71.4 / 50.9) | 61.8 (66.4 / 48.3) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 3, rank vote, α 0.4 | 67.5 (75.6 / 47.3) | 63.6 (71.0 / 44.9) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 5, no voting | 64.9 (68.8 / 52.9) | 59.5 (62.7 / 49.5) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 5, hypothesis vote, α 0.1 | 65.8 (70.3 / 52.3) | 61.0 (64.8 / 49.5) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 5, system vote, α 0.1 | 65.0 (69.0 / 52.7) | 60.4 (63.7 / 49.8) | – |
| 3. larger k | GRECO BART + T5 + GECToR, k = 5, rank vote, α 0.6 | 67.3 (75.4 / 47.1) | 63.3 (70.7 / 44.7) | – |
