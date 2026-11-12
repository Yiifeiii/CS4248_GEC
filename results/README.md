# Headline results

A summary of the project's main results. Every number is copied from the
detailed result pages linked in each section, which also give precision,
recall and every other run.

**Metrics.** BEA-2019 dev and test use ERRANT F0.5, BEA-2019's official
metric; BEA-2019 test is scored on its Codabench task because its references
are hidden. CoNLL-2014 test uses M2 F0.5, its official metric. dev_tune and
dev_test are the 2,000- and 2,384-sentence halves of BEA-2019 (W&I+LOCNESS)
dev.

**How settings were chosen.** Training checkpoints were chosen on dev_tune.
GRECO's vote weights and the best combination were chosen on the full BEA-2019
dev set. Nothing was tuned on CoNLL-2014 or BEA-2019 test.

## System combination (main result)

Our fine-tuned BART and T5 (top-k hypotheses each) and the pretrained GECToR,
combined with GRECO and ESC edit scores
([details](../greco_esc/README.md)).

| Model | BEA-2019 dev | BEA-2019 test | CoNLL-2014 test (M2) |
|---|---:|---:|---:|
| **GRECO + ESC, k = 3, `rank` vote 0.4** (best on dev: main result) | **60.35** | **77.24** | **66.14** |
| GRECO + ESC, k = 2, `system` vote 0.1 (best on BEA-2019 test) | 60.22 | 77.45 | 66.13 |
| GRECO alone, k = 2, `rank` vote 0.2 (best on CoNLL-2014 test) | 58.59 | 74.83 | 67.98 |
| GECToR (best single system) | 54.77 | 71.57 | 64.03 |

The main result was chosen on dev before any test score was seen, so its test
scores are held-out estimates. It is 5.7 points above GECToR on BEA-2019 test
and 2.1 on CoNLL-2014. The other two combination rows were picked by looking at
test scores, so they show the best each setup reaches, not held-out estimates.

- With ESC, using more than each system's 1-best (k > 1) does no harm: every k
  scores 76.73–77.45 on BEA-2019 test. Without ESC, larger k lowers the score.
- ESC is trained on BEA-2019 dev. It raises BEA-2019 test scores but lowers
  CoNLL-2014's M2 at k ≤ 3, which is why GRECO alone is best there.

Every run on every set: [dev](../greco_esc/results/dev.md),
[BEA-2019 test](../greco_esc/results/bea19.md),
[CoNLL-2014 test](../greco_esc/results/conll14.md).

## Single models and the two-stage curriculum

BART-base and T5-base, fine-tuned on all four BEA-2019 corpora (stage 1), then
on W&I train (stage 2). Beam 5, 1-best
([test results](../experiments/test/README.md),
[dev results](../experiments/training/README.md)).

| Model | dev_test | BEA-2019 test | CoNLL-2014 test (M2) |
|---|---:|---:|---:|
| BART, stage 1 only | 43.8 | 62.3 | 59.7 |
| BART, stage 1 + 2 | 52.3 | 69.3 | 62.9 |
| T5, stage 1 only | 46.4 | 67.6 | 61.7 |
| T5, stage 1 + 2 | 52.5 | 69.8 | 63.0 |

Stage 2 adds 6–9 points on dev_test. A longer T5 run
(`CS4248-GEC/gec-t5-base-long-s1s2`) is our best single model on dev_test at
53.2. The single-system rows in the combination results differ slightly from
these because they come from a different decoding (10 beams, re-ranked) used
to build the combination's candidate lists.

The trained models are public on Hugging Face; see
[`configs/models/`](../configs/models/README.md).

## Curriculum versus single-stage training

The two-stage curriculum beats all four single-stage shuffled controls for
both models (dev_test F0.5; [details](../experiments/curriculum-comparison.md)).

| Setup | BART | T5 |
|---|---:|---:|
| Curriculum (stage 1, then stage 2) | **52.26** | **52.50** |
| Best control (1A: same data and exposure, shuffled) | 49.01 | 50.74 |

## Tuning and ablations

- **T5 hyperparameters:** 21 runs; no change beat the default settings by more
  than seed noise. The defaults score 52.53 on dev_test (mean of 3 seeds)
  ([details](../t5_tuning/README.md)).
- **BART stage-2 hyperparameters:** 12 one-at-a-time variants of learning
  rate, weight decay, dropout and label smoothing; none beats the baseline on
  dev_test (52.26) ([details](../experiments/training/bart-ablations-20261105/results-all-12.md)).
- **BART decoding:** 5, 10 or 20 beams and diverse beam search change dev_tune
  F0.5 by at most 0.2 points ([details](../experiments/bart-decoding-20261106/results.md)).

All of these are single runs per setting (or three seeds where stated).
