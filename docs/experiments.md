# Experiment index

Where to find the write-up and results of each study. For a one-page summary of
the main numbers, see the [headline results](../results/README.md).

| Study | Location | What it contains |
|---|---|---|
| Headline results | [`results/`](../results/README.md) | Summary of the main numbers across all studies |
| Two-stage fine-tuning | [`experiments/training/`](../experiments/training/README.md) | The 12 BART/T5 training runs: configs, per-run scores, findings, released models |
| Curriculum vs controls | [`experiments/four-controls/`](../experiments/four-controls/README.md), [`curriculum-comparison.md`](../experiments/curriculum-comparison.md) | The two-stage curriculum against four single-stage shuffled controls |
| BART ablations | [`experiments/ablations/`](../experiments/ablations/README.md) | Entry point for the BART training and decoding studies |
| BART stage-2 hyperparameters | [`experiments/training/bart-ablations-20261105/`](../experiments/training/bart-ablations-20261105/README.md) | 12-run one-at-a-time sweep: design, provenance and results |
| BART decoding | [`experiments/bart-decoding-20261106/`](../experiments/bart-decoding-20261106/README.md) | Beam width and diverse beam search on a frozen checkpoint |
| T5 tuning | [`t5_tuning/`](../t5_tuning/README.md) | 21 runs: noise bands, learning-rate and regularization sweeps, final dev_test |
| GECToR | [`gector/`](../gector/README.md) | Grammarly's pretrained GECToR: setup and dev scores |
| GRECO combination and voting | [`greco_setup/`](../greco_setup/README.md), [`outputs/greco/tuning/`](../outputs/greco/tuning/dev.errant.beam/report.md) | Combining BART, T5 and GECToR with GRECO; vote modes and α tuning on dev |
| Test sets (single models, GRECO) | [`experiments/test/`](../experiments/test/README.md) | CoNLL-2014 and BEA-2019 test scores for the curriculum models and GRECO |
| GRECO + ESC (main result) | [`greco_esc/`](../greco_esc/README.md) | The final combination: every run on BEA-2019 dev, BEA-2019 test and CoNLL-2014 test |

## What is committed for each experiment

Experiment records contain configs, write-ups and machine-readable scores, and
most record the source commit they ran from. Predictions saved with an
experiment's scores under `experiments/` are committed too. Checkpoints, n-best
lists and logs are not.
