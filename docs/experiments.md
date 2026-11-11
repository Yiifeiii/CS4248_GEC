# Experiment index

This index separates the project's research records from generated runtime
files. The existing directory names are retained so current links and scripts
continue to work.

Implementation now lives under `src/gec/`; configuration files live under
`configs/`. The old top-level package and config paths are compatibility links.

| Area | Location | Purpose |
|---|---|---|
| BART ablations | [`experiments/ablations/`](../experiments/ablations/) | Entry point for BART training and decoding studies |
| BART training ablation record | [`experiments/training/bart-ablations-20261105/`](../experiments/training/bart-ablations-20261105/) | Stage-2 hyperparameter sweep, provenance, and results |
| BART decoding | [`experiments/bart-decoding-20261106/`](../experiments/bart-decoding-20261106/) | Frozen-checkpoint beam and diversity study |
| Four controls | [`experiments/four-controls/`](../experiments/four-controls/) | Matched-control comparisons for BART and T5 |
| T5 tuning | [`t5_tuning/`](../t5_tuning/) | T5 sweeps, noise bands, configs, and scores |
| Training runs | [`experiments/training/`](../experiments/training/) | Per-run metadata, scores, and selected result files |
| Test results | [`experiments/test/`](../experiments/test/) | CoNLL-2014 and BEA-2019 evaluation |
| Overall comparison | [`experiments/curriculum-comparison.md`](../experiments/curriculum-comparison.md) | Curriculum and control comparison |

The old package paths are compatibility links:

```text
training/    -> src/gec/training/
bart/        -> src/gec/bart/
t5/          -> src/gec/t5/
greco_setup/ -> src/gec/greco_setup/
greco_esc/   -> src/gec/greco_esc/
```

## Provenance rule

When adding an experiment, commit the config, a short README or result note,
the source commit, and compact machine-readable scores. Keep checkpoints,
predictions, n-best pools, and logs outside Git unless they are explicitly
needed as a small reproducibility fixture.
