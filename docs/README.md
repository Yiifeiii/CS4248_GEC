# Documentation

This directory contains the user-facing guides for the project. Existing
component READMEs remain beside their code so that historical links and
experiment commands continue to work.

- [Running the project](running.md): local setup, inference, evaluation,
  combination, tests, and cluster training.
- [Experiment index](experiments.md): where to find baselines, ablations,
  controls, decoding studies, and test-set results.
- [Public Hugging Face models](running.md#9-use-the-public-team-models):
  token-free downloading and inference guidance for reviewers.

Generated predictions, checkpoints, logs, caches, and local scratch files are
not part of the source of truth. They should be written to ignored directories
such as `outputs/`, `checkpoints/`, `logs/`, or `artifacts/`. Curated scores,
configs, manifests, and experiment notes belong under `experiments/`.
