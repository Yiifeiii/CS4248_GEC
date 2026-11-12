# Documentation

Guides for running the project and finding its results. Each component also has
its own README beside its code (for example [`training/`](../training/README.md)
or [`greco_esc/`](../greco_esc/README.md)).

- [Running the project](running.md): local setup, inference, evaluation,
  combination, tests, and cluster training.
- [Public Hugging Face models](running.md#9-use-the-public-team-models): how to
  download and run the trained BART and T5 models, with no account or token.
- [Experiment index](experiments.md): where each study's write-up and results are.
- [Headline results](../results/README.md): a one-page summary of the main numbers.

Results are committed as scores, write-ups and, under `experiments/`, the
predictions they were computed from. Checkpoints, n-best lists, logs and other
generated files are not committed; they go to git-ignored directories such as
`outputs/`, `checkpoints/` and `logs/`.
