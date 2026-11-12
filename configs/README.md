# Configurations

Every training run, ablation and tuning sweep in this project is described by a
JSON config in this directory. A config fixes the starting weights, the
training data and the hyperparameters, so that the run it describes can be
repeated. Results for each run are recorded under
[`experiments/`](../experiments/); see the
[experiment index](../docs/experiments.md).

| Directory | Contents | Written by |
|---|---|---|
| [`training/`](training/) | BART and T5 fine-tuning configs, plus the BART ablation and decoding studies | hand |
| [`t5_tuning/`](t5_tuning/) | One config per run of the T5 hyperparameter sweep | `t5_tuning/tune.py plan` |
| [`models/`](models/) | IDs of the released models on Hugging Face | hand |

## `training/`

- `bart-base.stage{1,2}.json`, `t5-base.stage{1,2}.json`: the main two-stage
  curriculum. Stage 1 starts from the pretrained model; stage 2 continues from
  the best stage-1 checkpoint.
- `*.long.*`: the same recipe trained for longer, plus a matched stage-1
  control ([description](../training/README.md#longer-runs-and-the-matched-control-long-configs)).
- `bart-base.gtrecipe.*`: stage 1 with the hyperparameters of
  [gotutiyan/GEC-BART](https://github.com/gotutiyan/GEC-BART)
  ([description](../experiments/training/README.md#the-gotutiyan-recipe-comparison-gtrecipe)).
- `four-controls/`: the matched controls for the curriculum comparison
  ([description](../experiments/four-controls/README.md)).
- `bart-ablations-20261105/`: the one-at-a-time BART stage-2 hyperparameter
  ablation. `runs.txt` maps each Slurm array index to a config
  ([description](training/bart-ablations-20261105/README.md)).
- `bart-decoding-20261106.json`: the beam-search settings for the BART decoding
  study, which uses a trained checkpoint and does no training
  ([description](../experiments/bart-decoding-20261106/README.md)).

Run a config with:

```bash
python -m training.finetune --config configs/training/bart-base.stage1.json
```

Training needs a GPU; the full runs were done on the NUS SoC cluster
([`cluster/README.md`](../cluster/README.md)). Paths may contain
`${CKPT_ROOT}`, the directory where checkpoints are written, which must be set
in the environment. The meaning of each field is described in
[`training/README.md`](../training/README.md).

## `t5_tuning/`

These configs are generated, not written by hand. `t5_tuning/tune.py plan`
copies `training/t5-base.stage1.json` or `training/t5-base.stage2.json` and
changes one setting (learning rate, weight decay, dropout, label smoothing or
seed) per run. Each run is also listed in
[`t5_tuning/manifest.json`](../t5_tuning/manifest.json), which `tune.py`
checks the configs against before training. To change the sweep, edit and rerun
`tune.py` rather than these files. The sweep and its results are documented in
[`t5_tuning/README.md`](../t5_tuning/README.md).

## `models/`

[`models/public_models.json`](models/public_models.json) lists the trained
models published on Hugging Face. See [`models/README.md`](models/README.md)
for how to download and run them.
