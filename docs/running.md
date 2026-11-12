# Running the project

Run commands from the repository root. The project does not have one single
model: it contains BART, T5, GECToR, GRECO/ESC combination workflows, and
training/ablation pipelines. Start with the smallest workflow that matches the
task.

## 1. Install a local environment

For data preparation, tests, and the training utilities:

```bash
# Use Python 3.11 or 3.12. Python 3.13+ is not supported by the training stack.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m spacy download en_core_web_sm

# Confirm that this exact interpreter has the training dependency.
python -c "import sys, transformers; print(sys.executable); print(transformers.__version__)"
```

The pretrained GECToR workflow uses its own environment and setup script; see
[`gector/README.md`](../gector/README.md). GRECO and ESC have separate
dependencies documented in [`greco_setup/README.md`](../greco_setup/README.md)
and [`greco_esc/README.md`](../greco_esc/README.md).

## 2. Check the checkout and tests

```bash
python -m unittest discover -s tests -v
```

If the M2 files are Git LFS pointers, retrieve them before gold evaluation:

```bash
git lfs install
git lfs pull
```

## 3. Prepare or verify data

Prepared data is already in `data/`. To rebuild the two-stage curriculum data
from the original M2 corpora:

```bash
python -m training.build_data
python -m training.check_data
```

The generated stage-1 files are intentionally ignored and are verified by
`data/curriculum/manifest.json`.

## 4. Run inference

The exact model/checkpoint arguments depend on the checkpoint location. The
existing model-specific instructions are the canonical references:

- BART generation and n-best exports: [`bart/`](../bart/)
- T5 generation and retokenization: [`t5/`](../t5/)
- GECToR setup and prediction: [`gector/README.md`](../gector/README.md)

All predictions should use the shared output convention:
`outputs/<system>/<split>.tgt`. Keep n-best candidates in consecutive groups,
with the best candidate first. This is the format consumed by GRECO.

For GECToR, after running `bash gector/setup.sh`, an example is:

```bash
gector/.venv/bin/python gector/predict.py \
  --input data/dev.src \
  --output outputs/gector/dev.tgt
```

## 5. Combine systems

The small, inspectable GRECO workflow can be run with:

```bash
python -m greco_setup.run_pilot
```

Use `--with-gector` when `outputs/gector/dev30.tgt` is available. For the
official GRECO scorer, tuning, and GRECO + ESC evaluation, follow the commands
in the component READMEs and `cluster/greco_tune.sh`.

## 6. Evaluate predictions

Development and experiment evaluation is performed by the existing training
and GRECO scripts. Test-set results and their provenance are documented in
[`experiments/test/README.md`](../experiments/test/README.md).

For the legacy CoNLL-2014 M2 scorer:

```bash
python2 m2scorer/scripts/m2scorer.py \
  predictions.txt data/test.conll14.m2
```

ERRANT evaluation requires the ERRANT environment described by the relevant
component README.

## 7. Run training and ablations on the cluster

Training requires a GPU and is designed for the NUS SoC Slurm cluster. Follow
[`cluster/README.md`](../cluster/README.md) for setup, model downloads, pilots,
submission, monitoring, and copying results back.

Before submitting ablations, preview the commands:

```bash
bash cluster/ablations/training/submit.sh --dry-run
bash cluster/ablations/decoding/submit.sh --dry-run
```

The canonical experiment descriptions are:

- [`experiments/ablations/README.md`](../experiments/ablations/README.md)
- [`experiments/four-controls/README.md`](../experiments/four-controls/README.md)
- [`t5_tuning/README.md`](../t5_tuning/README.md)

Do not submit a run merely because its result directory exists. Existing
curated scores are retained to prevent accidental duplicate training.

## 8. Where files should go

| File type | Location | Tracked? |
|---|---|---|
| Source code and tests | component directories, `tests/` | yes |
| Reusable configs | `configs/training/`, `configs/t5_tuning/` | yes |
| Curated experiment scores and notes | `experiments/` | yes |
| Predictions and n-best files | `outputs/` or `artifacts/` | no |
| Checkpoints and downloaded models | `checkpoints/`, `models/` | no |
| Slurm logs | `logs/` | no |
| Local environments and scratch | `.venv*`, `check/`, `tmp/`, `output/` | no |

## 9. Use the public team models

The five trained BART and T5 models are public on Hugging Face, so no account,
login or token is needed. What each model is and how it scores is listed in
[`configs/models/README.md`](../configs/models/README.md).

Set up the environment from [step 1](#1-install-a-local-environment). Every
script in this repository accepts a Hugging Face ID wherever it accepts a
checkpoint directory. The first run downloads the model into the local Hugging
Face cache (`~/.cache/huggingface` by default) and later runs reuse it.

To correct and score the dev_test split:

```bash
python -m training.evaluate \
  --model CS4248-GEC/gec-t5-base-s1s2 \
  --split dev_test \
  --out outputs/review/t5-s1s2
```

Replace the model ID with any of the five. This writes `dev_test.pred` (one
corrected sentence per line) and `dev_test.scores.json` to the `--out`
directory. The scripts add T5's `gec: ` prefix and restore characters that
T5's vocabulary cannot encode, so the output is ready for ERRANT, the M2 scorer
or Codabench. To produce n-best lists for GRECO, use
`bart/generate_nbest.py --model <ID>`, which works for both BART and T5.

To download all five models in advance, for example on a machine that will
later run offline:

```bash
bash scripts/download_public_models.sh
```

This needs the `hf` command from `huggingface_hub`, which `requirements-train.txt`
installs as a dependency of `transformers`. A single model can be fetched with
`hf download CS4248-GEC/<repo>`. On the NUS SoC cluster, jobs run offline, so
download on a compute node with `HF_HOME` set to the scratch cache; see
[`cluster/README.md`](../cluster/README.md).
