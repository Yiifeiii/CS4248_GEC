# Running the project

Run commands from the repository root. The project does not have one single
model: it contains BART, T5, GECToR, GRECO/ESC combination workflows, and
training/ablation pipelines. Start with the smallest workflow that matches the
task.

## 1. Install a local environment

For data preparation, tests, and the training utilities:

```bash
# Use Python 3.11 or 3.12 (3.13 removed lib2to3, which the M2 scorer conversion needs).
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

**BART and T5, 1-best.** `training.evaluate` corrects a split with a trained
model (a checkpoint folder or a Hugging Face ID), writes `<split>.pred`, and
scores it where references exist. Splits: `dev_tune`, `dev_test`,
`conll14_test` and `bea19_test` (BEA-2019 has no public references, so only
the predictions are written):

```bash
python -m training.evaluate --model CS4248-GEC/gec-bart-base-s1s2 \
  --split conll14_test bea19_test --out outputs/review/bart-s1s2
```

**BART and T5, n-best lists for GRECO.** `bart/generate_nbest.py` works for both
models (it adds T5's `gec: ` prefix itself). `bart/export_candidates.py` then
writes the shared `.tgt` format below:

```bash
python bart/generate_nbest.py --model CS4248-GEC/gec-t5-base-s1s2 \
  --input data/dev.src --output outputs/t5/dev.nbest.jsonl --num_beams 10 --num_return 10
python bart/export_candidates.py --nbest outputs/t5/dev.nbest.jsonl --out outputs/t5/dev.tgt --k 5
```

Without `--output`, `generate_nbest.py --text "She go to school ."` prints the
candidates for one sentence. The `t5/` folder holds T5 output retokenization and
early pilot scripts on the pretrained `vennify/t5-base-grammar-correction`, not
our fine-tuned models.

**GECToR.** After running `bash gector/setup.sh`:

```bash
gector/.venv/bin/python gector/predict.py \
  --input data/dev.src \
  --output outputs/gector/dev.tgt
```

All predictions for GRECO use the shared output convention
`outputs/<system>/<split>.tgt`: k consecutive lines per source sentence, best
candidate first, tokenized like the source. See the
[main README](../README.md#outputs).

## 5. Combine systems

The small, inspectable GRECO workflow can be run with:

```bash
python -m greco_setup.run_pilot
```

Use `--with-gector` when `outputs/gector/dev30.tgt` is available. For the
official GRECO scorer, tuning, and GRECO + ESC evaluation, follow the commands
in the component READMEs and `cluster/greco_tune.sh`.

## 6. Evaluate predictions

`training.evaluate` (section 4) already reports ERRANT and M2 scores for every
split with public references. To score any other prediction file against a gold
M2 file, use ERRANT's own commands, installed with `requirements-train.txt`:

```bash
errant_parallel -orig data/test.conll14.src -cor predictions.txt -out predictions.m2
errant_compare -hyp predictions.m2 -ref data/test.conll14.m2
```

CoNLL-2014's official metric is the bundled M2 scorer. It is Python 2 code;
`training.evaluate` runs it under Python 3 automatically, and to run it by hand
use a Python 2 environment:

```bash
python2 m2scorer/scripts/m2scorer.py predictions.txt data/test.conll14.m2
```

BEA-2019 test has no public references: submit predictions to its Codabench
task. Test-set results and their provenance are in
[`experiments/test/README.md`](../experiments/test/README.md) and
[`greco_esc/results/`](../greco_esc/README.md#results).

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
