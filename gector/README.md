# Official GECToR

Corrects sentences with Grammarly's official pretrained GECToR, RoBERTa encoder (`roberta_1_gectorv2.th`, reported F₀.₅ 64.0 on CoNLL-2014 and 71.8 on BEA-2019 test). The output is one corrected sentence per line, which is the GECToR input for [GRECO combination](../greco_setup/README.md).

| Path | Purpose | In git |
| --- | --- | --- |
| `predict.py` | Correct a file of tokenized sentences | yes |
| `setup.sh` | Create `.venv` and download the official files | yes |
| `.venv/` | Python environment with the GECToR port | no |
| `models/roberta_1_gectorv2.th` | Official weights (490 MB) | no |
| `models/output_vocabulary/` | Official tag vocabulary (5,000 tags) and detection tags | no |
| `models/verb-form-vocab.txt` | Verb forms used by `$TRANSFORM_VERB_*` tags | no |

Corrected sentences go to `outputs/gector/` at the repository root, with the other systems.

## Install

The official code needs Python 3.7 with AllenNLP 0.8.4 and PyTorch 1.10, which does not install on current machines. We load the official weights with [gotutiyan/gector](https://github.com/gotutiyan/gector), a PyTorch port that maps the official parameters and tag vocabulary onto its own model class. The setup script creates `gector/.venv` with Python ≥ 3.11, installs the port pinned to a reviewed revision, and downloads the files above:

```bash
PYTHON=python3.12 bash gector/setup.sh
```

## Correct sentences

From the repository root:

```bash
gector/.venv/bin/python gector/predict.py --input data/dev.src --output outputs/gector/dev.tgt
gector/.venv/bin/python gector/predict.py --input outputs/dev30.src --output outputs/gector/dev30.tgt
gector/.venv/bin/python gector/predict.py --input data/test.conll14.src --output outputs/gector/conll14.tgt
gector/.venv/bin/python gector/predict.py --input data/test.bea19.src --output outputs/gector/bea19.tgt
```

Defaults are Grammarly's recommended inference settings for this model: `--keep_confidence 0.2` (bias towards `$KEEP`, so fewer but more precise edits) and `--min_error_prob 0.5` (skip edits and sentences the detector is unsure about), with up to 5 correction rounds (`--n_iteration`). The device is picked automatically (CUDA, then Apple MPS, then CPU). Sentences longer than 80 subwords are tagged up to that point and the rest is copied unchanged. The full dev set takes about 5 minutes on an Apple M-series GPU.

`predict.py` works around two incompatibilities of the port with current libraries: the checkpoint stores CUDA tensors, so it is loaded onto the CPU first, and transformers 5 drops `max_length` from the config, so it is set again after loading.

This folder deliberately has no `__init__.py`: as a plain directory it never shadows the installed `gector` package.

## Results

ERRANT on the W&I+LOCNESS dev set:

| Sentences | Output | Precision | Recall | F₀.₅ |
| --- | --- | ---: | ---: | ---: |
| All 4,384 | `outputs/gector/dev.tgt` | 63.8 | 35.0 | 54.8 |
| First 30 | `outputs/gector/dev30.tgt` | 71.4 | 34.3 | 58.7 |
