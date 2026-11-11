# Running the project

Run commands from the repository root. The project does not have one single
model: it contains BART, T5, GECToR, GRECO/ESC combination workflows, and
training/ablation pipelines. Start with the smallest workflow that matches the
task.

## 1. Install a local environment

For data preparation, tests, and the training utilities:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m spacy download en_core_web_sm
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
| Source code and tests | `src/gec/`, `tests/` | yes |
| Reusable configs | `configs/training/`, `configs/t5_tuning/` | yes |
| Curated experiment scores and notes | `experiments/` | yes |
| Predictions and n-best files | `outputs/` or `artifacts/` | no |
| Checkpoints and downloaded models | `checkpoints/`, `models/` | no |
| Slurm logs | `logs/` | no |
| Local environments and scratch | `.venv*`, `check/`, `tmp/`, `output/` | no |

## 9. Use private Hugging Face models

Do not commit private model weights, access tokens, or Hugging Face cache
directories. A repository can safely contain the private model identifier, for
example `your-team/gec-bart-stage2`, while each user authenticates locally.

### Interactive development

Install the Hub CLI and authenticate once on the machine that will run the
model:

```bash
python -m pip install -U huggingface_hub
hf auth login
hf auth whoami
```

The token is stored by the Hugging Face client, not in Git. Then download the
team checkpoints into the normal cache:

```bash
bash scripts/fetch_hf_models.sh \
  YOUR_ORG/gec-bart-base-s1s2 \
  YOUR_ORG/gec-t5-base-s1s2
```

Use the private repository ID as the `init` or model name in a config, or use
environment expansion to keep the team-specific name out of a shared config:

```bash
export HF_BART_MODEL="YOUR_ORG/gec-bart-base-s1s2"
export HF_T5_MODEL="YOUR_ORG/gec-t5-base-s1s2"
```

The training code expands `${VARS}` in `init`, so a config can contain
`"init": "${HF_BART_MODEL}"`. You can also pass a local snapshot path when a
job must run offline. The account used for
`hf auth login` must have at least read access to the private organization and
repositories.

### Cluster jobs and CI

For a cluster or CI job, create a read-only Hugging Face token with the minimum
required repository permissions, store it in the scheduler/CI secret store,
and expose it only at runtime:

```bash
export HF_TOKEN="$HUGGINGFACE_READ_TOKEN"
export HF_HOME="/path/to/private/hf_cache"
```

Run the download on a network-enabled node before setting offline mode. The
training jobs can then use the cached snapshot without receiving the token.
Never put `HF_TOKEN=...` in a config, shell script, Slurm submission command,
README, notebook, or commit. If a token is ever exposed, revoke it immediately
and create a replacement.

### Reproducibility

Record the private model repository ID, revision or commit SHA, and a checksum
of the selected model in the experiment metadata. Do not record the token or
copy the private weights into this repository. Team members who lack access
should receive access through the Hugging Face organization rather than through
an exported credential.
