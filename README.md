# CS4248: Grammatical Error Correction

This project investigates whether combining alternative corrections from multiple grammatical error correction (GEC) systems improves precision, recall, and coverage of rare error types. The planned approach pools the top-k hypotheses from BART and T5 with GECToR's best correction, then combines their edits using GRECO.

**Main result:** combining our fine-tuned BART and T5 with GECToR using GRECO and ESC edit scores reaches **77.24 ERRANT F0.5 on BEA-2019 test** and **66.14 M2 F0.5 on CoNLL-2014 test**, against 71.57 and 64.03 for the best single system (GECToR). See the [headline results](results/README.md), the [GRECO + ESC write-up](greco_esc/README.md), and the [experiment index](docs/experiments.md) for every study.

## Run the project

Setup, tests, data, inference, evaluation, combination and cluster training are all in the runbook, [`docs/running.md`](docs/running.md). Run every command from the repository root.

## Use the trained models

Our five trained BART and T5 models are public on Hugging Face, so no account or token is needed. [`configs/models/README.md`](configs/models/README.md) lists which model is which and how each scores, and [`docs/running.md`](docs/running.md#9-use-the-public-team-models) shows how to download and run them.

## Current status

All planned experiments are complete and scored:

- **Two-stage fine-tuning** of `facebook/bart-base` and `google-t5/t5-base` ([`training/`](training/README.md)): stage 1 on all four BEA-2019 corpora (errorful sentences), stage 2 on W&I train including correct sentences. 12 runs, including longer training and a matched control ([results](experiments/training/README.md)); the five main models are public on Hugging Face.
- **Studies of that training:** four single-stage controls against the curriculum ([results](experiments/four-controls/README.md)), 12 BART stage-2 hyperparameter ablations and a BART decoding study ([results](experiments/ablations/README.md)), and 21 T5 tuning runs ([results](t5_tuning/README.md)).
- **GECToR:** Grammarly's pretrained model, run on dev and both test sets ([`gector/`](gector/README.md)).
- **System combination:** GRECO with three voting schemes and k = 1, 2, 3, 5 hypotheses per system ([`greco_setup/`](greco_setup/README.md)), then GRECO with ESC edit scores ([`greco_esc/`](greco_esc/README.md)). Both are scored on CoNLL-2014 test and on BEA-2019 test through Codabench ([test results](experiments/test/README.md)).

## Repository structure

| Path | Purpose |
| --- | --- |
| `data/` | Training, development, and test sentences, gold M2 annotations for scoring, and dataset licenses |
| `m2_to_parallel.py` | Convert M2 annotations into original/corrected sentence pairs (used to build `data/`) |
| `error_type_stats.py` | Count sentences and edits by error type (needs the original corpus M2 files, see below) |
| `error_type_counts.csv` | Existing error-type counts by corpus |
| `m2scorer/` | Bundled NUS MaxMatch scorer, supporting scripts, examples, and license |
| `training/` | Two-stage BART/T5 fine-tuning: data build and checks, training, evaluation ([details](training/README.md)) |
| `configs/` | Training, ablation, decoding and tuning configs, and the released model IDs ([details](configs/README.md)) |
| `bart/` | n-best generation for BART and T5, export to the shared output format, n-best statistics |
| `t5/` | T5 output retokenization, and early pilot scripts on a pretrained T5 |
| `gector/` | Grammarly's pretrained GECToR ([setup and usage](gector/README.md)) |
| `greco_setup/` | GRECO edit combination, voting and tuning ([setup and usage](greco_setup/README.md)) |
| `greco_esc/` | GRECO + ESC combination: the main results ([details](greco_esc/README.md)) |
| `t5_tuning/` | T5 hyperparameter tuning: 21 runs, noise bands, sweeps and final dev_test scores ([results and how to rerun](t5_tuning/README.md)) |
| `experiments/` | Per-run scores, predictions and write-ups of every training study and the test sets |
| `results/` | [Summary of the headline results](results/README.md) |
| `docs/` | [Runbook](docs/running.md) and [experiment index](docs/experiments.md) |
| `outputs/` | System predictions in the shared format (see [Outputs](#outputs)); new files here are git-ignored |
| `cluster/` | Slurm jobs and setup for the NUS SoC compute cluster ([details](cluster/README.md)) |
| `scripts/` | Helper to download the released models |
| `tests/` | Unit tests for training, data, T5 tuning, the ablations and the GRECO combination |
| `requirements-train.txt`, `requirements-bart.txt` | Python dependencies for training/evaluation and for BART n-best generation |

## Data and formats

Training uses the provided BEA-2019 corpora: FCE, Lang-8, NUCLE, and W&I+LOCNESS. No additional training data or manual annotation is planned.

| File | Contents | Built from | Lines / sentences |
| --- | --- | --- | ---: |
| `data/train.src`, `data/train.tgt` | Training pairs, erroneous sentences only | FCE train + dev, Lang-8, NUCLE, W&I A/B/C train (`--erroneous_only`) | 561,535 |
| `data/dev.src`, `data/dev.tgt` | Development pairs, all sentences | W&I+LOCNESS `ABCN.dev.gold.bea19.m2` | 4,384 |
| `data/dev.m2` | Dev gold edits, for ERRANT | W&I+LOCNESS `ABCN.dev.gold.bea19.m2` | 4,384 |
| `data/test.bea19.src` | BEA-2019 test input (no gold; score on Codabench) | W&I+LOCNESS `ABCN.test.bea19.orig` | 4,477 |
| `data/test.conll14.src` | CoNLL-2014 test input | `official-2014.combined.m2` | 1,312 |
| `data/test.conll14.m2` | CoNLL-2014 gold edits (both annotators), for the M² scorer | `official-2014.combined.m2` | 1,312 |
| `data/licenses/` | FCE, W&I and LOCNESS licenses | | |
| `data/curriculum/` | Stage-1/stage-2 training data and the dev_tune/dev_test split, built by `training/build_data.py` | | |

Each `.src` line is an original, tokenized sentence; the corresponding `.tgt` line is its corrected version. Preserve line alignment when processing these files. The `.src` files of dev and CoNLL-2014 test are exactly the source sentences of their `.m2` files.

The original corpus folders (`fce/`, `lang8.bea19/`, `nucle/`, `wi+locness/`, `conll14/`) were removed once `data/` was built. They remain in the Git history, e.g. `git checkout 04eec95 -- fce lang8.bea19 nucle wi+locness conll14` restores them.

M2 files store source sentences and edits using zero-based token offsets. For example:

```text
S She go to school .
A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0
```

This edit replaces token 1 (`go`) with `goes`. The end offset is exclusive, and the final field identifies the annotator.

### Outputs

Every system writes its predictions to `outputs/<system>/<split>.tgt` in one shared format, which [GRECO combination](greco_setup/README.md#from-system-output-files) reads directly:

- **k consecutive lines per source sentence**, best hypothesis first: lines k·(i−1)+1 … k·i belong to source line i. GECToR has k = 1, so its file is aligned line by line with the source like `data/dev.tgt`.
- **Tokenized like the source** (`It 's`, `do n't`, `minds .`). T5 produces ordinary text, so `t5/retokenize.py` splits it with spaCy before writing.

| File | System | k | Lines |
| --- | --- | ---: | ---: |
| `outputs/dev30.src` | First 30 sentences of `data/dev.src` | | 30 |
| `outputs/gector/dev30.tgt` | GECToR | 1 | 30 |
| `outputs/gector/dev.tgt` | GECToR, full dev set | 1 | 4,384 |
| `outputs/bart/dev30.beam.tgt` | BART, beam search | 5 | 150 |
| `outputs/bart/dev30.diverse.tgt` | BART, diverse beam search | 5 | 150 |
| `outputs/t5/dev30.tgt` | T5, beam search | 5 | 150 |

`outputs/bart/*.nbest.jsonl` keep BART's raw n-best lists with model scores, from which `bart/export_candidates.py` writes the `.tgt` files and `bart/nbest_stats.py` computes statistics. GRECO results go to `outputs/greco/`.

### Retrieve M2 annotations

The repository tracks `*.m2` through Git LFS. Without Git LFS, a clone contains small pointer files instead of the annotations. With Git LFS installed, retrieve them before conversion or gold evaluation:

```bash
git lfs install
git lfs pull
```

Prepared `.src` and `.tgt` files already contain sentence text. Consult the dataset license files before using or redistributing the corpora.

## Existing tools

### Convert M2 to parallel text

The converter uses Python 3 and standard-library modules:

```bash
python3 m2_to_parallel.py --data data/dev.m2
```

This writes `.src` and `.tgt` files beside the input, replacing existing outputs with the same names. Multiple input files can follow `--data`.

- `--erroneous_only`: drop examples whose corrected sentence is unchanged.
- `--all_annotators`: emit a separate pair for each annotator instead of using only annotator `0`.

The converter applies edits while tracking token-position changes caused by insertions and deletions. It skips the `noop`, `UNK`, and `Um` edit categories.

### Inspect error distributions

`error_type_stats.py` counts sentences, sentences with errors, and edit categories for annotator `0`, excluding `noop` edits. It reads the original corpus M2 files, so restore them first (see [Data and formats](#data-and-formats)); `error_type_counts.csv` already holds its results. Before running it, update its hardcoded `base` path to this checkout and choose an appropriate JSON output path; its current output is `~/errstats.json`.

```bash
python3 error_type_stats.py
```

The script prints corpus summaries and writes JSON statistics. It does not currently generate `error_type_counts.csv`.

### Evaluate corrections with M²

The bundled scorer accepts tokenized predictions, one sentence per line, and gold M2 annotations. Predictions must follow the gold sentence order and tokenization conventions.

The scorer is legacy Python 2 code. The project's own evaluation (`training/evaluate.py`, via `training/metrics.py`) runs it under Python 3.11 or 3.12 by converting a temporary copy, so the bundled files stay unchanged. To run it by hand, use a Python 2 environment, from the repository root:

```bash
python2 m2scorer/scripts/m2scorer.py predictions.txt data/test.conll14.m2
```

It reports precision, recall, and F₀.₅ by default. F₀.₅ gives precision more weight than recall, making unnecessary corrections costly. See [the scorer README](m2scorer/README) for options, prerequisites, annotation conventions, and attribution.

## Project plan

### 1. Approach

Pretrained encoder-decoder models such as BART (Lewis et al., 2020) and T5 (Raffel et al., 2020) provide strong GEC baselines (Katsumata & Komachi, 2020). The project targets two limitations identified in the proposal: over-correction, where unnecessary rewrites lower precision under F₀.₅, and weak coverage of rare grammatical error types.

We plan to fine-tune BART and T5 on FCE, Lang-8, NUCLE, and W&I+LOCNESS, and incorporate the pretrained GECToR tagging model (Omelianchuk et al., 2020). Our hypothesis is that correct edits missed by a model's top-ranked output may survive in lower-ranked hypotheses.

The proposed pipeline is:

1. Establish individual BART, T5, and GECToR baselines and standard 1-best system combinations.
2. Retain the top-k hypotheses from BART and T5 for **k ∈ {1, 3, 5}**.
3. Explore diverse beam search to reduce near-duplicate candidates, comparing it with plain beam search.
4. Pool edits from these hypotheses with edits from GECToR's best correction.
5. Implement agreement features and combine candidate edits using GRECO (Qorib & Ng, 2023).
6. Compare the resulting corrections with the individual systems and standard 1-best combinations.

The proposal positions this work as an extension of system combination approaches such as ESC (Qorib et al., 2022) and GRECO, which use each system's 1-best correction. Prior work motivates using n-best hypotheses through SMT hypothesis combination (Hoang et al., 2016) and neural cross-comparison (Liu et al., 2021). Pooling n-best hypotheses across modern neural GEC systems is the proposed research direction; its novelty remains a claim to assess through the literature review.

### 2. Evaluation and ablations

We will evaluate on the provided test sets using ERRANT and the M² scorer, reporting precision, recall, and F₀.₅. ERRANT will support error-type analysis alongside overall evaluation.

| Ablation | Comparison |
| --- | --- |
| Number of hypotheses | k = 1, 3, 5 |
| Decoding strategy | Diverse vs. plain beam search |
| GECToR contribution | With vs. without GECToR |
| Agreement features | With vs. without agreement features |
| System combination | Proposed n-best pooling vs. standard 1-best combinations |

Results will include overall scores and breakdowns by grammatical error type to assess both unnecessary corrections and coverage of rare errors.

### 3. External code and data

Planned external components:

| Component | Intended source or role |
| --- | --- |
| BART | Meta AI; Hugging Face `facebook/bart-base` |
| T5 | Google; Hugging Face `t5-base` |
| GECToR | Grammarly; official pretrained checkpoint |
| ESC | NUS NLP; official code and released weights for comparison |
| GRECO | NUS NLP; official code and released weights for combination |
| ERRANT | Bryant et al.; error annotation and evaluation |

The group will implement fine-tuning, n-best generation, edit pooling, agreement features, and analysis scripts. These external model integrations are planned dependencies and are not bundled in this repository, apart from the existing M² scorer.

## References

- Bryant, C., Felice, M., & Briscoe, T. (2017). Automatic annotation and evaluation of error types for grammatical error correction. ACL, 793–805. https://doi.org/10.18653/v1/P17-1074
- Hoang, D. T., Chollampatt, S., & Ng, H. T. (2016). Exploiting n-best hypotheses to improve an SMT approach to grammatical error correction. IJCAI, 2803–2809.
- Katsumata, S., & Komachi, M. (2020). Stronger baselines for grammatical error correction using a pretrained encoder-decoder model. AACL-IJCNLP, 827–832. https://doi.org/10.18653/v1/2020.aacl-main.83
- Lewis, M., et al. (2020). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. ACL, 7871–7880. https://doi.org/10.18653/v1/2020.acl-main.703
- Liu, Z., Yi, X., Sun, M., Yang, L., & Chua, T.-S. (2021). Neural quality estimation with multiple hypotheses for grammatical error correction. NAACL-HLT, 5441–5452. https://doi.org/10.18653/v1/2021.naacl-main.429
- Omelianchuk, K., Atrasevych, V., Chernodub, A., & Skurzhanskyi, O. (2020). GECToR – Grammatical error correction: Tag, not rewrite. BEA, 163–170. https://doi.org/10.18653/v1/2020.bea-1.16
- Qorib, M. R., Na, S.-H., & Ng, H. T. (2022). Frustratingly easy system combination for grammatical error correction. NAACL-HLT, 1964–1974. https://doi.org/10.18653/v1/2022.naacl-main.143
- Qorib, M. R., & Ng, H. T. (2023). System combination via quality estimation for grammatical error correction. EMNLP, 12746–12759. https://doi.org/10.18653/v1/2023.emnlp-main.785
- Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140), 1–67.
- Raina, V., & Gales, M. (2023). Minimum Bayes' risk decoding for system combination of grammatical error correction systems. IJCNLP-AACL, Volume 2, 105–112. https://doi.org/10.18653/v1/2023.ijcnlp-short.12

For use of the bundled M² scorer, its README requests citation of Dahlmeier, D., & Ng, H. T. (2012), *Better Evaluation for Grammatical Error Correction*, NAACL-HLT.

## AI declaration

AI coding agents were used to help write and debug code in this repository. The
research ideas, experiment design and core parts of the work are the group's own,
and every result reported here comes from experiments the group designed and ran.
