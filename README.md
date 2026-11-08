# CS4248: Grammatical Error Correction

Completed experiments: [BART/T5 recipe and matched-order curriculum comparisons](experiments/curriculum-comparison.md).

Completed experiment: [single-stage BART/T5 controls and curriculum comparison](experiments/mixed-control/README.md).

This project investigates whether combining alternative corrections from multiple grammatical error correction (GEC) systems improves precision, recall, and coverage of rare error types. The planned approach pools the top-k hypotheses from BART and T5 with GECToR's best correction, then combines their edits using GRECO.

## Current status

A standalone [manual GRECO combination workflow](greco_setup/README.md) accepts manually entered BART/T5 sentences or edits. It includes conflict-aware beam search, an adapter for the official GRECO scorer, examples, and setup instructions. Candidate generation can be tested without model dependencies; pretrained inference requires a separate Python 3.10 environment and downloaded weights.

The repository also contains datasets, prepared source/target sentence pairs, an M2-to-parallel conversion script, an error statistics script, and the bundled NUS M² scorer. [Two-stage fine-tuning](training/README.md) of `facebook/bart-base` and `t5-base` is set up: stage 1 on all four corpora (errorful sentences), stage 2 on W&I train including correct sentences, with jobs for the [SoC cluster](cluster/README.md). Agreement features and the proposed ablations remain planned work.

## Repository structure

| Path | Purpose |
| --- | --- |
| `data/` | Training, development, and test sentences, gold M2 annotations for scoring, and dataset licenses |
| `m2_to_parallel.py` | Convert M2 annotations into original/corrected sentence pairs (used to build `data/`) |
| `error_type_stats.py` | Count sentences and edits by error type (needs the original corpus M2 files, see below) |
| `error_type_counts.csv` | Existing error-type counts by corpus |
| `m2scorer/` | Bundled NUS MaxMatch scorer, supporting scripts, examples, and license |
| `bart/` | BART n-best generation, export to the shared output format, n-best statistics |
| `t5/` | T5 top-k generation and retokenization of its output |
| `gector/` | Grammarly's pretrained GECToR ([setup and usage](gector/README.md)) |
| `greco_setup/` | GRECO edit combination ([setup and usage](greco_setup/README.md)) |
| `outputs/` | System predictions in the shared format, see [Outputs](#outputs) |
| `training/` | Two-stage BART/T5 fine-tuning: data build and checks, training, evaluation ([details](training/README.md)) |
| `cluster/` | Slurm jobs and setup for the NUS SoC compute cluster ([details](cluster/README.md)) |
| `tests/` | Unit tests for the GRECO combination and training data code |

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

The repository tracks `*.m2` through Git LFS. In the inspected checkout, the dataset M2 files are pointer files rather than downloaded annotations. With Git LFS installed, retrieve them before conversion or gold evaluation:

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

The scorer is legacy Python 2 code and is not directly compatible with Python 3. In a compatible Python 2 environment, run from the repository root:

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

### 4. Breakdown of work

| Responsibility | Allocation |
| --- | --- |
| Fine-tune baseline models and lead comparative evaluation | Haleema |
| Implement and test enhancements and ablations | Raeeda, Yifei |
| Lead report writing and consolidate results | One member |
| Contribute report sections and the final presentation | All members |

Work will proceed concurrently, with report writing alongside model development.

### 5. Schedule

| Dates | Planned work |
| --- | --- |
| 1–7 October | Run baseline models |
| 8 October–4 November | Develop and tune enhanced n-best models; write the report in parallel |
| 5–12 November | Final model evaluation and presentation video filming |

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
