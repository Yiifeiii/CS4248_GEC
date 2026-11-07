# GRECO fine-tuning on stage 1

This documents how the released GRECO scorer is fine-tuned on the stage-1 curriculum data, how that data is prepared, and how the learning rate is tuned. Stage 2 is not run, for the reason in section 1.

**Status.**

| Part | State |
|---|---|
| Stage-1 split and sample (`prepare_stage1.py`) | Implemented and run on the local data |
| Labelled records (`build_labels.py`) | Implemented. Tested on hand-checked cases and on a small real file. |
| Candidates from the `s1s2` checkpoints (step P2) | **Not run.** The checkpoints are not public. They need `hf auth login` with an account in the `CS4248-GEC` organisation. |
| Full label build, training, tuning, comparison | Not run |

## 1. Why stage 2 is skipped

GRECO's own training data is the W&I+LOCNESS training set. Qorib & Ng (2023), Section 5.1, say the model is trained on "the unique corrections of nine GEC systems" on that set. They report 34,308 sentences, which give 65,824 hypotheses after filtering. The paper also says NUCLE and FCE are used only for NeuQE, a different model, and no Lang-8 or synthetic data is used for GRECO's training. Source: <https://arxiv.org/html/2310.14947v1>, read through a page summary.

Our stage 2 is that same set: 33,943 pairs. The gap to 34,308 is 365 sentences, which are exactly the sentences our filters drop: 308 UNK-only, 53 over 128 tokens, 4 empty. Source: `data/curriculum/manifest.json`.

Stage 2 would show the scorer data it has already learned from. Any change in score could not be attributed to new information, so stage 2 is skipped.

## 2. Stage 1 overlaps the W&I training data

Stage 1 includes the errorful W&I sentences: `wi.A` 8,160, `wi.B` 9,124 and `wi.C` 5,397, a total of 22,681 of 561,338 pairs (4.0%). That total equals the number of errorful sentences in the W&I training set. The same reason for skipping stage 2 applies to those pairs.

**Decision.** The main run uses stage 1 with the `wi.*` pairs removed, which leaves 538,657 pairs. These are FCE, Lang-8 and NUCLE sentences. The full stage 1, which matches the T5 definition, is a secondary run.

Other facts about the dev splits:

- The paper chose hyperparameters on the BEA-2019 dev set. Our `dev_tune` and `dev_test` come from the W&I dev set, so scores are not directly comparable with the paper's tables.
- 35 stage-1 sentences overlap `dev.src`, with at most 13 words each (from `manifest.json`). They are kept, and the count is recorded here.

## 3. Glossary

- **Original.** The released GRECO checkpoint, `models/greco/checkpoint.bin`, from upstream commit `8e509aa`. It is the starting point and the baseline. Tuning is only worth reporting if it beats the original.
- **Stage 1 (clean).** BEA-2019 errorful pairs from FCE, Lang-8 and NUCLE: 538,657 pairs.
- **Hypothesis.** One candidate correction of a source sentence. GRECO scores hypotheses, not sentence pairs.
- **Group.** The four hypotheses for one source sentence. One JSON line holds one group. The loader needs every group to have the same size, and the paper uses four.
- **Sentence target.** The F0.5 of a hypothesis against the gold correction, between 0 and 1. It is stored as `labels[0][0]` and is what the scorer's score head is trained to predict.
- **Word label.** 1 if a token should be kept, 0 if it should change. Source labels say which source words the gold corrects. Hypothesis labels say which hypothesis words differ from the gold.
- **Gap label.** One per gap between hypothesis words, plus one at the end. 0 means a word must be inserted in that gap.
- **Candidate pool.** The fixed set of BART, T5 and GECToR hypotheses on the dev sets that `greco_setup/combine.py` searches over. The pool is the same for every model, so only the scorer changes.
- **dev_tune / dev_test.** The 2,000-sentence and 2,384-sentence W&I dev splits. Every choice uses dev_tune. dev_test is scored once.
- **F0.5 for reporting.** Correction-level ERRANT F0.5, multiplied by 100, as in the T5 work. M2 F0.5 is a secondary check.

## 4. Data and models

| Item | Made by | Size |
|---|---|---|
| Stage 1 (full) | `data/curriculum/stage1.*`, from `training/build_data.py` | 561,338 pairs |
| Stage 1 (clean) | `prepare_stage1.py`, step P1 | 538,657 pairs |
| Stage 1 sample | `prepare_stage1.py`, step P1 | 50,000 sources, seed 4248 |
| Candidates | BART and T5 `s1s2` n-best, step P2 | k = 5 each per source |
| Labelled groups | `build_labels.py`, step P3 | up to 50,000 groups of 4 |
| dev_tune / dev_test | `data/curriculum/dev_*.*` | 2,000 and 2,384 sentences |
| Original | `models/greco/checkpoint.bin` from `greco_setup/setup.sh` | record its sha256 |

The stage-1 files are git-ignored and rebuilt by `training/build_data.py` (see `training/README.md`). The outputs of `greco_tuning/` go to `data/greco/`, which is also git-ignored.

## 5. Candidates and labels

**Candidates (step P2).** The candidate models are the published `s1s2` checkpoints, `CS4248-GEC/gec-bart-base-s1s2` and `CS4248-GEC/gec-t5-base-s1s2`. The plain Hugging Face models (`facebook/bart-base`, `google-t5/t5-base`) are not GEC-trained and are not used.

These models were trained on the stage-1 corpora, so their outputs on the sampled sources are closer to gold than outputs on unseen text. The paper does not state, in the text I read, which data its nine systems were trained on, so this bias cannot be compared with the paper directly.

**Choosing four per source (in `build_labels.py`).** Candidates are merged by rank, alternating between BART and T5: BART rank 1, T5 rank 1, BART rank 2, and so on. A candidate is dropped if it is empty, duplicates an earlier one, equals the source (no edit), equals the gold (already perfect), or, when a tokenizer is given, is too long for the loader (section 6, P3). The first four usable candidates form the group. A source with fewer than four is skipped.

**Labels (ERRANT).** Every edit is found with ERRANT, which is the same edit matching the F0.5 metric uses. For one hypothesis:

- *Source labels:* ERRANT edits from the source to the gold. A source token inside a gold edit gets 0.
- *Hypothesis labels:* ERRANT edits from the hypothesis to the gold, meaning the changes the hypothesis still needs. A hypothesis token inside one of those edits gets 0.
- *Gap labels:* an edit from the hypothesis to the gold that inserts words at position j gives gap j a 0.
- *Masks:* the positions above that have an edit, True where the label is 0. These are used for `--edit_weight`.
- *Sentence target:* F0.5 of the hypothesis's edits (source → hypothesis) matched against the gold edits (source → gold), with the same edit key of span and replacement. A perfect hypothesis scores 1, and a hypothesis that matches no gold edit scores 0.

ERRANT parses with spaCy. Each source, gold and hypothesis must come out with exactly the same tokens as its whitespace split. If a sentence does not, that hypothesis is skipped and counted in the output statistics.

**Record layout.** Checked against the upstream loader at `8e509aa`, with the real tokenizer (section 8):

```json
[
  {"source": ["He","eat","apple","."], "hyp": ["She","eats","apple","."],
   "labels": [[0.5556], [1,0,1,1], [0,1,1,1]],
   "gap": [1,1,1,1,1],
   "masks": [[false,true,false,false], [true,false,false,false]],
   "gap_mask": [false,false,false,false,false]},
  ...three more records of the same source...
]
```

The loader's `labels[0][0]` is the first token's label, which the score head is regressed to. `labels[1]` is the source and `labels[2]` the hypothesis. A check with the upstream `TrainDataset` read this group as 4 hypotheses, with `[CLS]` at 0.556, `He eat apple .` as `1 0 1 1` and `She eats apple .` as `0 1 1 1`.

## 6. Process

Step numbers match `greco_tuning/README.md` and the commands in section 10.

**P0. Get the original.** `bash greco_setup/setup.sh` in the GRECO environment. Record the sha256 of `models/greco/checkpoint.bin`.

**P1. Stage 1: remove W&I, draw the sample.** Writes `data/greco/stage1.clean.*`, `stage1.sample.*` and `stage1.manifest.json`. Run in the repo's Python 3 environment. *Done.*

**P2. Candidates.** Generate the n-best lists on the cluster or a GPU, after `hf auth login`. The sample is 50,000 sources, so this is the step with the largest compute cost. *Not run.*

**P3. Label.** Build the groups with `build_labels.py`, with the DeBERTa tokenizer. The tokenizer drops any candidate whose source and hypothesis together are longer than 200 subword tokens, because the loader would truncate it and misalign the labels. The next candidate is used in its place. Run in the label environment (section 10). *Implemented and tested.*

## 7. Training and tuning

**Starting point.** Every run starts from the original checkpoint, through `--pretrained_path`, which loads the state dict onto the DeBERTa-v3-large backbone that `--model_name_or_path` builds.

**Fixed settings**, from the upstream retraining command: `word_dropout 0.25`, `alpha 1`, `edit_weight 2.0`, `rank_multiplier 5`, batch 32 with gradient accumulation 4. Set `--gamma 0.2` explicitly. The upstream code references `self.d` and `self.bucket_size` in places that `__init__` does not set, so check the first run for an `AttributeError`.

**Tuned setting: learning rate.** Continuing from the original with upstream's 2e-5 may move the scorer too far. The grid is `{5e-6, 1e-5, 2e-5}`. This grid is proposed, not tested. The number of epochs depends on how long one epoch takes, so time one epoch before fixing it. Proposed: 1 to 2 epochs.

**Selection.** For each learning rate, save a checkpoint every epoch. Score each checkpoint on the dev_tune pool with `combine.py` and collect the output. Use ERRANT F0.5 of the combined output as the selection metric, not the training loss.

**Noise band.** Rerun the best learning rate with seeds 42, 123 and 7. The spread across seeds is the band. A learning rate is chosen only if it beats the original's dev_tune F0.5 by more than the band. Otherwise the original is kept.

**Decision rule.** The default is the original. If no learning rate beats it by more than the band, stage-1 fine-tuning does not help GRECO on this pool. That is a valid result.

## 8. Comparison

Compare **original** and **stage 1 (clean)**, optionally with **stage 1 (full)**. Report:

- **Metric:** ERRANT F0.5 of the combined output on the shared pool. M2 F0.5 from `training/metrics.py` as a check.
- **Uncertainty:** a paired bootstrap over dev_tune sentences gives a 95% interval for each difference. A difference counts only if the interval excludes zero and it is larger than the seed band.
- **dev_test:** score the original and the chosen model once each, after the choices are fixed.

## 9. Open decisions and known limits

- **In-sample candidates.** The `s1s2` checkpoints were trained on the stage-1 corpora. Section 5 explains the effect.
- **Cost.** Step P2 generates 10 candidates for each of 50,000 sources, and training runs on up to 50,000 groups of four. Neither has been timed. Measure both before choosing the sample size or the number of epochs.
- **Full or clean stage 1.** The clean run is the main one. The full run is a secondary comparison.
- **Label definitions.** The sentence target and the word labels follow the upstream code's use of them. They are not verified against the paper's own labelling procedure, which is not in the text I could read.

## 10. Where the code is and how to run it

| Step | File | Status | Environment |
|---|---|---|---|
| Original checkpoint | `greco_setup/setup.sh` | Exists | GRECO (Python 3.10) |
| Stage 1 split and sample (P1) | `greco_tuning/prepare_stage1.py` | Implemented, run | repo Python 3 |
| Candidates (P2) | `bart/generate_nbest.py`, `bart/export_candidates.py` | Exists, not run with the `s1s2` checkpoints | GPU, with HF login |
| Labels (P3) | `greco_tuning/build_labels.py` | Implemented, tested | label environment (below) |
| Training | upstream `train.py` with `--pretrained_path` | Exists upstream | GRECO |
| Combine and collect | `greco_setup/combine.py`, `greco_setup/prepare.py` | Exists | GRECO |
| ERRANT and M2 scores | `greco_setup/evaluate_errant.py`, `training/metrics.py` | Exists | ERRANT env and repo |
| Tuning loop and bootstrap | `greco_tuning/tune.py` | Not written | — |

**Label environment.** `build_labels.py` needs ERRANT, `en_core_web_sm`, and a fast DeBERTa tokenizer. The tested setup was:

```bash
python3 -m venv .venv-label
.venv-label/bin/pip install errant "transformers==4.46.3" sentencepiece protobuf torch
.venv-label/bin/python -m spacy download en_core_web_sm
```

The tests passed in this environment. `transformers` must be pinned, because newer releases failed to load the DeBERTa tokenizer here.

## 11. How to reproduce

Run from the repo root. Steps P0 and P2 need the GRECO environment and the cluster.

```bash
# P0: original checkpoint (GRECO environment)
bash greco_setup/setup.sh

# P1: stage 1 without W&I, plus a 50,000-sentence sample (tested)
python3 -m greco_tuning.prepare_stage1 --sample 50000 --seed 4248

# P2: candidates from the s1s2 checkpoints (cluster or GPU, after hf auth login)
for m in bart-base:bart t5-base:t5; do
  name=${m%%:*}; sys=${m##*:}
  python3 bart/generate_nbest.py --model CS4248-GEC/gec-${name}-s1s2 \
    --input data/greco/stage1.sample.src \
    --output outputs/greco-s1/${sys}.nbest.jsonl --num_beams 10 --num_return 10 --batch_size 64
  python3 bart/export_candidates.py --nbest outputs/greco-s1/${sys}.nbest.jsonl \
    --out outputs/greco-s1/${sys}.tgt --k 5
done

# P3: labelled groups (label environment, with the DeBERTa tokenizer)
.venv-label/bin/python -m greco_tuning.build_labels \
  --source data/greco/stage1.sample.src --gold data/greco/stage1.sample.tgt \
  --system bart=outputs/greco-s1/bart.tgt --system t5=outputs/greco-s1/t5.tgt \
  --k 5 --tokenizer microsoft/deberta-v3-large --max-length 200 \
  --out data/greco/stage1.train.jsonl

# Upstream GRECO code at the pinned commit, used for training
git clone https://github.com/nusnlp/greco external/greco
(cd external/greco && git checkout 8e509aaacf4e5ec3276d3427371471fd82a7f58b)

# Section 7: learning-rate grid (GRECO environment, run from external/greco)
for LR in 5e-6 1e-5 2e-5; do
  python train.py --do_train \
    --model_name_or_path microsoft/deberta-v3-large \
    --pretrained_path models/greco/checkpoint.bin \
    --output_dir ${CKPT_ROOT}/greco.s1.lr${LR}.seed42 \
    --learning_rate ${LR} --word_dropout 0.25 --alpha 1 \
    --edit_weight 2.0 --rank_multiplier 5 --gamma 0.2 \
    --per_device_train_batch_size 32 --gradient_accumulation_steps 4 \
    --num_train_epochs 2 --save_strategy epoch --seed 42 \
    --data ${REPO}/data/greco/stage1.train.jsonl --data_mode hierarchical
done

# Scoring and collection: greco_setup/combine.py and prepare.py, as in the GRECO pilot (to write: tune.py)
```

`--num_train_epochs 2` is a placeholder until the timing check in section 9 is done. The `train.py` command is the upstream retraining command with `--pretrained_path` added.
