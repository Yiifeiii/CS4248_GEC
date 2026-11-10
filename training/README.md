# Two-stage fine-tuning of BART and T5

See [BART hyperparameter experiments](../experiments/ablations/README.md) for grouped training/beam-search scripts, completed F-scores, and reproduction commands.

For the nine BART-only stage-2 hyperparameter ablations, see the
[configuration grid and launch instructions](configs/bart-ablations-20261105/README.md)
and the [submitted experiment log](../experiments/training/bart-ablations-20261105/README.md).

We fine-tune `facebook/bart-base` and `google-t5/t5-base` from their pretrained (not GEC-trained) weights in two stages:

1. **Stage 1:** all four BEA-2019 training corpora (FCE, Lang-8, NUCLE, W&I+LOCNESS), errorful sentences only.
2. **Stage 2:** W&I train, including its correct sentences, starting from the best stage-1 checkpoint.

The stage-1 checkpoints are kept, so stage 1 alone (`*.s1`) can be compared with stage 1 followed by stage 2 (`*.s1s2`).

How to run the jobs on the SoC cluster is in [`cluster/README.md`](../cluster/README.md).

The results of every run so far, the findings, and the trained models on Hugging Face (`CS4248-GEC/*`) are in [`experiments/training/README.md`](../experiments/training/README.md).

## From scratch, end to end

1. **On your laptop:** set up an environment and build the data (see [Data](#data-datacurriculum)):
   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate      # 3.11 or 3.12; 3.13 lacks lib2to3
   pip install torch -r requirements-train.txt && python -m spacy download en_core_web_sm
   git lfs install && git lfs fetch origin 3e50997    # the original W&I train M2 files
   python -m training.build_data && python -m training.check_data
   ```
2. **On the cluster:** clone, set up the environment, download the base models and copy `stage1.*` over ([one-time setup](../cluster/README.md#one-time-setup)).
3. **Run** the pilots, then the stage-1 and stage-2 jobs ([submitting jobs](../cluster/README.md#submitting-jobs)).
4. **On your laptop:** copy back the results and the best checkpoints ([copying results back](../cluster/README.md#copying-results-back)). Commit `experiments/training/`.

## Data (`data/curriculum/`)

Build and verify the data from the repository root:

```bash
python -m training.build_data     # needs transformers, internet (tokenizers) and `git lfs fetch origin 3e50997`
python -m training.check_data     # exits non-zero if any check fails
```

| File | Contents | Pairs | In git |
|---|---|---:|---|
| `stage1.{src,tgt,corpus}` | `data/train.*`, as released: the errorful sentences of FCE train+dev, Lang-8, NUCLE and W&I A/B/C, shuffled (seed 4248). `.corpus` names each line's source corpus | 561,338 | no (78 MB; rebuild) |
| `stage2.{src,tgt,corpus}` | W&I A/B/C train: 22,681 errorful + 11,262 correct sentences, shuffled | 33,943 | yes |
| `dev_tune.{src,tgt,m2,ids}` | 2,000 random sentences of W&I+LOCNESS dev, used to choose checkpoints | 2,000 | yes |
| `dev_test.{src,tgt,m2,ids}` | the other 2,384, used to report results | 2,384 | yes |
| `manifest.json` | every count, filter, drop count and sha256 hash | | yes |

**Where stage 1 comes from:** `data/train.src/.tgt` (561,535 lines) is exactly the 7 per-corpus files of commit `3e50997` joined end to end, verified by sha256. In order, with line counts:

| Corpus | Lines |
|---|---:|
| FCE train | 17,715 |
| FCE dev | 1,370 |
| Lang-8 | 498,360 |
| NUCLE | 21,353 |
| W&I A | 8,194 |
| W&I B | 9,138 |
| W&I C | 5,405 |

FCE dev is training data here. Our evaluation set is the separate W&I+LOCNESS dev set, and training on all of FCE is standard BEA-2019 practice.

**Where stage 2 comes from:**
- The provided `m2_to_parallel.py` (unchanged, annotator 0) converts the original W&I A/B/C train M2 files, without `--erroneous_only`, at 34,308 sentences.
- The errorful pairs are exactly the W&I lines of `data/train.*`.
- LOCNESS has no training split; it is only in dev and test.

### Cleaning

The defaults are the minimum that training needs:
- **Over 128 T5 tokens:** 151 pairs dropped in stage 1 and 53 in stage 2. Truncating them would cut the source and target at different points. The same pairs are dropped for BART, so both models see identical data.
  - This is 0.03% of the 561,489 non-empty stage-1 pairs, and 0.16% of the 33,943 stage-2 pairs left after cleaning.
  - The filter applies only to training data. The dev sets are not filtered, so their over-limit pairs are still scored: 2 of 2,000 in `dev_tune` and 5 of 2,384 in `dev_test`.
  - So the cost is a small loss of long training examples, not of evaluation data. The effect on scores has not been measured.
  - Counted with the same rule as `build_data.py`. The tokenizer used (from `vennify/t5-base-grammar-correction`) reproduces the 151 stage-1 figure exactly, which supports that it matches the T5 tokenizer used in training.
- **Empty target** (a whole-sentence deletion such as "now ," → ""): 46 dropped in stage 1 and 4 in stage 2.
- **Only edit is `UNK`:** 308 dropped in stage 2. An error was marked but not corrected, so these "correct" targets still contain the error.

Three more filters are opt-in, for ablations. Builds with them must go to another directory, e.g. `--out data/curriculum-clean`:
- `--normalize-quotes`: Lang-8 writes double quotes PTB-style as `` and ''. No Lang-8 line contains `"`, while dev and test use `"`.
- `--ratio-filter`: drops 234 pairs whose target length is far from the source length.
- `--dedupe`: drops exact repeats (2,321 in stage 1).

Training sources that also appear in dev or test (listed in `manifest.json`) are short boilerplate ("Dear Sir ,") and W&I story prompts that every essay on that prompt begins with ("Michael closed the door and knew at that moment he had made a mistake ."). There is no overlap with CoNLL-14.

## Models

Prefixes:
- **BART:** inputs are used as they are.
- **T5:** each source gets the prefix `gec: `.

Both models read and write text tokenized like the corpora (`It 's`, `do n't`), so no retokenization is needed.

T5's vocabulary can't encode 45 characters in our data, including `` ` ^ ~ < { } `` and accented letters, and NFKC-normalizes others (`…` → `...`). So "São" comes back as "So". `training/common.py:restore_from_source` aligns T5's output with the source and copies the original tokens back where the output simply copied them. `check_data` confirms that this reproduces every dev and stage-2 sentence; 316 of them would differ without it.

`training/configs/` has one config per stage and model. `init` is the starting weights; stage 2 starts from `${CKPT_ROOT}/<model>.s1/best`.

| | BART s1 | T5 s1 | BART s2 | T5 s2 |
|---|---|---|---|---|
| Learning rate (AdamW, linear decay) | 5e-5 | 1e-4 | 1e-5 | 3e-5 |
| Warmup steps | 1,000 | 1,000 | 100 | 100 |
| Batch (sentences) | 64 × 2 accumulation = 128 | same | 32 | 32 |
| Epochs | 3 (≈13k steps) | 3 | up to 5, early stopping after 2 evaluations without improvement | same |
| ERRANT on dev_tune | every 2,000 steps, first 1,000 sentences, greedy | same | every epoch, all 2,000, beam 5 | same |

**Shared settings:**
- weight decay 0.01, label smoothing 0.1, dropout 0.1, seed 42
- bf16 on GPU, batches grouped by length
- dev_tune loss and a checkpoint every 1,000 steps in stage 1 and every 500 in stage 2, keeping the last 2

The best checkpoint is the one with the highest dev_tune ERRANT F0.5. BART generation always sets `no_repeat_ngram_size=0`, because bart-base inherits 3 from its summarization defaults, which would forbid copying repeated phrases.

The epoch counts are starting values. Run the 200-step pilot first (`cluster/pilot.sh`) and adjust them to the measured speed.

### Longer runs and the matched control (`*.long.*` configs)

In the first runs, every best checkpoint was the final step, so the `long` configs train for longer with the same hyperparameters:
- **`long.s1`:** stage 1 for 8 epochs instead of 3 (≈35k steps).
- **`long.s1s2`:** stage 2 for 10 epochs instead of 5, from `long.s1/best`. It has no early stopping, so it always runs 10,610 steps (1,061 per epoch).
- **`long.s1cont`, the control:** continues `long.s1/best` with exactly the stage-2 recipe, but on stage-1 data. It uses the same learning rate, warmup, batch of 32 and 10,610 steps (`max_steps` in the config), and runs ERRANT every 1,061 steps.

Ten stage-2 epochs are only 340k sentences, about 0.6 of one stage-1 epoch. Comparing `long.s1s2` with `long.s1cont` therefore separates the effect of W&I data from the effect of simply training longer. Both pick their best checkpoint the same way.

## Scripts

| Script | Purpose |
|---|---|
| `build_data.py` | build `data/curriculum/` (see above) |
| `check_data.py` | verify a build: counts, hashes, provenance, dev partition, tokenizer round trips |
| `finetune.py` | train one stage from a config; resumes automatically from the last checkpoint |
| `evaluate.py` | decode dev_tune, dev_test, CoNLL-14 or BEA-19 test with a checkpoint; ERRANT (primary) and M2 scores |
| `common.py` | shared helpers: prefixes, T5 character restoring, batched decoding |
| `metrics.py` | ERRANT via `greco_setup/evaluate_errant.py` and the bundled M2 scorer |

`finetune.py` writes these to `${CKPT_ROOT}/<run>/`:
- `checkpoint-*/`
- `best/`: model, tokenizer, `best.json` and `dev_tune.pred`
- `final/`
- `errant_eval.json`, `trainer_state.json` and `run_info.json`: config, git commit and data-manifest hash

`cluster/train.sh` copies the small files, the scores and the predictions to `experiments/training/<run>/`.

For n-best lists for GRECO, `bart/generate_nbest.py --model <checkpoint>` works for both models. It adds the T5 prefix and restores T5's lost characters automatically; see `cluster/nbest.sh`.

### Local smoke test

On any machine (CPU or Apple GPU), this takes a few minutes:

```bash
export CKPT_ROOT=/tmp/gec-smoke HF_HUB_OFFLINE=0
python -m training.finetune --config training/configs/bart-base.stage1.json --output-dir /tmp/gec-smoke/bart \
    --max-steps 12 --train-limit 500 --dev-limit 16 --errant-steps 6
python -m training.evaluate --model /tmp/gec-smoke/bart/best --split dev_tune --limit 8 --out /tmp/gec-smoke/eval
```

The real batch of 128 sentences per step is slow on a laptop. For a quicker check, copy the config and lower `batch_size`.

Unit tests: `python -m unittest tests/test_training.py`.
