# Fine-tuning experiments

This is a log of the BART/T5 fine-tuning runs: what each one tests, its config, and its results. The code and the procedure are in [`training/README.md`](../../training/README.md) and [`cluster/README.md`](../../cluster/README.md).

Each run's folder (`<run>/`) is written by `cluster/train.sh` and holds:
- the config
- `run_info.json`: git commit, data-manifest hash, GPU
- `trainer_state.json`: the loss curve
- `errant_eval.json`: dev_tune F0.5 during training
- `best_checkpoint.json`
- predictions and scores (`*.pred`, `*.scores.json`) for dev_tune and dev_test

The checkpoints are not in git. They are on the cluster's scratch disk; copies of `best/` can be pulled with the commands in `cluster/README.md`.

## Getting the trained models

The five main models are private repositories in the [CS4248-GEC](https://huggingface.co/CS4248-GEC) organization on Hugging Face. Each has a model card with its training details, scores and usage.

| Hugging Face repository | Run | dev_test F0.5 |
|---|---|---|
| [`CS4248-GEC/gec-bart-base-s1`](https://huggingface.co/CS4248-GEC/gec-bart-base-s1) | `bart-base.s1` | 43.8 |
| [`CS4248-GEC/gec-bart-base-s1s2`](https://huggingface.co/CS4248-GEC/gec-bart-base-s1s2) | `bart-base.s1s2` | 52.3 |
| [`CS4248-GEC/gec-t5-base-s1`](https://huggingface.co/CS4248-GEC/gec-t5-base-s1) | `t5-base.s1` | 46.4 |
| [`CS4248-GEC/gec-t5-base-s1s2`](https://huggingface.co/CS4248-GEC/gec-t5-base-s1s2) | `t5-base.s1s2` | 52.5 |
| [`CS4248-GEC/gec-t5-base-long-s1s2`](https://huggingface.co/CS4248-GEC/gec-t5-base-long-s1s2) | `t5-base.long.s1s2` | 53.2 (best) |

**Access:**
- You need to be a member of the organization. Log in once with `hf auth login`, using a token with read access.
- Every script in this repo then takes the repository name wherever it takes a checkpoint folder, for example:

  ```bash
  python -m training.evaluate --model CS4248-GEC/gec-t5-base-s1s2 --split dev_test --out check/
  python bart/generate_nbest.py --model CS4248-GEC/gec-bart-base-s1s2 ...
  ```
- The scripts add T5's `gec: ` prefix and restore the characters T5 drops, so the `.pred` files they write are ready for ERRANT, the M2 scorer or Codabench.
- **On the cluster**, jobs run offline: download first on a compute node with `hf download CS4248-GEC/<repo>` (`HF_HOME` must be the scratch cache), or copy the folder over.

**Verifying a model:**
1. **Same file:** `checkpoints.sha256` lists the sha256 of every run's `model.safetensors`, including the 7 runs that aren't uploaded. With the models in `checkpoints/<run>/best/`, run `cd checkpoints && shasum -a 256 -c ../experiments/training/checkpoints.sha256`. The Hugging Face page of each `model.safetensors` shows the same sha256.
2. **Same scores:** run `training.evaluate` on `dev_test` as above, then compare `check/dev_test.scores.json` with `<run>/dev_test.scores.json` here. Hardware other than the H100 with bf16 may decode a few sentences differently, so expect a match within a few tenths of a point.

**How results are reported:**
- **Primary:** ERRANT P/R/F0.5 on `dev_test`, the 2,384 W&I+LOCNESS dev sentences never used for choosing checkpoints, decoded with beam 5.
- **Full dev:** sums the ERRANT edit counts of dev_tune and dev_test. This makes it comparable to published BEA-19 dev scores.
- **Training-time F0.5 values** (`[errant]` log lines, `errant_eval.json`) are on part of dev_tune, sometimes decoded greedily. Use them only to compare checkpoints within one run.

## Runs

| Run | Config | Starts from | Tests |
|---|---|---|---|
| `bart-base.s1`, `t5-base.s1` | `<model>.stage1.json` | base model | stage 1: 3 epochs on BEA-19 train (errorful pairs) |
| `bart-base.s1s2`, `t5-base.s1s2` | `<model>.stage2.json` | `<model>.s1/best` | + stage 2: up to 5 epochs on W&I train, including correct sentences |
| `<model>.long.s1` | `<model>.long.stage1.json` | base model | does a longer stage 1 help? 8 epochs instead of 3 |
| `<model>.long.s1s2` | `<model>.long.stage2.json` | `<model>.long.s1/best` | does a longer stage 2 help? 10 epochs, no early stopping |
| `<model>.long.s1cont` | `<model>.long.stage1cont.json` | `<model>.long.s1/best` | control for `long.s1s2`: the same recipe and steps, but stage-1 data. Separates "W&I data helps" from "more training helps" |
| `bart-base.gtrecipe.s1` | `bart-base.gtrecipe.stage1.json` | `facebook/bart-base` | stage 1 with [gotutiyan/GEC-BART](https://github.com/gotutiyan/GEC-BART)'s hyperparameters (below) |
| `bart-base.gtrecipe.s1s2` | `bart-base.gtrecipe.stage2.json` | `bart-base.gtrecipe.s1/best` | our usual stage 2 on top of it |

### The gotutiyan-recipe comparison (`gtrecipe`)

**Why:** `gotutiyan/gec-bart-base` is trained on the same data as our stage 1, BEA-19 train without unchanged pairs (about 560k pairs). Yet it scores 47.2 ERRANT F0.5 on BEA-19 dev, against 44.4 for our `bart-base.s1`. The whole gap is precision: 56.0 vs 49.6.

**Question:** does copying their hyperparameters close that gap, and does stage 2 then add as much as before?

| | gotutiyan (their README) | `bart-base.s1` | `bart-base.gtrecipe.s1` |
|---|---|---|---|
| Learning rate | 3e-5 | 5e-5 | **3e-5** |
| Warmup steps | 500 | 1,000 | **500** |
| Batch | 64 × 4 = 256 | 64 × 2 = 128 | **64 × 4 = 256** |
| Epochs (updates) | 5 (about 11k) | 3 (13.2k) | **5 (about 11k)** |
| Label smoothing, max length, bf16, beam | 0.1, 128, yes, 5 | the same | the same |
| Optimizer | Adam (weight decay not stated) | AdamW, weight decay 0.01 | AdamW, weight decay 0.01 |
| Checkpoint selection | lowest loss on FCE dev | best dev_tune ERRANT F0.5 | best dev_tune ERRANT F0.5, every 1,000 steps |

**What still differs from gotutiyan:**
- **Checkpoint selection:** we keep ERRANT selection, so every run here is chosen the same way.
- **Weight decay:** theirs isn't stated.
- **Code:** they train with their own `accelerate` loop on transformers 4.23; we use HF `Seq2SeqTrainer` on 4.46.

`bart-base.gtrecipe.s1s2` uses exactly the `bart-base.stage2.json` recipe, so it compares directly with `bart-base.s1s2`.

Submit:

```bash
g1=$(sbatch --parsable -J bart-base.gtrecipe.s1 cluster/job.sbatch cluster/train.sh training/configs/bart-base.gtrecipe.stage1.json)
sbatch -J bart-base.gtrecipe.s1s2 --dependency=afterok:$g1 cluster/job.sbatch cluster/train.sh training/configs/bart-base.gtrecipe.stage2.json
```

## Results

ERRANT on dev_test (2,384 sentences, beam 5) and on the full dev set (4,384). Scores are percentages. **Best step** is the checkpoint chosen on dev_tune, out of the run's total steps.

| Run | dev_test P | dev_test R | **dev_test F0.5** | full dev P | full dev R | **full dev F0.5** | Best step |
|---|---|---|---|---|---|---|---|
| `bart-base.s1` | 49.0 | 30.8 | **43.8** | 49.6 | 31.1 | **44.4** | 13,155 / 13,155 |
| `bart-base.s1s2` | 57.7 | 37.9 | **52.3** | 58.7 | 38.4 | **53.1** | 5,305 / 5,305 |
| `bart-base.long.s1` | 46.8 | 30.8 | **42.4** | 47.7 | 31.0 | **43.1** | 12,000 / 35,080 |
| `bart-base.long.s1s2` | 55.2 | 37.8 | **50.6** | 56.7 | 38.7 | **51.9** | 3,183 / 10,610 |
| `bart-base.long.s1cont` | 45.3 | 30.7 | **41.4** | 46.4 | 31.2 | **42.3** | 1,061 / 10,610 |
| `bart-base.gtrecipe.s1` | 55.7 | 26.3 | **45.6** | 55.9 | 26.5 | **45.7** | 5,000 / 10,960 |
| `bart-base.gtrecipe.s1s2` | 58.7 | 35.9 | **52.1** | 59.8 | 36.4 | **53.0** | 4,244 / 5,305 |
| `t5-base.s1` | 55.6 | 27.8 | **46.4** | 56.3 | 27.9 | **46.8** | 13,155 / 13,155 |
| `t5-base.s1s2` | 57.4 | 39.1 | **52.5** | 58.9 | 39.7 | **53.7** | 5,305 / 5,305 |
| `t5-base.long.s1` | 54.3 | 30.8 | **47.1** | 55.2 | 31.0 | **47.8** | 22,000 / 35,080 |
| `t5-base.long.s1s2` | 57.4 | 41.0 | **53.2** | 58.9 | 41.7 | **54.4** | 6,366 / 10,610 |
| `t5-base.long.s1cont` | 53.3 | 30.8 | **46.5** | 54.5 | 31.1 | **47.4** | 1,061 / 10,610 |

**References:** published scores on full BEA-19 dev, ERRANT, from the [GEC-BART README](https://github.com/gotutiyan/GEC-BART). They regenerated the references with `errant_m2 -auto`, so the numbers are close to ours but not exactly comparable.
- `gotutiyan/gec-bart-base`: 56.0 / 29.0 / **47.2**
- `gotutiyan/gec-bart-large`: 60.2 / 33.7 / **52.0**

These are single runs with one seed. Differences of about 1 F0.5 point are within run-to-run noise.

## Findings

1. **Stage 2 is the main lever, and the gain comes from the W&I data.**
   - Stage 2 adds +6 to +9 dev_test F0.5 in every pairing.
   - The matched controls (`long.s1cont`) get the same extra training on stage-1 data instead, and gain nothing. Their best checkpoint is their first evaluation, and they end up below their starting model: BART 42.4 → 41.4, T5 47.1 → 46.5.
   - From the same starting model, W&I gives +9.2 (BART) and +6.7 (T5) over the control.
2. **A longer stage 1 doesn't help.**
   - BART: dev_tune plateaus after about 1.5 epochs. Over the rest of the run, precision falls from 0.56 to 0.50 and recall rises from 0.24 to 0.35, so the Lang-8-heavy data makes it edit more eagerly. dev_test: −1.4.
   - T5 keeps its precision and plateaus at about epoch 4. dev_test: +0.7, within noise.
   - Most best checkpoints come from mid-run, at epochs 2–5 of 8.
3. **A longer stage 2 helps T5 slightly and BART not at all.**
   - T5 `long.s1s2` keeps improving until about epoch 6. It is our best model, at 53.2 dev_test and 54.4 full dev, but only 0.7 above `t5-base.s1s2`.
   - BART peaks at epoch 1–3 of stage 2 and then loses precision.
4. **gotutiyan's hyperparameters help BART's stage 1, but the gain disappears after stage 2.**
   - Stage 1 with learning rate 3e-5 and batch 256: precision 49.0 → 55.7, close to their 56.0. dev_test F0.5 43.8 → 45.6.
   - After stage 2, both recipes are tied (52.1 vs 52.3). `gtrecipe.s1s2` keeps more precision (58.7 vs 57.7) but less recall (35.9 vs 37.9).
5. **Every model after stage 2 beats gotutiyan's bart-large on full dev** (52.0): our five s1s2 models score 51.9–54.4, all at base size. Stage-1-only models are 1.5–4 points below their bart-base (47.2).

**Recommendation:** keep the original recipe (3 + 5 epochs) as the default; it is the cheapest and is within noise of the best. Use `t5-base.long.s1s2` if one best model is needed. Real gains are more likely to come from a larger model, such as `bart-large`, or from changing the data, than from training longer.
