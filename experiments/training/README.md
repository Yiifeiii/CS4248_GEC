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

ERRANT on dev_test (2,384 sentences, beam 5) and on the full dev set (4,384). Scores are percentages.

| Run | dev_test P | dev_test R | **dev_test F0.5** | full dev P | full dev R | **full dev F0.5** | Best step |
|---|---|---|---|---|---|---|---|
| `bart-base.s1` | 49.0 | 30.8 | **43.8** | 49.6 | 31.1 | **44.4** | 13,155 (final) |
| `bart-base.s1s2` | 57.7 | 37.9 | **52.3** | 58.7 | 38.4 | **53.1** | 5,305 (final) |
| `t5-base.s1` | 55.6 | 27.8 | **46.4** | 56.3 | 27.9 | **46.8** | 13,155 (final) |
| `t5-base.s1s2` | 57.4 | 39.1 | **52.5** | 58.9 | 39.7 | **53.7** | 5,305 (final) |
| `bart-base.long.s1` | 46.8 | 30.8 | **42.4** | | | | 12,000 of 35,080 |
| `t5-base.long.s1` | | | running | | | | |
| `bart-base.long.s1s2` | | | running | | | | |
| `bart-base.long.s1cont` | | | queued | | | | |
| `t5-base.long.s1s2`, `t5-base.long.s1cont` | | | queued | | | | |
| `bart-base.gtrecipe.s1`, `.s1s2` | | | queued | | | | |

**References:** published scores on full BEA-19 dev, ERRANT, from the [GEC-BART README](https://github.com/gotutiyan/GEC-BART). They regenerated the references with `errant_m2 -auto`, so the numbers are close to ours but not exactly comparable.
- `gotutiyan/gec-bart-base`: 56.0 / 29.0 / **47.2**
- `gotutiyan/gec-bart-large`: 60.2 / 33.7 / **52.0**

## Findings so far

- **Stage 2 helps a lot:** +8.5 dev_test F0.5 for BART and +6.1 for T5. Both precision and recall go up. After stage 2, BART and T5 are tied, and both base-size models beat gotutiyan's bart-large on full dev (53.1 and 53.7 vs 52.0).
- **A longer stage 1 doesn't help BART.**
  - dev_tune F0.5 plateaus after about 1.5 epochs (0.455–0.469 from step 6k to 35k).
  - Over the same stretch, precision falls (0.56 → 0.50) and recall rises (0.24 → 0.35): more Lang-8-heavy training makes the model edit more eagerly, not more accurately.
  - Its best checkpoint, at step 12k with the learning rate still high, scores 1.4 points below the 3-epoch run on dev_test.
