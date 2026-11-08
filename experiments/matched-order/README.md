# Matched curriculum-order experiment: BART and T5

Question: does broad-correction-first, W&I-later ordering help compared with
mixing exactly the same training presentations throughout?

Four runs: BART curriculum / BART shuffled / T5 curriculum / T5 shuffled. This
experiment is distinct from Dasha's original recipe and the earlier single-stage
controls. Within each model pair, only input order and artifact paths differ.

## Eight matched rounds and exact exposure

Use three copies of verified stage1 and five copies of verified stage2, retaining
all repeated/overlapping examples: 1,853,729 presentations. Each training-row ID
occurs exactly as often in both streams. Overlapping sentence pairs therefore
also have equal total multiplicity.

| Round | Examples per round | Curriculum stream | Shuffled stream |
|---|---:|---|---|
| 1–3 | 561,338 | Independently shuffled stage1 data | Corresponding slice of globally shuffled full stack |
| 4–8 | 33,943 | Independently shuffled stage2 data | Corresponding slice of the same shuffled stack |

Round boundaries match. No examples are added, removed or padded to align batches.
No dev/test examples are added to training. Stage source/target hashes are checked
against the original manifest. `exposure.ids` records every training-row occurrence;
preflight proves equal ID multisets and records hashes for both streams and IDs.

These are **eight logical rounds, not eight passes over the combined dataset**.
Trainer performs one physical pass over each fixed expanded stream (`epochs: 1`);
the eight boundaries are recorded in `round_end_examples`. A batch may straddle
a boundary. Sample positions/batch sizes match, with no optimizer reset there.

## Matched optimization and evaluation

- Original pretrained initialization: facebook/bart-base or google-t5/t5-base.
  Preflight hashes both checkpoints; every task verifies those hashes.
- Single GPU/process, batch64, accumulation1. The incomplete final batch is kept:
  all 1,853,729 presentations are consumed in 28,965 updates. A real tiny-model
  training test verifies ordered consumption, including the incomplete tail.
- BART LR5e-5; T5 LR1e-4. Identical single linear schedule and 2,000 warmup
  updates within each family; seed42, AdamW, weight decay0.01, dropout0.1,
  label smoothing0.1 and shared tokenization/length limits.
- Sequential sampler, no global shuffling/length grouping: the file contains the
  desired order. Preserve-order configs reject accumulation, early stopping and
  max-step/sample truncation that would break exact exposure.
- Shared loss/checkpoint/ERRANT interval: 2,000 updates. Full 2,000-sentence
  dev_tune, five beams; no early stopping. Final updates are also scored.
- **Primary endpoint: final weights**, evaluated on dev_tune and dev_test with
  five-beam ERRANT/M2 after equal processed-example budgets. Best-dev_tune weights
  and learning curves are secondary diagnostics; no best-checkpoint dev_test search.

Any difference reflects changed data order interacting with the shared training
process. This is one seed per model, not a significance study. Do not compare a new
run with Dasha's old score as an order-only test: her optimization recipe differs.

## Running and artifacts

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
bash cluster/submit_order_control.sh --dry-run
# Configured SoC cluster only:
bash cluster/submit_order_control.sh --submit
```

CPU preflight checks dependencies/software/tests/data, builds both streams and
records initialization hashes. Four GPU jobs depend on it, at most two concurrent,
each requesting one H100-47 for up to12h. Array indices: 0 BART curriculum,
1 BART shuffled, 2 T5 curriculum, 3 T5 shuffled. Models/data stay on scratch.
Source snapshots must include root `m2_to_parallel.py` and bundled `m2scorer/`,
as well as training modules and historical reference metadata.

Configs: `training/configs/matched-order/`. Code: `training/build_order_control.py`,
`training/ordered.py` and opt-in integration in `training/finetune.py`. Existing
configs retain their previous sampling behavior. Results live under
`experiments/training/matched-order-20261108/<family>-<order>/final/` for primary
scores, and `best/` for dev_tune diagnostics. Retain exposure/initialization
manifests, full run metadata and trainer state with results.

Status: submitted on 8 November 2026. CPU preflight **1095333**; dependent
GPU array **1095334**, indices0–3 as above, at most two concurrent tasks.
Executed source: `2c3d817`. Dedicated snapshot:
`~/CS4248_GEC_experiments/matched-order/`. Local validation: 29 tests,
28 passed and one environment-dependent GRECO integration test skipped. Shell
syntax, no-submit preview and whitespace checks passed. No scores claimed yet.
