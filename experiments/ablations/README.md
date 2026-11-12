# BART hyperparameter ablations

This is the entry point for the completed BART training and decoding studies.
All scores below are ERRANT correction F0.5 percentages. Training changes weights;
decoding changes search on a frozen checkpoint. These are single-seed experiments.

| Study | Training parameters | Data scored | Results |
|---|---|---|---|
| Stage-2 training: 12 configurations | Learning rate, weight decay, dropout, label smoothing | dev_tune (2,000), dev_test (2,384) | [All F-scores, deltas and selected epochs](../training/bart-ablations-20261105/results-all-12.md) |
| Frozen BART decoding: 7 configurations | Beams, groups, diversity penalty; candidate k=1/3/5 | dev_tune (2,000) only | [All F-scores and candidate diagnostics](../bart-decoding-20261106/results.md) |

## Training hyperparameters

Each configuration starts independently from the same stage-1 checkpoint, `CS4248-GEC/gec-bart-base-s1`.
The baseline retains learning rate 1e-5, weight decay 0.01, dropout 0.1 and label
smoothing 0.1. Everything else follows the recorded stage-2 recipe: linear learning
rate schedule, 100 warmup steps, up to five epochs, seed 42, batch 32, checkpoint
selection by dev_tune F0.5, patience two, and final five-beam evaluation.
Dropout variants change residual, attention and activation dropout together.

| Parameter | Values tested |
|---|---|
| Peak learning rate | 5e-6, **1e-5**, 2e-5 |
| Weight decay | 0, **0.01**, 0.05, 0.1 |
| Dropout | 0, 0.05, **0.1**, 0.3 |
| Label smoothing | 0, 0.05, **0.1**, 0.2 |

Bold values are the shared baseline. This is a one-factor-at-a-time design, not
the Cartesian product of these values. Nine initial runs plus three follow-ups
make 12 configurations including the baseline.

Baseline: **54.02 dev_tune / 52.26 dev_test**. Weight decay 0.1 improves dev_tune
to 54.20 but gives 52.07 on dev_test. No variant beats baseline on dev_test.
The follow-ups were chosen after the first results were inspected, so their
held-out comparisons are exploratory. Small differences need additional seeds.

- [Launch, task and preflight scripts](../../cluster/ablations/training/)
- [Configurations and fixed array orders](../../configs/training/bart-ablations-20261105/)
- [Design, controls and provenance](../training/bart-ablations-20261105/README.md)
- [Machine-readable results](../training/bart-ablations-20261105/summary-all-12.json)
- Implementation: `training/finetune.py`, `training/dropout.py`.
- Tests: `tests/test_bart_ablations.py`.

## Beam-search hyperparameters

All seven configurations use the same completed BART stage-2 baseline weights.
Ordinary search tests 5/10/20 beams. Diverse search uses 10 beams with two groups
at penalties 0.1/0.5/1.0, or five groups at penalty 0.5. No T5 or GRECO is involved.

The ordinary one-best control scores **53.96 / 54.09 / 54.07** at 5/10/20 beams.
The separately generated and rescored n-best route scores **53.93 / 54.11 / 54.07**.
These routes use different precision and ranking procedures; compare within a
route. Diverse search does not beat ordinary beam10 candidate-top1 F0.5.
These are dev_tune results, not held-out improvements.

Candidate k=3/5 results measure gold-edit coverage and diversity, **not achieved
F0.5**. A candidate pool has no single correction output until a selection rule
is supplied. The results document reports this distinction explicitly.

- [Launch and task scripts](../../cluster/ablations/decoding/)
- [Search configuration](../../configs/training/bart-decoding-20261106.json)
- [Design and scoring definitions](../bart-decoding-20261106/README.md)
- [Machine-readable results](../bart-decoding-20261106/results-summary.json)
- Implementation: `training/bart_decoding_ablation.py`, `training/candidate_quality.py`.
- Tests: `tests/test_bart_decoding.py`.

## Preview, submit and validate

Run from the repository root. Preview commands never submit Slurm jobs.

```bash
bash cluster/ablations/training/submit.sh --dry-run
bash cluster/ablations/training/submit.sh --dry-run configs/training/bart-ablations-20261105/runs-005.txt
bash cluster/ablations/decoding/submit.sh --dry-run
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
```

On the configured SoC cluster, replace `--dry-run` with `--submit` to allocate
resources. Training performs a CPU preflight before its GPU array; both workflows
allow at most two concurrent GPU tasks. See [cluster setup](../../cluster/README.md).
Do not resubmit completed runs just to read their scores.

Historical run IDs, source commits and result directories are retained so saved
results can be traced to the executed versions. Curated score JSONs and summaries
are committed; checkpoints, generated predictions, n-best pools and local plots
are excluded. Earlier 100-sentence pilots and pretrained-checkpoint evaluations
are separate exploratory work, outside these controlled BART studies.
