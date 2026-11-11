# Frozen BART decoding and candidate-quality ablations

**Completed:** all seven tasks finished successfully with exit code 0. See the
[consolidated results](results.md), retrieved and checked on 6 November 2026.

This BART-only study changes inference search, not weights. It uses the completed
`CS4248-GEC/gec-bart-base-s1s2` baseline checkpoint throughout. T5, GECToR, and GRECO
are not loaded or combined. Seven independent jobs use all 2,000 original
tokenized sentences in `data/curriculum/dev_tune.src`; `dev_tune.m2` is used only
for evaluation. No dev_test sentences or annotations are used for this sweep.

## Search grid

| Array index | Setup | Beams | Groups | Diversity penalty |
|---|---|---:|---:|---:|
| 0 | `beam5` | 5 | 1 (ordinary search) | 0 |
| 1 | `beam10` | 10 | 1 (ordinary search) | 0 |
| 2 | `beam20` | 20 | 1 (ordinary search) | 0 |
| 3 | `groups2-p05` | 10 | 2 | 0.5 |
| 4 | `groups5-p05` | 10 | 5 | 0.5 |
| 5 | `groups2-p01` | 10 | 2 | 0.1 |
| 6 | `groups2-p10` | 10 | 2 | 1.0 |

The [config](../../configs/training/bart-decoding-20261106.json) fixes the model
weight hash, five returned sequences, length penalty 1, no-repeat n-gram size 0,
batch size 4, and batch token budget 400. Decoding is deterministic beam search,
without sampling. Group counts divide evenly into the total beam count.

Compare beam5/10/20 for search width; beam10 versus the two penalty-0.5 grouped
variants for diversity; and the three two-group variants for penalty strength.
Group count also changes beams per group: two groups have five beams each,
whereas five groups have two each.

## What k=1,3,5 means without a combination system

Each job generates one pool of up to five unique outputs per source, ranks them
by teacher-forced average token log probability, and reuses that same ranked pool
for k=1,3,5. There is no group-balanced selection. The generator rescores diverse
outputs without their diversity penalty so rankings are comparable.

- **Candidate top1:** official ERRANT and M2 P/R/F0.5 for the first ranked candidate.
  It is a deployable one-output result, with no gold-based selection.
- **Conventional one-best control:** ordinary beam5/10/20 are also decoded using
  `training.evaluate`, returning one sequence. This reproduces the standalone
  baseline evaluation route and distinguishes it from rescoring an n-best pool.
- **Gold-edit coverage at k:** the fraction of gold edits found anywhere among
  the top k unique candidates, using ERRANT extraction and exact span/correction
  matches. A larger pool can contain useful edits missed by top1.
- **Diversity/availability at k:** mean number of available unique candidates,
  number of sources with a full k, and mean pairwise token dissimilarity
  (`1 - SequenceMatcher.ratio`). Higher diversity is not automatically better.

Coverage is a reference-assisted diagnostic, not system recall or an achieved
F0.5. Edits from different candidates can conflict, and a candidate can contain
both correct and incorrect changes. No gold-aware sentence selection is used to
produce candidate top1. Do not label k=3/5 coverage as improved correction F0.5.
The coverage calculation requires single-annotator gold and excludes noop/UNK
entries; it is kept separate from official ERRANT/M2 scoring.

The n-best generator deduplicates outputs, so fewer than k unique candidates may
exist. Metrics use only actual unique candidates. Exported `candidates-k*.tgt`
files follow the existing fixed-line-count format and pad missing slots with top1;
padding is not credited as diversity or additional gold coverage.

N-best generation uses the existing BART fp16 inference route; conventional
one-best controls use the existing evaluator's bf16 autocast. Compare like-for-like
metrics across setups rather than attributing every difference between those two
routes to beam search alone.

## Running and outputs

```bash
HF_HOME=/tmp/gec-ablation-test-cache HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
bash cluster/ablations/decoding/submit.sh --dry-run
# On the cluster, after syncing these files and confirming the baseline is cached:
bash cluster/ablations/decoding/submit.sh --submit
```

The submission script creates a seven-task GPU array, at most two tasks running
concurrently. Each requests one H100-47 and a four-hour time limit. Jobs run in
the existing `gec-train` environment and use the scratch HF cache offline. Every
task verifies the frozen checkpoint SHA-256 before decoding.

Artifacts live under `outputs/bart-decoding-20261106/<setup>/`: source and gold,
manifest with source/gold hashes, n-best JSONL, k-specific candidate exports,
candidate-top1 predictions/scores, candidate-pool metrics, and summary.json.
Ordinary setups also have `standard-one-best/`. The manifest records the exact
setup, device, and whether any weight updates/combination occurred (both false).
The generation wall time includes subprocess/model-loading overhead. Complete
candidate pools can be reused on resubmission only if their recorded settings
and source alignment match; incomplete generations use the generator's partial
resume mechanism.

Settings are explored on dev_tune. After the settings are frozen, a final
comparison against the baseline can be run separately on held-out data. This
study does not claim that larger beams or more diverse outputs necessarily help.

## Submission

- Submitted on 6 November 2026 as array **1081371**, indices 0–6, at most two
  concurrent tasks. The grid table above is the array-index mapping.
- Executed source commit: `8d898d4581441e3144f150e78f0f2e9f399741c7`.
- Local validation: 35 tests ran, 34 passed, one unrelated GRECO integration test
  skipped. Coverage diagnostics, no-submit preview, and real grouped generation on
  tiny offline BART were tested; these are not the full-data experiment results.
- Jobs use one H100-47 request each, eight CPUs, and a four-hour limit on gpu-long.
- All seven tasks completed successfully. See [scores and conclusions](results.md).

```bash
squeue -j 1081371
sacct -j 1081371 --format=JobID,State,ExitCode,Elapsed -X
```

Logs are `logs/slurm/bart-decoding-<individual task job ID>.out`. The log's
individual job ID may differ from array notation `1081371_<index>`.
