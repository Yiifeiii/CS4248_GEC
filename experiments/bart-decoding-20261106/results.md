# Completed BART-only decoding results

All seven tasks in array 1081371 completed with exit code 0. Retrieved and checked on 6 November 2026. All setups use the same frozen BART weights and 2,000 dev_tune sentences; source and gold hashes match across all seven manifests. No training, T5, GECToR, or GRECO was involved.

## Candidate ranking and edit coverage

F0.5 is measured on candidate top1 ranked by teacher-forced average token log probability. Coverage at k is the percentage of gold edits appearing anywhere among the top-k unique candidates; it is a reference-assisted diagnostic, not achieved recall or F0.5 of a correction system.

| Setup | Beams | Groups | Penalty | Candidate top1 F0.5 | Gold coverage k=1 | Gold coverage k=3 | Gold coverage k=5 | Mean unique at k=5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| beam5 | 5 | 1 | 0 | 53.93 | 38.99 | 53.79 | 59.11 | 4.94 |
| beam10 | 10 | 1 | 0 | 54.11 | 39.02 | 53.55 | 59.31 | 4.94 |
| beam20 | 20 | 1 | 0 | 54.07 | 38.99 | 53.73 | 59.49 | 4.95 |
| groups2-p05 | 10 | 2 | 0.5 | 53.94 | 38.99 | 53.79 | 59.13 | 4.91 |
| groups5-p05 | 10 | 5 | 0.5 | 54.03 | 38.88 | 52.06 | 54.60 | 3.79 |
| groups2-p01 | 10 | 2 | 0.1 | 53.96 | 38.99 | 53.79 | 58.93 | 4.46 |
| groups2-p10 | 10 | 2 | 1 | 53.93 | 38.99 | 53.79 | 59.11 | 4.94 |

## Conventional standalone one-best controls

These use training.evaluate with one returned sequence and bf16 autocast. They are separate from rescoring an fp16-generated n-best list; use within-route comparisons.

| Beams | ERRANT P | ERRANT R | ERRANT F0.5 |
|---|---:|---:|---:|
| 5 | 59.69 | 38.99 | 53.96 |
| 10 | 59.92 | 38.94 | 54.09 |
| 20 | 59.89 | 38.94 | 54.07 |

## Findings

- Beam10 gives the highest candidate-top1 F0.5: 54.11, versus 53.93 at beam5 (+0.18 points). The conventional one-best control gives 54.09 versus 53.96 (+0.13). These are small differences on dev_tune, not evidence of a reliable held-out gain.
- Beam20 gives the largest top-5 gold-edit coverage, 59.49%, versus 59.11% at beam5 (+0.38 percentage points). It does not give the best top1 score.
- More candidates increase gold-edit coverage, but this does not mean a deployed system would select those edits correctly or avoid false edits.
- Five groups at penalty 0.5 reduces mean unique outputs from about 4.94 to 3.79 and top-5 coverage from 59.31% (ordinary beam10) to 54.60%. Grouped search does not guarantee more diverse or useful shortlisted outputs.
- Two groups with penalties 0.1, 0.5, or 1.0 does not improve candidate-top1 F0.5 over ordinary beam10. Its top-5 coverage is also lower.
- All settings were explored on dev_tune. Freeze any chosen settings before a separate held-out comparison. Do not call these scores dev_test results.

## Artifacts and runtime

Machine-readable scores, P/R, coverage, uniqueness, and diversity are in [results-summary.json](results-summary.json). Retrieved raw summaries/manifests are stored locally under outputs/bart-decoding-20261106/<setup>/; full n-best lists and fixed-k exports remain on the cluster. Only curated scores and provenance are versioned; full generated outputs remain outside Git.

Observed Slurm task elapsed times: beam5 8m56s, beam10 9m55s, beam20 11m40s, groups2-p05 58m25s, groups5-p05 7m21s, groups2-p01 6m43s, groups2-p10 7m01s. Ordinary jobs also run a conventional one-best control, so total job times are not comparable measures of generation speed. Allocations also differ; use measured generation timings with caution rather than attributing the slow task entirely to its search settings.
