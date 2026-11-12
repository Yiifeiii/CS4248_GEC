# BART stage-2 training configurations

The original nine configurations are listed in `runs.txt`; the three 0.05
follow-ups are in `runs-005.txt`. Run names and output paths are historical IDs.

See the [experiment design and results](../../../experiments/training/bart-ablations-20261105/README.md)
and [all ablation workflows](../../../experiments/ablations/README.md).

```bash
bash cluster/ablations/training/submit.sh --dry-run
bash cluster/ablations/training/submit.sh --dry-run configs/training/bart-ablations-20261105/runs-005.txt
```
