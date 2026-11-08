#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:---dry-run}"
case "$mode" in
  --dry-run) echo 'Preview: BART + T5, single-stage exposure-matched mixture, 2 GPU jobs; no submission.' ;;
  --submit)
    mkdir -p logs/slurm
    preflight=$(sbatch --parsable --partition=normal --gpus=0 --cpus-per-task=4 \
      --mem-per-cpu=6000 --time=00:45:00 -J mixed-check \
      cluster/job.sbatch cluster/mixed_control_preflight.sh)
    preflight="${preflight%%;*}"
    echo "Preflight: $preflight"
    sbatch --parsable --array=0-1%2 --time=12:00:00 -J mixed-control \
      --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
      cluster/job.sbatch cluster/mixed_control_task.sh ;;
  *) echo 'Usage: bash cluster/submit_mixed_control.sh [--dry-run|--submit]' >&2; exit 2 ;;
esac
