#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:---dry-run}" in
  --dry-run) echo 'Preview: BART/T5 x 1A,1B,2A,2B; eight controls, at most two concurrent. No submission.' ;;
  --submit)
    mkdir -p logs/slurm
    check=$(sbatch --parsable --partition=normal --gpus=0 --cpus-per-task=4 \
      --mem-per-cpu=6000 --time=00:45:00 -J four-controls-check \
      cluster/job.sbatch cluster/four_controls_preflight.sh)
    check="${check%%;*}"
    echo "Preflight: $check"
    sbatch --parsable --array=0-7%2 --time=12:00:00 -J four-controls \
      --dependency="afterok:$check" --kill-on-invalid-dep=yes \
      cluster/job.sbatch cluster/four_controls_task.sh ;;
  *) echo 'Usage: bash cluster/submit_four_controls.sh [--dry-run|--submit]' >&2;exit 2 ;;
esac
