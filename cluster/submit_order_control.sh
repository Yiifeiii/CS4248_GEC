#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:---dry-run}"
case "$mode" in
  --dry-run) echo 'Preview: 4 ordering-only runs; 0 BART curriculum, 1 BART shuffled, 2 T5 curriculum, 3 T5 shuffled. No submission.' ;;
  --submit)
    mkdir -p logs/slurm
    preflight=$(sbatch --parsable --partition=normal --gpus=0 --cpus-per-task=4 \
      --mem-per-cpu=6000 --time=00:45:00 -J order-check \
      cluster/job.sbatch cluster/order_control_preflight.sh)
    preflight="${preflight%%;*}"
    echo "Preflight: $preflight"
    sbatch --parsable --array=0-3%2 --time=12:00:00 -J matched-order \
      --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
      cluster/job.sbatch cluster/order_control_task.sh ;;
  *) echo 'Usage: bash cluster/submit_order_control.sh [--dry-run|--submit]' >&2; exit 2 ;;
esac
