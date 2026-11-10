#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode="${1:---dry-run}"
[[ "$mode" == --dry-run || "$mode" == --submit ]] || {
    echo 'Usage: bash cluster/ablations/decoding/submit.sh [--dry-run|--submit]' >&2; exit 2; }
echo 'Frozen BART only: 7 search setups on all 2000 dev_tune sentences; k=1,3,5.'
echo 'No training, T5 or GRECO. k>1 metrics are edit coverage/diversity, not final correction F0.5.'
if [[ "$mode" == --dry-run ]]; then
    echo 'Preview only; no jobs submitted.'
    exit 0
fi
mkdir -p logs/slurm
sbatch --parsable --job-name=bart-decoding --partition=gpu-long --time=04:00:00 \
    --array=0-6%2 cluster/job.sbatch cluster/ablations/decoding/task.sh
