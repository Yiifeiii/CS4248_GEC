#!/bin/bash
# Explicit submission only: this file is not run by tests or configuration generation.
# Default is a preview. Pass --submit only when ready to allocate cluster resources.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode="${1:---dry-run}"
[[ "$mode" == --dry-run || "$mode" == --submit ]] || {
    echo "Usage: bash cluster/ablations/training/submit.sh [--dry-run|--submit] [run-list]" >&2; exit 2; }
config_dir=training/configs/bart-ablations-20261105
run_list="${2:-$config_dir/runs.txt}"
runs=()
while IFS= read -r slug; do runs+=("$slug"); done < "$run_list"
(( ${#runs[@]} > 0 )) || { echo 'Expected at least one run' >&2; exit 1; }
for slug in "${runs[@]}"; do
    [[ "$slug" =~ ^[a-z0-9-]+$ ]] || { echo "Invalid run name: $slug" >&2; exit 1; }
    [[ -f "$config_dir/$slug.json" ]] || { echo "Missing config: $slug" >&2; exit 1; }
done
printf 'Stage-2 ablations, independent runs from CS4248-GEC/gec-bart-base-s1:\n'
printf '  %s\n' "${runs[@]}"
if [[ "$mode" == --dry-run ]]; then
    echo 'Preview only. --submit schedules a CPU preflight and a GPU array (at most 2 running).'
    exit 0
fi

mkdir -p logs/slurm
# job.sbatch activates conda and sets HF_HOME; preflight needs no GPU.
preflight=$(sbatch --parsable --job-name=bart-ablation-check --partition=normal \
    --gpus=0 --cpus-per-task=4 --mem-per-cpu=6000 --time=00:45:00 \
    cluster/job.sbatch cluster/ablations/training/preflight.sh)
preflight="${preflight%%;*}"
array=$(sbatch --parsable --job-name=bart-ablations --partition=gpu-long \
    --time=04:00:00 --array="0-$((${#runs[@]}-1))%2" --dependency="afterok:$preflight" \
    --export="ALL,BART_ABLATION_RUN_LIST=$run_list" \
    --kill-on-invalid-dep=yes cluster/job.sbatch cluster/ablations/training/task.sh)
array="${array%%;*}"
printf 'Preflight job: %s\nTraining array: %s\n' "$preflight" "$array"
printf 'Monitor: squeue -j %s,%s\n' "$preflight" "$array"
