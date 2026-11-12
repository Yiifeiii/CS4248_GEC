#!/bin/bash
# Called by job.sbatch once per task in the selected ablation run list.
set -euo pipefail

config_dir=configs/training/bart-ablations-20261105
runs=()
run_list="${BART_ABLATION_RUN_LIST:-$config_dir/runs.txt}"
while IFS= read -r slug; do runs+=("$slug"); done < "$run_list"
index="${SLURM_ARRAY_TASK_ID:?Submit using cluster/ablations/training/submit.sh}"
[[ "$index" =~ ^[0-9]+$ ]] && (( index < ${#runs[@]} )) || {
    echo "Invalid array task: $index" >&2; exit 1; }
run="bart-ablations-20261105/${runs[$index]}"
config="$config_dir/${runs[$index]}.json"
out="experiments/training/$run"
mkdir -p "$out"

python -m training.finetune --config "$config" --resume auto
# Match Dasha's final reporting; selection/early stopping still use dev_tune only.
python -m training.evaluate --model "$CKPT_ROOT/$run/best" --split dev_tune dev_test --out "$out"
cp "$CKPT_ROOT/$run/run_info.json" "$CKPT_ROOT/$run/trainer_state.json" \
   "$CKPT_ROOT/$run/errant_eval.json" "$config" "$out/"
cp "$CKPT_ROOT/$run/best/best.json" "$out/best_checkpoint.json"
echo "ABLATION_COMPLETE: $run"
