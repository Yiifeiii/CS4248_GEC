#!/bin/bash
# Called by job.sbatch once for each task in the nine-run Slurm array.
set -euo pipefail

config_dir=training/configs/bart-ablations-20261105
runs=()
while IFS= read -r slug; do runs+=("$slug"); done < "$config_dir/runs.txt"
index="${SLURM_ARRAY_TASK_ID:?Submit using cluster/submit_bart_ablations.sh}"
[[ "$index" =~ ^[0-8]$ ]] || { echo "Invalid array task: $index" >&2; exit 1; }
run="bart-ablations-20261105/${runs[$index]}"
config="$config_dir/${runs[$index]}.json"
out="experiments/training/$run"
mkdir -p "$out"

python -m training.finetune --config "$config" --resume auto
# Hyperparameter selection uses dev_tune. Hold dev_test for the final comparison.
python -m training.evaluate --model "$CKPT_ROOT/$run/best" --split dev_tune --out "$out"
cp "$CKPT_ROOT/$run/run_info.json" "$CKPT_ROOT/$run/trainer_state.json" \
   "$CKPT_ROOT/$run/errant_eval.json" "$config" "$out/"
cp "$CKPT_ROOT/$run/best/best.json" "$out/best_checkpoint.json"
echo "ABLATION_COMPLETE: $run"
