#!/bin/bash
# Train one stage, then score its best checkpoint on dev_tune and dev_test.
# Called by cluster/job.sbatch:  train.sh <config> [extra finetune.py arguments]
# Scores, predictions and run metadata go to experiments/training/<run>/;
# checkpoints stay in $CKPT_ROOT/<run>/.
set -euo pipefail

config="$1"
shift
run=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['run'])" "$config")
out="experiments/training/$run"

python -m training.finetune --config "$config" --resume auto "$@"
python -m training.evaluate --model "$CKPT_ROOT/$run/best" --split dev_tune dev_test --out "$out"

ckpt="$CKPT_ROOT/$run"
cp "$ckpt/run_info.json" "$ckpt/trainer_state.json" "$ckpt/errant_eval.json" "$config" "$out/"
cp "$ckpt/best/best.json" "$out/best_checkpoint.json"
echo "results in $out"
