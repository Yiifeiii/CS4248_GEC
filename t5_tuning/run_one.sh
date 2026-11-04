#!/bin/bash
# Train one stage-2 run, then score its best checkpoint on dev_tune only.
# Called by cluster/job.sbatch from the repository root:
#   sbatch -J <run> cluster/job.sbatch t5_tuning/run_one.sh t5_tuning/configs/<run>.json
# Unlike cluster/train.sh, this does not score dev_test, so dev_test stays untouched.
set -euo pipefail

config="$1"
run=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['run'])" "$config")
out="t5_tuning/runs/$run"

python -m training.finetune --config "$config" --resume auto
python -m training.evaluate --model "$CKPT_ROOT/$run/best" --split dev_tune --out "$out"
