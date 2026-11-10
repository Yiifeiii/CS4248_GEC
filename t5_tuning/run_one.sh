#!/bin/bash
# Train one run, score its best checkpoint on dev_tune only, then check the run against its config
# (tune.py verify: config, data, versions, prediction count, loss curve).
# Called by cluster/job.sbatch from the repository root:
#   sbatch -J <run> cluster/job.sbatch t5_tuning/run_one.sh t5_tuning/configs/<run>.json
# Unlike cluster/train.sh, this does not score dev_test, so dev_test stays untouched.
set -euo pipefail

config="$1"
run=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['run'])" "$config")
out="t5_tuning/runs/$run"

python -m training.finetune --config "$config" --resume auto
python -m training.evaluate --model "$CKPT_ROOT/$run/best" --split dev_tune --out "$out"
python t5_tuning/tune.py verify --run "$run"
