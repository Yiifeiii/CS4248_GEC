#!/bin/bash
# Throughput pilot: 200 training steps with the real batch size, in a throwaway
# directory. Read steps/second and peak GPU memory from the log before choosing
# --time for the full run.
# Called by cluster/job.sbatch:  pilot.sh <config>
set -euo pipefail

config="$1"
run=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['run'])" "$config")
dir="$CKPT_ROOT/pilot/$run"
mkdir -p "$CKPT_ROOT/pilot"
rm -rf "$dir"
nvidia-smi --query-gpu=memory.used --format=csv -l 30 > "$dir.gpu-memory.csv" &
monitor=$!
python -m training.finetune --config "$config" --resume no --output-dir "$dir" --max-steps 200 --errant-steps 200
kill "$monitor"
echo "peak GPU memory (MiB): $(sort -n -k1 "$dir.gpu-memory.csv" | grep -o '^[0-9]*' | tail -1)"
