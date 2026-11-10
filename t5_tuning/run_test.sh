#!/bin/bash
# Score one verified run's best checkpoint on dev_test, once. Submitted by tune.py test:
#   sbatch -J test-<run> cluster/job.sbatch t5_tuning/run_test.sh <run> t5_tuning/runs/final-dev_test/seed<seed>
set -euo pipefail

run="$1"
out="$2"
if [ -e "$out/dev_test.scores.json" ]; then
    echo "$out already has dev_test scores; dev_test is scored once" >&2
    exit 1
fi
python -m training.evaluate --model "$CKPT_ROOT/$run/best" --split dev_test --out "$out"
lines=$(wc -l < "$out/dev_test.pred")
if [ "$lines" -ne 2384 ]; then
    echo "$out/dev_test.pred has $lines lines, expected 2384" >&2
    exit 1
fi
