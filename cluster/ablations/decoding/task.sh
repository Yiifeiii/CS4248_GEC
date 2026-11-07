#!/bin/bash
set -euo pipefail
python -m training.bart_decoding_ablation --index "${SLURM_ARRAY_TASK_ID:?Array index required}"
