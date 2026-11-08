#!/bin/bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID:?Array index required}" in
  0) family=bart ;;
  1) family=t5 ;;
  *) exit 2 ;;
esac
python - <<'CHECK'
import json
from pathlib import Path
import torch, transformers
reference=json.loads(Path('experiments/training/bart-base.s1s2/run_info.json').read_text())
assert reference['versions']=={'torch':torch.__version__,'transformers':transformers.__version__}, 'Historical training software mismatch'
CHECK
# Each job builds its own deterministic copy; no shared-output write race.
data_root="$CKPT_ROOT/mixed-control-data/$family"
# Full data/import validation runs in the CPU preflight dependency.
python -m training.build_mixed_control --output "$data_root/train"
export MIXED_CONTROL_TRAIN="$data_root/train"
config="training/configs/mixed-control/$family.json"
bash cluster/train.sh "$config"
cp "$data_root/train.manifest.json" "experiments/training/$family-base.mixed-control/data_manifest.json"
