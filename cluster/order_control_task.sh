#!/bin/bash
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID:?Array index required}" in
  0) family=bart; order=curriculum ;;
  1) family=bart; order=shuffled ;;
  2) family=t5; order=curriculum ;;
  3) family=t5; order=shuffled ;;
  *) exit 2 ;;
esac
export ORDER_DATA_ROOT="$CKPT_ROOT/matched-order-data-20261108"
python - "$ORDER_DATA_ROOT" "$order" <<'PY'
from pathlib import Path
import json,sys
from training.build_order_control import sha
root,order=Path(sys.argv[1]),sys.argv[2]
m=json.loads((root/'manifest.json').read_text())
assert m['total_examples']==1853729 and m['same_exposure_indices']
for suffix in ('src','tgt'):
    assert sha(root/order/f'train.{suffix}')==m['streams'][order][suffix+'_sha256']
from transformers.utils import cached_file
for model, record in json.loads((root/'initialization.json').read_text()).items():
    assert sha(cached_file(model,'model.safetensors'))==record['weights_sha256']
PY
config="training/configs/matched-order/$family-$order.json"
run="matched-order-20261108/$family-$order"
out="experiments/training/$run"
mkdir -p "$out"
cp "$ORDER_DATA_ROOT/manifest.json" "$out/exposure_manifest.json"
cp "$ORDER_DATA_ROOT/initialization.json" "$out/initialization.json"
python -m training.finetune --config "$config" --resume auto
# Primary endpoint: final weights, after the exact same processed-example budget.
python -m training.evaluate --model "$CKPT_ROOT/$run/final" --split dev_tune dev_test --out "$out/final"
# Checkpoint-selection diagnostics only; do not substitute best for the final endpoint.
python -m training.evaluate --model "$CKPT_ROOT/$run/best" --split dev_tune --out "$out/best"
cp "$CKPT_ROOT/$run/run_info.json" "$CKPT_ROOT/$run/trainer_state.json" \
   "$CKPT_ROOT/$run/errant_eval.json" "$config" "$out/"
cp "$CKPT_ROOT/$run/best/best.json" "$out/best_checkpoint.json"
echo "ORDER_CONTROL_COMPLETE: $run"
