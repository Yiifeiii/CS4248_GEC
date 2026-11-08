#!/bin/bash
set -euo pipefail
runs=(bart-1a bart-1b bart-2a bart-2b t5-1a t5-1b t5-2a t5-2b)
index="${SLURM_ARRAY_TASK_ID:?Array index required}"
[[ "$index" =~ ^[0-7]$ ]] || exit 2
slug="${runs[$index]}"
export FOUR_CONTROL_DATA="$CKPT_ROOT/four-controls-data-20261108"
config="training/configs/four-controls/$slug.json"
out="experiments/training/four-controls-20261108/$slug"
python - "$config" <<'PY'
from pathlib import Path
import json,sys,os
from transformers.utils import cached_file
from training.build_order_control import sha
cfg=json.loads(Path(sys.argv[1]).read_text());root=Path(os.environ['FOUR_CONTROL_DATA'])
m=json.loads((root/'manifest.json').read_text());stream=cfg['control_stream']
for suffix in ('src','tgt'):
    assert sha(root/stream/f'train.{suffix}')==m['streams'][stream][suffix+'_sha256']
records=json.loads((root/'initialization.json').read_text())
assert sha(cached_file(cfg['init'],'model.safetensors'))==records[cfg['init']]['weights_sha256']
PY
mkdir -p "$out"
if [[ -f source_commit.txt ]]; then cp source_commit.txt "$out/source_commit.txt"; fi
cp "$FOUR_CONTROL_DATA/manifest.json" "$out/exposure_manifest.json"
cp "$FOUR_CONTROL_DATA/initialization.json" "$out/initialization.json"
python -m training.finetune --config "$config" --resume auto
run="four-controls-20261108/$slug"
# Same selection rule and final scorer as the original curriculum reference.
python -m training.evaluate --model "$CKPT_ROOT/$run/best" --split dev_tune dev_test --out "$out/best"
# Fixed-budget endpoint is retained separately, never substituted for best scores.
python -m training.evaluate --model "$CKPT_ROOT/$run/final" --split dev_tune dev_test --out "$out/final"
cp "$CKPT_ROOT/$run/run_info.json" "$CKPT_ROOT/$run/trainer_state.json" \
   "$CKPT_ROOT/$run/errant_eval.json" "$config" "$out/"
cp "$CKPT_ROOT/$run/best/best.json" "$out/best_checkpoint.json"
echo "FOUR_CONTROL_COMPLETE: $slug"
