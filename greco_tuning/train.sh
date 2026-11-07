#!/bin/bash
# GRECO training on the stage-1 groups: learning rate 2e-5, 3 epochs, seed 42.
# Writes $CKPT_ROOT/greco.s1.lr2e-5.seed42/pytorch_model.bin, the state dict that score.sh loads.
# Submit from the repository root on the cluster:
#   sbatch -J greco-train --time=24:00:00 cluster/job.sbatch greco_tuning/train.sh
# Needs greco_setup/setup.sh already run in the greco environment, and the DeBERTa backbone
# in the scratch Hugging Face cache (see greco_tuning/README.md).
set -euo pipefail

REPO=$PWD
DATA=$REPO/data/greco/stage1.train.jsonl
CKPT=$REPO/models/greco/checkpoint.bin
OUT=${CKPT_ROOT}/greco.s1.lr2e-5.seed42

test -f "$DATA" || { echo "missing $DATA" >&2; exit 1; }
test -f "$CKPT" || { echo "missing $CKPT; run greco_setup/setup.sh" >&2; exit 1; }

set +u
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate greco
set -u

cd "$REPO/external/greco"
python train.py --do_train \
  --model_name_or_path microsoft/deberta-v3-large \
  --pretrained_path "$CKPT" \
  --output_dir "$OUT" \
  --learning_rate 2e-5 --word_dropout 0.25 --alpha 1 \
  --edit_weight 2.0 --rank_multiplier 5 --gamma 0.2 \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 4 \
  --num_train_epochs 3 --save_strategy epoch --seed 42 \
  --data "$DATA" --data_mode hierarchical
