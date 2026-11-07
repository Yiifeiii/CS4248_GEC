#!/bin/bash
# Score one GRECO checkpoint on the dev_tune pool with the mini experiment's settings:
# official combination, beam 16, batch 8, vote_coef 0.4 (the paper's value) and 0 (control),
# and k = 1, 3, 5. Writes greco_tuning/runs/<name>/vote<v>/k<k>/greco.pred.
# Submit from the repository root on the cluster, with the checkpoint's full path:
#   sbatch -J greco-score --time=6:00:00 cluster/job.sbatch greco_tuning/score.sh \
#     $CKPT_ROOT/greco.s1.lr2e-5.seed42/pytorch_model.bin s1
#   sbatch -J greco-score --time=6:00:00 cluster/job.sbatch greco_tuning/score.sh \
#     $PWD/models/greco/checkpoint.bin original
# SETTINGS=select scores only vote 0.4, k 3 (used to pick the best epoch); default is the full grid.
# Needs dev_pool.sh finished (greco_tuning/runs/pool/m2 exists).
set -euo pipefail

CKPT=$(realpath "$1")
NAME=$2
REPO=$PWD
POOL=$REPO/greco_tuning/runs/pool/m2
OUT=$REPO/greco_tuning/runs/$NAME

grep -q "attention_mask=texts\['attention_mask'\]" external/greco/models.py || {
  echo "GRECO models.py lacks the attention-mask fix; rerun greco_setup/setup.sh" >&2; exit 1; }

set +u
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate greco
set -u

cd "$REPO/external/greco"
for vote in 0.4 0; do
  for k in 1 3 5; do
    # SETTINGS=select scores only vote 0.4, k 3: the mini experiment's best, used to pick the epoch
    if [ "${SETTINGS:-grid}" = select ] && { [ "$vote" != 0.4 ] || [ "$k" != 3 ]; }; then continue; fi
    inputs=("$POOL/gector-1.m2")
    for r in $(seq 1 "$k"); do inputs+=("$POOL/bart-$r.m2" "$POOL/t5-$r.m2"); done
    dir="$OUT/vote$vote/k$k"
    mkdir -p "$dir"
    python run_combination.py --model greco --lm_model microsoft/deberta-v3-large \
      --checkpoint "$CKPT" --beam_size 16 --batch_size 8 --vote_coef "$vote" \
      --output_path "$dir/greco.pred" --data "${inputs[@]}"
  done
done
