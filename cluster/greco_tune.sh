#!/bin/bash
# GRECO vote tuning job: run greco_setup.tune_vote in the "greco" env, then score the
# predictions with ERRANT in the "gec-train" env. Submit through job.sbatch:
#   sbatch -J greco-test --time=00:30:00 cluster/job.sbatch cluster/greco_tune.sh dev_tune --limit 20 --k 1 3 --alphas 0 0.4
#   sbatch -J greco-tune cluster/job.sbatch cluster/greco_tune.sh dev_tune
#   sbatch -J greco-test-set cluster/job.sbatch cluster/greco_tune.sh dev_test \
#       --best-from outputs/greco/tuning/dev_tune/best.json
# The first argument is the split; the rest go to tune_vote.py.
set -euo pipefail

SPLIT="$1"
shift
GRECO_ENV="${GRECO_ENV:-greco}"
SCORE_ENV="${SCORE_ENV:-gec-train}"
export TRANSFORMERS_OFFLINE=1

# tune_vote.py writes --limit runs to <split>.first<N>
OUT_SPLIT="$SPLIT"
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--limit" ]; then OUT_SPLIT="$SPLIT.first${args[$((i + 1))]}"; fi
done

# job.sbatch runs this script in a new bash, which does not inherit the conda function
set +u
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$GRECO_ENV"
set -u
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
python -m greco_setup.tune_vote --split "$SPLIT" --device cuda "$@"

set +u
conda activate "$SCORE_ENV"
set -u
python -m greco_setup.score_tuning --split "$OUT_SPLIT"
