#!/bin/bash
# GRECO vote tuning job: extract ERRANT edits of the system outputs and, after GRECO
# (greco_setup.tune_vote in the "greco" env), score the predictions with ERRANT (both in
# the "gec-train" env). Submit through job.sbatch:
#   sbatch -J greco-test --time=00:30:00 cluster/job.sbatch cluster/greco_tune.sh dev_tune --limit 20 --k 1 3 --alphas 0 0.4
#   sbatch -J greco-tune cluster/job.sbatch cluster/greco_tune.sh dev_tune
#   sbatch -J greco-test-set cluster/job.sbatch cluster/greco_tune.sh dev_test \
#       --best-from outputs/greco/tuning/dev_tune.errant/best.json
# The first argument is the split; the rest go to tune_vote.py (--edits difflib uses the
# old difflib edits and writes to <split> instead of <split>.errant).
set -euo pipefail

SPLIT="$1"
shift
GRECO_ENV="${GRECO_ENV:-greco}"
SCORE_ENV="${SCORE_ENV:-gec-train}"
export TRANSFORMERS_OFFLINE=1
# The ERRANT steps run one process per CPU; keep each to one thread
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
step() { echo "== $(date +%T) $*"; }

# The folder tune_vote.py writes: <split>[.errant][.first<N>]
EDITS=errant
LIMIT=""
args=("$@")
for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--limit" ]; then LIMIT=".first${args[$((i + 1))]}"; fi
    if [ "${args[$i]}" = "--edits" ]; then EDITS="${args[$((i + 1))]}"; fi
done
OUT_SPLIT="$SPLIT"
[ "$EDITS" = errant ] && OUT_SPLIT="$OUT_SPLIT.errant"
OUT_SPLIT="$OUT_SPLIT$LIMIT"

# job.sbatch runs this script in a new bash, which does not inherit the conda function
set +u
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$SCORE_ENV"
set -u
step extracting ERRANT edits
[ "$EDITS" = errant ] && python -m greco_setup.extract_edits

set +u
conda activate "$GRECO_ENV"
set -u
step GRECO
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
python -m greco_setup.tune_vote --split "$SPLIT" --device cuda "$@"

set +u
conda activate "$SCORE_ENV"
set -u
step scoring
python -m greco_setup.score_tuning --split "$OUT_SPLIT"
