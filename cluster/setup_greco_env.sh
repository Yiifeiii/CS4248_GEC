#!/bin/bash
# One-time setup of the GRECO environment on the cluster. Run on a compute node (it
# needs internet and memory), from the repository root:
#   salloc -t 60 && srun --pty bash
#   bash cluster/setup_greco_env.sh
# Creates conda env "greco" (Python 3.10), clones and patches upstream GRECO into
# external/greco, puts the 1.7 GB checkpoint and the DeBERTa-v3-large download on
# scratch, so jobs can run offline.
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
SCRATCH="${SCRATCH:-/mnt/scratch/${USER:0:1}/$USER}"
export HF_HOME="${HF_HOME:-$SCRATCH/cs4248-gec/hf_cache}"

set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda env list | grep -q '^greco ' || conda create -y -n greco -c conda-forge --override-channels python=3.10
conda activate greco
set -u

# The checkpoint goes to scratch; models/greco links to it
mkdir -p "$SCRATCH/cs4248-gec/greco" models "$HF_HOME"
[ -e models/greco ] || ln -s "$SCRATCH/cs4248-gec/greco" models/greco

bash greco_setup/setup.sh

# GRECO pins torch 1.12, which cannot run on H100 GPUs (sm_90). Replace it with a CUDA 11.8
# build; the other pinned packages (transformers 4.23.1) stay.
python -m pip install "torch==2.1.2" --index-url https://download.pytorch.org/whl/cu118

python - <<'EOF'
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
AutoModel.from_pretrained("microsoft/deberta-v3-large")
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "- DeBERTa cached in", __import__("os").environ["HF_HOME"])
EOF
