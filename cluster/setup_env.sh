#!/bin/bash
# One-time environment setup on the SoC cluster. Run inside an interactive compute
# session, not on xlogin (the login node has too little memory for conda):
#   salloc -t 120
#   srun --pty bash
#   bash cluster/setup_env.sh
# Installs Miniconda if missing, then creates the gec-train environment.
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ ! -x "$CONDA_BASE/bin/conda" ]; then
    cd "$HOME"
    curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_BASE"
    "$CONDA_BASE/bin/conda" init bash
    cd - > /dev/null
fi
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda-forge only: Anaconda's default channels need their Terms of Service accepted first
conda env list | grep -q '^gec-train ' || conda create -y -n gec-train -c conda-forge --override-channels python=3.11
conda activate gec-train
set -u

# The default Linux wheel bundles CUDA 12 libraries; if torch.cuda.is_available() prints False
# on a GPU node, pick the wheel for the driver's CUDA version (nvidia-smi) from pytorch.org.
pip install torch
pip install -r requirements-train.txt
python -m spacy download en_core_web_sm
python -c "import torch, transformers, errant; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__)"
