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

# The default Linux wheel is now built for CUDA 13, which the cluster's driver (12.9 in
# October 2026) is too old for. CUDA 12.8 builds run on it; if the driver changes, pick the
# build for the CUDA version that nvidia-smi reports (pytorch.org lists them).
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
python -m spacy download en_core_web_sm
python -c "import torch, transformers, errant; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__)"
