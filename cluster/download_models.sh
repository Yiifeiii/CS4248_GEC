#!/bin/bash
# Download the base models into the scratch Hugging Face cache. Run on xlogin (it is
# meant for downloads) with the gec-train environment active; jobs then run offline.
#   conda activate gec-train && bash cluster/download_models.sh
set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/scratch/${USER:0:1}/$USER}"
export HF_HOME="${HF_HOME:-$SCRATCH/cs4248-gec/hf_cache}"
mkdir -p "$HF_HOME"
for model in facebook/bart-base google-t5/t5-base; do
    huggingface-cli download "$model" --include "*.json" "*.safetensors" "*.model" "*.txt"
done
echo "models cached in $HF_HOME"
