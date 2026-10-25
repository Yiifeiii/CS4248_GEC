#!/bin/bash
# Download the base models into the scratch Hugging Face cache; jobs then run offline.
# Run on a compute node, which has internet access; xlogin's per-user memory limit
# aborts large downloads:
#   salloc -t 30 && srun --pty bash
#   conda activate gec-train && bash cluster/download_models.sh
# Re-running resumes and skips files that are already complete.
set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/scratch/${USER:0:1}/$USER}"
export HF_HOME="${HF_HOME:-$SCRATCH/cs4248-gec/hf_cache}"
# The Xet download backend buffers chunks in memory and aborted on xlogin
# ("memory allocation of ... bytes failed"); plain HTTP streams to disk.
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME"
for model in facebook/bart-base google-t5/t5-base; do
    huggingface-cli download "$model" --include "*.json" "*.safetensors" "*.model" "*.txt" --max-workers 2
done
echo "models cached in $HF_HOME"
