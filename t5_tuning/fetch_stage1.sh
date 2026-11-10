#!/bin/bash
# Download the frozen stage-1 T5 model (CS4248-GEC/gec-t5-base-s1) into the scratch
# Hugging Face cache, then check its hash against experiments/training/checkpoints.sha256.
# Run on a compute node from the repository root, after `hf auth login` with an
# account that is a member of the CS4248-GEC organization:
#   salloc -t 30 && srun --pty bash
#   conda activate gec-train && bash t5_tuning/fetch_stage1.sh
set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/scratch/${USER:0:1}/$USER}"
export HF_HOME="${HF_HOME:-$SCRATCH/cs4248-gec/hf_cache}"

# Same workaround as cluster/download_models.sh: the Xet backend aborts on xlogin.
export HF_HUB_DISABLE_XET=1

repo=CS4248-GEC/gec-t5-base-s1
snapshot=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$repo'))")
echo "cached in $snapshot"

expected=$(grep 't5-base.s1/best/model.safetensors' experiments/training/checkpoints.sha256 | cut -d' ' -f1)
actual=$(sha256sum "$snapshot/model.safetensors" | cut -d' ' -f1)
if [ "$expected" = "$actual" ]; then
    echo "sha256 matches the recorded stage-1 model"
else
    echo "sha256 MISMATCH: expected $expected, got $actual" >&2
    exit 1
fi
