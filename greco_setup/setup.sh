#!/usr/bin/env bash
# Run from the repository root with an activated Python 3.10 environment.
set -euo pipefail
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for upstream GRECO dependencies"'
mkdir -p external models/greco
if [ ! -d external/greco ]; then
    git clone https://github.com/nusnlp/greco.git external/greco
fi
git -C external/greco checkout --detach 8e509aaacf4e5ec3276d3427371471fd82a7f58b
# Upstream GRECO.score() drops the attention mask, so padding in a batch changes the
# scores of shorter candidates. Apply the one-line fix unless it is already applied.
patch=$PWD/greco_setup/greco-attention-mask.patch
if git -C external/greco apply --reverse --check "$patch" 2>/dev/null; then
    echo "GRECO attention-mask patch already applied"
else
    git -C external/greco apply "$patch"
fi
# GRECO_KEEP_TORCH=1 keeps an already installed torch (e.g. a CUDA build for H100s)
# instead of the pinned torch 1.12.1
if [ "${GRECO_KEEP_TORCH:-0}" = 1 ]; then
    python -c 'import torch'
    python -m pip install -r <(grep -v '^torch==' external/greco/requirements.txt)
else
    python -m pip install -r external/greco/requirements.txt
fi
# Missing from upstream requirements; the DeBERTa-v3 tokenizer needs it (<=3.20 for transformers 4.23)
python -m pip install "protobuf==3.20.3"
if [ ! -s models/greco/checkpoint.bin ]; then
    curl --fail --location --retry 3 --continue-at - \
        https://huggingface.co/nusnlp/GRECO/resolve/main/checkpoint/checkpoint.bin \
        --output models/greco/checkpoint.bin.partial
    mv models/greco/checkpoint.bin.partial models/greco/checkpoint.bin
fi
