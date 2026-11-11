#!/usr/bin/env bash
# Download private or public Hugging Face checkpoints into the configured cache.
# Authentication is handled by `hf auth login` or HF_TOKEN; no token is stored
# in this repository.
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
    echo 'The Hugging Face CLI is required. Install huggingface_hub first.' >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <org/model> [<org/model> ...]" >&2
    exit 2
fi

for repo in "$@"; do
    echo "Downloading $repo"
    hf download "$repo"
done
