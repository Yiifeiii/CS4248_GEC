#!/usr/bin/env bash
# Download the team's public Hugging Face checkpoints.
# No Hugging Face account or token is required.
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
    echo 'The Hugging Face CLI is required. Install huggingface_hub first:' >&2
    echo '  python -m pip install -U huggingface_hub' >&2
    exit 1
fi

models=(
  CS4248-GEC/gec-bart-base-s1
  CS4248-GEC/gec-bart-base-s1s2
  CS4248-GEC/gec-t5-base-s1
  CS4248-GEC/gec-t5-base-s1s2
  CS4248-GEC/gec-t5-base-long-s1s2
)

for model in "${models[@]}"; do
  echo "Downloading $model"
  hf download "$model"
done
