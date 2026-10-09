#!/usr/bin/env bash
# Creates gector/.venv and downloads the official weights into gector/models.
# Run from anywhere: bash gector/setup.sh (PYTHON defaults to python3, must be >= 3.11).
set -euo pipefail
cd "$(dirname "$0")"
python=${PYTHON:-python3}
"$python" -c 'import sys; assert sys.version_info >= (3, 11), "Use Python 3.11 or newer for gotutiyan/gector"'
[ -d .venv ] || "$python" -m venv .venv
.venv/bin/python -m pip install "git+https://github.com/gotutiyan/gector@a4a342ed7c4733a34263443b605b7edffbef7099"

data=https://raw.githubusercontent.com/grammarly/gector/master/data
mkdir -p models/output_vocabulary
for f in labels.txt d_tags.txt non_padded_namespaces.txt; do
    curl --fail --location --retry 3 "$data/output_vocabulary/$f" --output "models/output_vocabulary/$f"
done
curl --fail --location --retry 3 "$data/verb-form-vocab.txt" --output models/verb-form-vocab.txt
if [ ! -s models/roberta_1_gectorv2.th ]; then
    curl --fail --location --retry 3 \
        https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th \
        --output models/roberta_1_gectorv2.th.partial
    mv models/roberta_1_gectorv2.th.partial models/roberta_1_gectorv2.th
fi
