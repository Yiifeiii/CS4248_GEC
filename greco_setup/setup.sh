#!/usr/bin/env bash
# Run from the repository root with an activated Python 3.10 environment.
set -euo pipefail
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for upstream GRECO dependencies"'
mkdir -p external models/greco
if [ ! -d external/greco ]; then
    git clone https://github.com/nusnlp/greco.git external/greco
fi
git -C external/greco checkout --detach 8e509aaacf4e5ec3276d3427371471fd82a7f58b
python -m pip install -r external/greco/requirements.txt
if [ ! -s models/greco/checkpoint.bin ]; then
    curl --fail --location --retry 3 \
        https://huggingface.co/nusnlp/GRECO/resolve/main/checkpoint/checkpoint.bin \
        --output models/greco/checkpoint.bin.partial
    mv models/greco/checkpoint.bin.partial models/greco/checkpoint.bin
fi
