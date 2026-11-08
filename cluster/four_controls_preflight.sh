#!/bin/bash
set -euo pipefail
bash cluster/mixed_control_preflight.sh
export FOUR_CONTROL_DATA="$CKPT_ROOT/four-controls-data-20261108"
python -m training.build_four_controls --out "$FOUR_CONTROL_DATA"
python - "$FOUR_CONTROL_DATA" <<'PY'
from pathlib import Path
import json,sys
from transformers.utils import cached_file
from training.build_order_control import sha
root=Path(sys.argv[1]);records={}
for model in ('facebook/bart-base','google-t5/t5-base'):
    path=Path(cached_file(model,'model.safetensors'))
    records[model]={'path':str(path),'weights_sha256':sha(path)}
(root/'initialization.json').write_text(json.dumps(records,indent=2)+'\n')
PY
echo FOUR_CONTROLS_PREFLIGHT_PASSED
