#!/bin/bash
set -euo pipefail
test -f m2_to_parallel.py
test -f m2scorer/scripts/util.py
python - <<'PY'
import json
from pathlib import Path
import torch,transformers
reference=json.loads(Path('experiments/training/bart-base.s1s2/run_info.json').read_text())
assert reference['versions']=={'torch':torch.__version__,'transformers':transformers.__version__}
PY
python -m unittest discover -s tests -v
python -m training.check_data
export FOUR_CONTROL_DATA="$CKPT_ROOT/four-controls-data-20261108"
python -m training.build_four_controls --out "$FOUR_CONTROL_DATA"
python - "$FOUR_CONTROL_DATA" <<'PY'
from pathlib import Path
import json,sys
from transformers.utils import cached_file
from training.control_data import sha
root=Path(sys.argv[1]);records={}
for model in ('facebook/bart-base','google-t5/t5-base'):
    path=Path(cached_file(model,'model.safetensors'))
    records[model]={'path':str(path),'weights_sha256':sha(path)}
(root/'initialization.json').write_text(json.dumps(records,indent=2)+'\n')
PY
echo FOUR_CONTROLS_PREFLIGHT_PASSED
