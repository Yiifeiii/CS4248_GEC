#!/bin/bash
set -euo pipefail
test -f m2_to_parallel.py
test -f m2scorer/scripts/util.py
python - <<'PY'
import json
from pathlib import Path
import torch, transformers
from training.build_data import Cleaner
from training.metrics import score
reference=json.loads(Path('experiments/training/bart-base.s1s2/run_info.json').read_text())
assert reference['versions']=={'torch':torch.__version__,'transformers':transformers.__version__}
PY
python -m unittest discover -s tests -v
python -m training.check_data
echo MIXED_CONTROL_PREFLIGHT_PASSED
