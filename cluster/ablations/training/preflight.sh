#!/bin/bash
# CPU compute-node job: verify the shared stage-1 checkpoint, data and tests.
set -euo pipefail
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1

python - <<'PY'
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig
import torch
import transformers

reference = json.loads(Path('experiments/training/bart-base.s1s2/run_info.json').read_text())
versions = dict(torch=torch.__version__, transformers=transformers.__version__)
assert versions == reference['versions'], (versions, reference['versions'])
baseline = json.loads(Path('training/configs/bart-ablations-20261105/baseline.json').read_text())
for key, value in reference['config'].items():
    if key not in ('run', 'init', 'output_dir'):
        assert baseline[key] == value, f'Historical baseline mismatch: {key}'

folder = Path(snapshot_download('CS4248-GEC/gec-bart-base-s1',
    allow_patterns=['*.json', '*.safetensors', '*.txt'], max_workers=2))
expected = {line.split()[1]: line.split()[0] for line in
            Path('experiments/training/checkpoints.sha256').read_text().splitlines()}
digest = hashlib.sha256()
with (folder / 'model.safetensors').open('rb') as source:
    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
        digest.update(block)
assert digest.hexdigest() == expected['bart-base.s1/best/model.safetensors'], 'Stage-1 hash mismatch'
config = AutoConfig.from_pretrained(folder, local_files_only=True)
assert config.model_type == 'bart'
for key in ('dropout', 'attention_dropout', 'activation_dropout'):
    assert getattr(config, key) == 0.1, f'Baseline dropout differs at {key}'
assert config.encoder_layerdrop == config.decoder_layerdrop == 0
record = dict(model='CS4248-GEC/gec-bart-base-s1', snapshot=str(folder),
              weights_sha256=digest.hexdigest(), ablation='bart-ablations-20261105',
              versions=versions, baseline_reference='experiments/training/bart-base.s1s2/run_info.json')
out = Path('experiments/training/bart-ablations-20261105')
out.mkdir(parents=True, exist_ok=True)
(out / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
print('Stage-1 checkpoint verified:', digest.hexdigest(), flush=True)
PY

export HF_HUB_OFFLINE=1
python -m training.check_data
python -m unittest discover -s tests -v
echo BART_ABLATIONS_PREFLIGHT_PASSED
