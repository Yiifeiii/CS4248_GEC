#!/bin/bash
# CPU checks before any GPU run: the data build, the tests, then tune.py preflight (configs,
# software versions, starting models and the stage-1 model's recorded hash).
# tune.py submit submits this first and makes every training run depend on it passing:
#   sbatch --partition=normal --gpus=0 ... cluster/job.sbatch t5_tuning/preflight.sh --stage 2 --phase 1
set -euo pipefail

python -m training.check_data
python -m unittest tests.test_training tests.test_t5_tuning
python t5_tuning/tune.py preflight "$@"
echo T5_TUNING_PREFLIGHT_PASSED
