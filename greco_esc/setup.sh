#!/bin/bash
# One-time setup for ESC, inside the activated "greco" environment (Python 3.10), from the
# repository root:
#   conda activate greco && bash greco_esc/setup.sh
# Clones upstream ESC at a pinned commit into external/esc (git-ignored, like external/greco)
# and adds scikit-learn, the one ESC dependency the greco environment lacks. ESC's other
# dependency, ERRANT, is not needed: prepare.py writes every M2 file ESC reads.
set -euo pipefail
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "activate the greco environment first"'
mkdir -p external
[ -d external/esc ] || git clone https://github.com/nusnlp/esc.git external/esc
git -C external/esc checkout --detach cdefd3bc6e48dbe83f08c88d1690d2c3c956cb9e
python -m pip install "scikit-learn==1.3.2"
python -c "import sklearn, torch; print('esc ok: scikit-learn', sklearn.__version__, 'torch', torch.__version__)"
