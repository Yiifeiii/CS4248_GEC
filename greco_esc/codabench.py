"""Zip each finished BEA-2019 test output for the BEA-2019 Codabench task.

    python3 -m greco_esc.codabench            # from the repository root, after test.sh

Codabench takes a zip holding one .txt file of 4,477 lines (one corrected sentence per line). This
writes codabench/<name>.zip for every finished output under greco_esc/runs/bea19:

  k<k>/<run>/pred.txt   GRECO runs and ESC alone  -> bea19_k<k>_<run>.zip ('+' becomes '-')
  text/<system>-1.txt   the single systems       -> bea19_<system>-1.zip

Outputs that already have a zip are skipped, so it can be rerun as more runs finish (--redo
repackages everything). Outputs without exactly 4,477 lines are still running and are skipped too.
"""

import sys
import zipfile
from pathlib import Path

from greco_esc import settings

N_SENTENCES = 4477
OUT = Path('codabench')


def name(path, runs):
    """The zip's name for one output path."""
    if path.parent == runs / 'text':
        return f'bea19_{path.stem}'
    return f'bea19_{path.parent.parent.name}_{path.parent.name}'.replace('+', '-')


def main(redo=False):
    runs = settings.ROOT / 'bea19'
    OUT.mkdir(exist_ok=True)
    outputs = sorted(runs.glob('k*/*/pred.txt')) + sorted((runs / 'text').glob('*-1.txt'))
    for path in outputs:
        zip_path = OUT / f'{name(path, runs)}.zip'
        if zip_path.exists() and not redo:
            continue
        lines = path.read_text(encoding='utf-8').rstrip('\n').split('\n')
        if len(lines) != N_SENTENCES:
            print(f'skip {path}: {len(lines)} lines, not {N_SENTENCES} (still running?)')
            continue
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr(f'{zip_path.stem}.txt', '\n'.join(lines) + '\n')
        print(f'packed {zip_path}')
    print(f'{len(list(OUT.glob("*.zip")))} zips in {OUT}/')


if __name__ == '__main__':
    main(redo='--redo' in sys.argv[1:])
