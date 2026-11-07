"""Stage-1 data for GRECO: drop the W&I sentences.

Reads data/curriculum/stage1.{src,tgt,corpus}, written by training/build_data.py, and writes:

  data/greco/stage1.clean.{src,tgt,corpus}   stage 1 without the wi.* corpora
  data/greco/stage1.manifest.json            counts and sha256 of every file written

The W&I+LOCNESS sentences are removed because GRECO was already trained on that set
(Qorib & Ng 2023, Section 5.1).

    python3 -m greco_tuning.prepare_stage1
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from greco_setup.prepare import read_lines

WI_PREFIX = 'wi.'
COLUMNS = ('src', 'tgt', 'corpus')


def write_lines(path, lines):
    path.write_text(''.join(line + '\n' for line in lines), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = {c: read_lines(data_dir / f'stage1.{c}') for c in COLUMNS}
    total = len(columns['src'])
    if any(len(v) != total for v in columns.values()):
        raise ValueError('stage1 src, tgt and corpus have different line counts')

    keep = [i for i, corpus in enumerate(columns['corpus']) if not corpus.startswith(WI_PREFIX)]
    manifest = {'source': {'pairs': total, 'by_corpus': dict(Counter(columns['corpus']))},
                'clean': {'pairs': len(keep), 'removed_prefix': WI_PREFIX, 'files': {}}}

    for c in COLUMNS:
        lines = [columns[c][i] for i in keep]
        manifest['clean']['files'][f'stage1.clean.{c}'] = write_lines(out_dir / f'stage1.clean.{c}', lines)

    (out_dir / 'stage1.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='data/curriculum')
    parser.add_argument('--out-dir', default='data/greco')
    args = parser.parse_args()
    manifest = prepare(args.data_dir, args.out_dir)
    print(f"stage 1: {manifest['source']['pairs']} pairs; clean: {manifest['clean']['pairs']}")


if __name__ == '__main__':
    main()
