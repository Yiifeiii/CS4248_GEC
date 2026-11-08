"""Build an exposure-matched, stage-free training mixture; never read dev/test."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def mixed_pairs(stage1, stage2, seed=42):
    # Match the prescribed 3 stage-1 + 5 stage-2 epochs, including repeated W&I
    # examples already present in stage1. Deduplication would change exposure.
    pairs = list(stage1) * 3 + list(stage2) * 5
    random.Random(seed).shuffle(pairs)
    return pairs


def build(root, output, seed=42):
    manifest = json.loads((root / 'manifest.json').read_text())
    stages = []
    hashes = {}
    for name in ('stage1', 'stage2'):
        columns = []
        for suffix in ('src', 'tgt'):
            path = root / f'{name}.{suffix}'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != manifest[name]['sha256'][suffix]:
                raise ValueError(f'{path}: data hash mismatch; rebuild/normalize curriculum data first')
            hashes[f'{name}.{suffix}'] = digest
            columns.append(path.read_text().splitlines())
        if len(columns[0]) != len(columns[1]) or len(columns[0]) != manifest[name]['pairs']:
            raise ValueError(f'{name}: source/target counts do not match manifest')
        stages.append(list(zip(*columns)))
    pairs = mixed_pairs(*stages, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    for i, suffix in enumerate(('src', 'tgt')):
        path = Path(f'{output}.{suffix}')
        with path.open('w') as stream:
            for pair in pairs:
                stream.write(pair[i] + '\n')
        hashes[f'mixed.{suffix}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {'seed': seed, 'stage1_repetitions': 3, 'stage2_repetitions': 5,
              'stage_pairs': [len(s) for s in stages], 'pairs': len(pairs), 'sha256': hashes,
              'deduplicated': False, 'uses_dev_or_test': False}
    Path(f'{output}.manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum', type=Path, default=Path('data/curriculum'))
    parser.add_argument('--output', type=Path, default=Path('data/mixed-control/train'))
    args = parser.parse_args()
    print(json.dumps(build(args.curriculum, args.output), indent=2))


if __name__ == '__main__': main()
