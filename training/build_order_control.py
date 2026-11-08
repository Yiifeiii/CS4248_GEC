"""Build two streams with identical examples/exposures and different ordering."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exposure_indices(n1, n2, seed=42):
    if n1 < 1 or n2 < 1:
        raise ValueError('Both stages need examples')
    rng = random.Random(seed)
    curriculum, boundaries = [], []
    # Independent within-round shuffling retains ordinary per-epoch randomization,
    # but stage1 examples always precede stage2 examples.
    for start, count, repetitions in ((0, n1, 3), (n1, n2, 5)):
        for _ in range(repetitions):
            block = list(range(start, start+count))
            rng.shuffle(block)
            curriculum.extend(block)
            boundaries.append(len(curriculum))
    shuffled = curriculum.copy()
    random.Random(seed+1).shuffle(shuffled)
    assert Counter(curriculum) == Counter(shuffled)
    return curriculum, shuffled, boundaries


def build(curriculum_dir, output, seed=42):
    reference = json.loads((curriculum_dir/'manifest.json').read_text())
    pairs, hashes, sizes = [], {}, []
    for stage in ('stage1','stage2'):
        columns = []
        for suffix in ('src','tgt'):
            p = curriculum_dir/f'{stage}.{suffix}'
            digest = sha(p)
            if digest != reference[stage]['sha256'][suffix]:
                raise ValueError(f'{p}: curriculum hash mismatch')
            hashes[f'{stage}.{suffix}'] = digest
            columns.append(p.read_text().splitlines())
        if len(columns[0]) != len(columns[1]) or len(columns[0]) != reference[stage]['pairs']:
            raise ValueError(f'{stage}: pair-count mismatch')
        sizes.append(len(columns[0]))
        pairs.extend(zip(*columns))
    ordered, shuffled, boundaries = exposure_indices(*sizes,seed)
    output.mkdir(parents=True,exist_ok=True)
    record = {'seed':seed, 'shuffle_seed':seed+1, 'stage_pairs':sizes,
              'round_sizes':[sizes[0]]*3+[sizes[1]]*5, 'round_end_examples':boundaries,
              'total_examples':len(ordered), 'same_exposure_indices':True,
              'input_sha256':hashes, 'streams':{}, 'dev_test_in_training':False}
    for name, indices in [('curriculum',ordered),('shuffled',shuffled)]:
        folder = output/name
        folder.mkdir(exist_ok=True)
        for col,suffix in enumerate(('src','tgt')):
            path = folder/f'train.{suffix}'
            with path.open('w') as stream:
                for i in indices: stream.write(pairs[i][col]+'\n')
            record['streams'].setdefault(name,{})[suffix+'_sha256'] = sha(path)
        ids = folder/'exposure.ids'
        ids.write_text(''.join(str(i)+'\n' for i in indices))
        record['streams'][name]['ids_sha256'] = sha(ids)
    (output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum',type=Path,default=Path('data/curriculum'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.curriculum,args.out),indent=2))


if __name__=='__main__': main()
