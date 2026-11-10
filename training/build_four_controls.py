"""Build matched-exposure and uniformly exposed training controls."""
from collections import Counter
import argparse
import json
from pathlib import Path
import random

from training.control_data import sha


def streams(stage1,stage2,seed=42):
    matched=list(stage1)*3+list(stage2)*5
    random.Random(seed).shuffle(matched)
    unique=list(dict.fromkeys([*stage1,*stage2]))
    output={'matched':matched}
    for epochs in (3,5):
        rows=[]
        for epoch in range(epochs):
            block=unique.copy()
            random.Random(seed+epoch).shuffle(block)
            rows.extend(block)
        output[f'uniform{epochs}']=rows
        assert Counter(rows)==Counter({p:epochs for p in unique})
    expected=Counter()
    for pair in stage1:expected[pair]+=3
    for pair in stage2:expected[pair]+=5
    assert Counter(matched)==expected
    return output,len(unique)


def build(curriculum,out):
    reference=json.loads((curriculum/'manifest.json').read_text())
    stages=[];hashes={}
    for name in ('stage1','stage2'):
        cols=[]
        for suffix in ('src','tgt'):
            p=curriculum/f'{name}.{suffix}'
            digest=sha(p)
            if digest!=reference[name]['sha256'][suffix]:raise ValueError(f'{p}: hash mismatch')
            hashes[f'{name}.{suffix}']=digest
            cols.append(p.read_text().splitlines())
        if len(cols[0])!=len(cols[1]) or len(cols[0])!=reference[name]['pairs']:
            raise ValueError(f'{name}: count mismatch')
        stages.append(list(zip(*cols)))
    data,n_unique=streams(*stages)
    out.mkdir(parents=True,exist_ok=True)
    manifest={'stage_pairs':[len(s) for s in stages],'unique_pairs':n_unique,
              'input_sha256':hashes,'seed':42,'dev_test_in_training':False,'streams':{}}
    for name,rows in data.items():
        folder=out/name;folder.mkdir(exist_ok=True)
        record={'pairs':len(rows),'logical_passes':None if name=='matched' else int(name[-1])}
        for col,suffix in enumerate(('src','tgt')):
            p=folder/f'train.{suffix}'
            with p.open('w') as f:
                for row in rows:f.write(row[col]+'\n')
            record[suffix+'_sha256']=sha(p)
        manifest['streams'][name]=record
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum',type=Path,default=Path('data/curriculum'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.curriculum,args.out),indent=2))


if __name__=='__main__':main()
