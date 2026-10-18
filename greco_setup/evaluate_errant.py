"""Evaluate a dev30 pilot with ERRANT and compute oracle recall of the edit pool.

Run from the repo root with any Python that has errant installed (not the GRECO env):
    python -m greco_setup.evaluate_errant --output-dir outputs/greco/bart-t5-gector-dev30
Requires <output-dir>/gold.m2 and k1/, k3/, k5/ from run_pilot.py. Writes <output-dir>/errant.json.

Oracle recall: the share of gold edits that appear in any proposal pooled for a
sentence, i.e. the best recall GRECO could reach by choosing among them.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import errant

SYSTEM_FILES = {
    'bart': ('outputs/bart/dev30.beam.tgt', 5),
    't5': ('outputs/t5/dev30.tgt', 5),
    'gector': ('outputs/gector/dev30.tgt', 1),
}


def compare(src, hyp, gold, tmp):
    """Return ERRANT span-based correction and detection scores."""
    tmp = Path(tmp)
    (tmp / 'hyp.txt').write_text('\n'.join(hyp) + '\n', encoding='utf-8')
    tools = Path(sys.executable).parent
    subprocess.run([tools / 'errant_parallel', '-orig', src, '-cor', tmp / 'hyp.txt', '-out', tmp / 'hyp.m2'],
                   check=True, capture_output=True)
    scores = {}
    for mode, flags in (('correction', []), ('detection', ['-ds'])):
        out = subprocess.run([tools / 'errant_compare', '-hyp', tmp / 'hyp.m2', '-ref', gold, *flags],
                             check=True, capture_output=True, text=True).stdout.splitlines()
        values = out[next(i for i, line in enumerate(out) if line.startswith('TP')) + 1].split()
        scores[mode] = dict(zip(['tp', 'fp', 'fn', 'precision', 'recall', 'f0_5'], map(float, values)))
    return scores


def read_gold_edits(path):
    sentences = []
    for block in path.read_text(encoding='utf-8').strip().split('\n\n'):
        edits = set()
        for line in block.split('\n')[1:]:
            fields = line[2:].split('|||')
            # noop marks a correct sentence; UNK has no correction to match
            if fields[1] in ('noop', 'UNK'):
                continue
            start, end = map(int, fields[0].split())
            edits.add((start, end, ' '.join(fields[2].split())))
        sentences.append(edits)
    return sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/greco/bart-t5-gector-dev30'))
    root = parser.parse_args().output_dir
    gold = root / 'gold.m2'
    src = Path('outputs/dev30.src')
    sources = src.read_text(encoding='utf-8').splitlines()

    systems = {}
    for name in json.loads((root / 'manifest.json').read_text())['systems']:
        path, k = SYSTEM_FILES[name]
        systems[f'{name} top-1'] = Path(path).read_text(encoding='utf-8').splitlines()[::k]
    for k in (1, 3, 5):
        systems[f'GRECO k={k}'] = (root / f'k{k}/greco.pred').read_text(encoding='utf-8').splitlines()

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, hyp in systems.items():
            assert len(hyp) == len(sources), name
            results[name] = compare(src, hyp, gold, tmp)
            results[name]['sentences_changed'] = sum(h != s for h, s in zip(hyp, sources))

    annotator = errant.load('en')
    gold_edits = read_gold_edits(gold)
    oracle = {}
    for k in (1, 3, 5):
        records = json.loads((root / f'k{k}/input.json').read_text(encoding='utf-8'))
        pooled = reachable = 0
        for record, gold_set in zip(records, gold_edits):
            orig = annotator.parse(record['source'])
            pool = set()
            for proposal in record['proposals']:
                cor = annotator.parse(proposal['sentence'])
                pool |= {(e.o_start, e.o_end, e.c_str) for e in annotator.annotate(orig, cor)}
            pooled += len(pool)
            reachable += len(pool & gold_set)
        total = sum(len(g) for g in gold_edits)
        oracle[f'k={k}'] = {'pooled_edits': pooled, 'gold_reachable': reachable,
                            'gold_total': total, 'oracle_recall': reachable / total}

    report = {'metric': 'ERRANT span-based correction and detection (-ds)', 'sentences': len(sources),
              'results': results, 'oracle': oracle}
    (root / 'errant.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for name, r in results.items():
        c = r['correction']
        print(f"{name:<13} P {c['precision'] * 100:5.1f}  R {c['recall'] * 100:5.1f}  F0.5 {c['f0_5'] * 100:5.1f}")
    for k, o in oracle.items():
        print(f"{k}: {o['gold_reachable']}/{o['gold_total']} gold edits reachable "
              f"({o['oracle_recall'] * 100:.1f}%), {o['pooled_edits']} pooled edits")


if __name__ == '__main__':
    main()
