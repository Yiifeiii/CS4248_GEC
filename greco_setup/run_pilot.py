"""Run BART beam + T5 (+ optionally GECToR) dev30 combination at k=1,3,5 (no gold evaluation).

Run with the dedicated GRECO environment:
    python -m greco_setup.run_pilot                 # BART + T5
    python -m greco_setup.run_pilot --with-gector   # BART + T5 + GECToR

Inputs are the shared prediction files outputs/<system>/dev30*.tgt (k lines per
source sentence, best first). GECToR contributes its single best correction.
"""

import argparse
import json
from pathlib import Path
import re

from greco_setup.combine import GrecoScorer, combine, read_record


def normalize(text):
    """Basic spacing alignment; preserve punctuation runs such as '..'.

    This is a lightweight tokenizer, not ERRANT linguistic alignment.
    """
    text = re.sub(r'([.,!?;:"()]+)', r' \1 ', text)
    text = re.sub(r"(?i)\b(\w+)(n't|'s|'re|'ve|'ll|'d|'m)\b", r'\1 \2', text)
    return ' '.join(text.split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        help='default: outputs/greco/bart-t5-dev30 (or bart-t5-gector-dev30)')
    parser.add_argument('--with-gector', action='store_true', help='add GECToR 1-best proposals')
    parser.add_argument('--beam-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    systems = ['bart', 't5'] + (['gector'] if args.with_gector else [])
    if args.output_dir is None:
        args.output_dir = Path('outputs/greco/' + '-'.join(systems) + '-dev30')
    sources = Path('data/dev.src').read_text(encoding='utf-8').splitlines()[:30]
    paths = {'bart': 'outputs/bart/dev30.beam.tgt', 't5': 'outputs/t5/dev30.tgt',
             'gector': 'outputs/gector/dev30.tgt'}
    preds = {s: Path(paths[s]).read_text(encoding='utf-8').splitlines() for s in systems}
    expected = {'bart': 150, 't5': 150, 'gector': 30}
    if len(sources) != 30 or any(len(preds[s]) != expected[s] for s in systems):
        raise ValueError('Expected 5 BART/T5 lines and 1 GECToR line for each of the first 30 dev sentences')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'source': 'data/dev.src', 'sentence_range': [1, 30],
        **{s: paths[s] for s in systems}, 'systems': systems,
        'k': [1, 3, 5], 'beam_size': args.beam_size, 'batch_size': args.batch_size,
        'device': 'cpu', 'normalization': 'Basic punctuation and contraction spacing; preserves punctuation runs',
        'alignment': 'Token-level SequenceMatcher, not ERRANT',
        'evaluation': None,
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    inputs = {}
    for k in (1, 3, 5):
        records = []
        for i, source in enumerate(sources):
            proposals = [{'system': s, 'rank': rank + 1, 'sentence': normalize(preds[s][5 * i + rank])}
                         for s in ('bart', 't5') for rank in range(k)]
            if args.with_gector:
                proposals.append({'system': 'gector', 'rank': 1, 'sentence': normalize(preds['gector'][i])})
            records.append({'source': source, 'proposals': proposals})
        folder = args.output_dir / f'k{k}'
        folder.mkdir(exist_ok=True)
        (folder / 'input.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
        inputs[k] = records
    scorer = GrecoScorer('external/greco', 'models/greco/checkpoint.bin', 'cpu')
    for k, records in inputs.items():
        results = []
        for i, record in enumerate(records, 1):
            results.append(combine(*read_record(record), scorer=scorer,
                                   beam_size=args.beam_size, batch_size=args.batch_size))
            print(f'k={k}: {i}/30', flush=True)
        folder = args.output_dir / f'k{k}'
        (folder / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
        (folder / 'greco.pred').write_text('\n'.join(r['best_sentence'] for r in results) + '\n')


if __name__ == '__main__':
    main()
