"""Run BART beam + T5 dev30 combination at k=1,3,5 (no gold evaluation).

Run with the dedicated GRECO environment:
    python -m greco_setup.run_pilot
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
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/greco/bart-t5-dev30'))
    parser.add_argument('--beam-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    sources = Path('data/dev.src').read_text(encoding='utf-8').splitlines()[:30]
    bart = json.loads(Path('outputs/bart/dev30/beam/proposals.json').read_text(encoding='utf-8'))
    t5 = Path('output/t5/dev30/t5.tgt').read_text(encoding='utf-8').splitlines()
    if len(sources) != 30 or [r['source'] for r in bart] != sources or len(t5) != 150:
        raise ValueError('Expected aligned BART and T5 predictions for the first 30 dev sentences')
    if any(len(r['proposals']) < 5 for r in bart):
        raise ValueError('BART must provide at least five hypotheses per sentence')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'source': 'data/dev.src', 'sentence_range': [1, 30],
        'bart': 'outputs/bart/dev30/beam/proposals.json',
        't5': 'output/t5/dev30/t5.tgt', 'systems': ['bart', 't5'],
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
            proposals = [dict(p, sentence=normalize(p['sentence'])) for p in bart[i]['proposals'][:k]]
            proposals.extend({'system': 't5', 'rank': rank + 1, 'sentence': normalize(t5[5 * i + rank])}
                             for rank in range(k))
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
