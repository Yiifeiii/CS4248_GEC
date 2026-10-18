"""Evaluate the saved dev30 pilot with the bundled NUS M2 scorer.

Run under Python 3.10 (lib2to3 is required):
    python -m greco_setup.evaluate_pilot [--output-dir outputs/greco/bart-t5-gector-dev30]
Requires <output-dir>/gold.m2 with the matching 30 gold records.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from lib2to3.refactor import RefactoringTool, get_fixers_from_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/greco/bart-t5-dev30'))
    root = parser.parse_args().output_dir
    gold = root / 'gold.m2'
    sources = Path('data/dev.src').read_text().splitlines()[:30]
    assert [line[2:] for line in gold.read_text().splitlines() if line.startswith('S ')] == sources
    inputs = json.loads((root / 'k1/input.json').read_text())
    systems = {}
    for system in json.loads((root / 'manifest.json').read_text())['systems']:
        path = root / f'{system}.top1.pred'
        sentences = [next(p['sentence'] for p in r['proposals'] if p['system'] == system) for r in inputs]
        path.write_text('\n'.join(sentences) + '\n')
        systems[f'{system} top-1'] = path
    systems.update({f'GRECO k={k}': root / f'k{k}/greco.pred' for k in (1, 3, 5)})
    rows = []
    # Convert syntax in temporary copies; leave the bundled Python 2 scorer intact.
    converter = RefactoringTool(get_fixers_from_package('lib2to3.fixes'))
    with tempfile.TemporaryDirectory() as tmp:
        for name in ('util.py', 'levenshtein.py', 'm2scorer.py'):
            text = Path('m2scorer/scripts', name).read_text()
            text = str(converter.refactor_string(text, name))
            text = text.replace("puffer.decode('utf8')", 'puffer')
            text = text.replace('line.decode("utf8")', 'line')
            Path(tmp, name).write_text(text)
        for name, path in systems.items():
            assert len(path.read_text().splitlines()) == 30
            output = subprocess.check_output([sys.executable, str(Path(tmp, 'm2scorer.py')),
                                              str(path), str(gold)], text=True)
            print(name + '\n' + output, flush=True)
            (path.parent / (path.stem + '.m2score.txt')).write_text(output)
            values = [float(v) * 100 for v in re.findall(r':\s*([0-9.]+)', output)]
            if len(values) != 3:
                raise ValueError('Unexpected scorer output: ' + output)
            rows.append(dict(system=name, precision=values[0], recall=values[1], f0_5=values[2]))
    with (root / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['system', 'precision', 'recall', 'f0_5'])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'metric': 'NUS M2 corpus-level F0.5 (not ERRANT)', 'sentences': 30,
        'scale': 'percent', 'beta': 0.5, 'max_unchanged_words': 2,
        'gold_source': 'https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz',
        'gold_subset_sha256': hashlib.sha256(gold.read_bytes()).hexdigest(),
        'scorer_compatibility': 'lib2to3 syntax conversion and removal of Python 2 text decode calls; original algorithm unchanged',
        'baselines': 'Same normalized top-1 proposals supplied to GRECO',
        'results': rows,
    }
    (root / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
