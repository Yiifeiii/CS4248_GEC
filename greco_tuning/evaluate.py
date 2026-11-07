"""Score GRECO runs on dev_tune, next to the single-system baselines.

For each run, reads greco_tuning/runs/<name>/vote<v>/k<k>/greco.pred and scores it against
data/curriculum/dev_tune.m2 with ERRANT and M2 (training/metrics.py). Writes
greco_tuning/runs/<name>/metrics.csv and prints the table. With two runs, the second is
compared with the first at every setting.

    python3 -m greco_tuning.evaluate --select s1-epoch1 s1-epoch2 s1-epoch3
    python3 -m greco_tuning.evaluate original s1-epoch2
"""

import argparse
import csv
import json
from pathlib import Path

from greco_setup.prepare import read_lines
from greco_tuning.make_m2 import POOL, SRC
from training.metrics import score

GOLD = Path('data/curriculum/dev_tune.m2')
RUNS = Path('greco_tuning/runs')
VOTES = ('0.4', '0')
KS = (1, 3, 5)
SELECT_VOTE, SELECT_K = '0.4', 3  # the mini experiment's best setting
FIELDS = ['run', 'vote_coef', 'k', 'inputs', 'errant_p', 'errant_r', 'errant_f0_5', 'm2_p', 'm2_r', 'm2_f0_5']


def row(run, vote, k, inputs, result):
    e, m = result['errant']['correction'], result['m2']
    return {'run': run, 'vote_coef': vote, 'k': k, 'inputs': inputs,
            'errant_p': e['precision'] * 100, 'errant_r': e['recall'] * 100, 'errant_f0_5': e['f0_5'] * 100,
            'm2_p': m['precision'] * 100, 'm2_r': m['recall'] * 100, 'm2_f0_5': m['f0_5'] * 100}


def baselines(sources):
    rows = []
    for name in ('bart', 't5', 'gector'):
        lines = read_lines(POOL / f'{name}.tgt')
        k = len(lines) // len(sources)
        rows.append(row(f'{name} top-1', '-', '-', 1, score(SRC, lines[::k], GOLD)))
    return rows


def run_rows(name, sources, settings=None):
    rows = []
    for vote, k in settings or [(v, k) for v in VOTES for k in KS]:
        pred = RUNS / name / f'vote{vote}' / f'k{k}' / 'greco.pred'
        hyps = read_lines(pred)
        if len(hyps) != len(sources):
            raise ValueError(f'{pred}: {len(hyps)} lines, expected {len(sources)}')
        rows.append(row(name, vote, k, 1 + 2 * k, score(SRC, hyps, GOLD)))
    return rows


def select(names, sources):
    """Pick the epoch with the best ERRANT F0.5 at vote 0.4, k 3, as BART and T5 keep best/."""
    rows = [run_rows(name, sources, [(SELECT_VOTE, SELECT_K)])[0] for name in names]
    print_table(rows)
    best = max(rows, key=lambda r: r['errant_f0_5'])
    print(f"\nbest: {best['run']} (ERRANT F0.5 {best['errant_f0_5']:.2f})")
    (RUNS / 'selection.json').write_text(json.dumps({'best': best['run'], 'rows': rows}, indent=2) + '\n',
                                         encoding='utf-8')


def print_table(rows):
    print(f"{'run':<16}{'vote':>6}{'k':>4}{'ERRANT F0.5':>13}{'M2 F0.5':>10}")
    for r in rows:
        print(f"{r['run']:<16}{str(r['vote_coef']):>6}{str(r['k']):>4}"
              f"{r['errant_f0_5']:>13.2f}{r['m2_f0_5']:>10.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run names under greco_tuning/runs/')
    parser.add_argument('--select', action='store_true',
                        help='compare epoch runs at vote 0.4, k 3 and write runs/selection.json')
    args = parser.parse_args()
    sources = read_lines(SRC)
    if args.select:
        select(args.runs, sources)
        return

    all_rows = baselines(sources)
    by_run = {}
    for name in args.runs:
        by_run[name] = run_rows(name, sources)
        with open(RUNS / name / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(by_run[name])
        all_rows += by_run[name]

    print_table(all_rows)
    if len(args.runs) == 2:
        first, second = (by_run[name] for name in args.runs)
        print(f'\nchange in ERRANT F0.5 ({args.runs[1]} minus {args.runs[0]}):')
        for a, b in zip(first, second):
            print(f"  vote {a['vote_coef']:>3}  k {a['k']}:  {b['errant_f0_5'] - a['errant_f0_5']:+.2f}")


if __name__ == '__main__':
    main()
