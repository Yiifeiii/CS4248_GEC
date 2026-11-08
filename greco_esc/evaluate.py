"""Score every GRECO + ESC output on dev_test, next to the single systems and ESC alone.

Scores each prediction against data/curriculum/dev_test.m2 with ERRANT (primary) and the M2
scorer (training/metrics.py), then writes

  greco_esc/runs/dev_test/results.csv   one row per output
  greco_esc/runs/dev_test/results.md    ERRANT F0.5 by k: GRECO with and without ESC and voting

Run in gec-train from the repository root (run.sh does this as its last step):

    python3 -m greco_esc.evaluate
"""

import csv
from pathlib import Path

from greco_esc import settings
from greco_setup.prepare import read_lines

FIELDS = ['system', 'k', 'vote_coef', 'esc', 'inputs',
          'errant_p', 'errant_r', 'errant_f0_5', 'm2_p', 'm2_r', 'm2_f0_5']


def row(system, k, vote, esc, inputs, result):
    e, m = result['errant']['correction'], result['m2']
    return {'system': system, 'k': k, 'vote_coef': vote, 'esc': esc, 'inputs': inputs,
            'errant_p': e['precision'] * 100, 'errant_r': e['recall'] * 100, 'errant_f0_5': e['f0_5'] * 100,
            'm2_p': m['precision'] * 100, 'm2_r': m['recall'] * 100, 'm2_f0_5': m['f0_5'] * 100}


def outputs():
    """(system, k, vote_coef, esc, path) for every output the pipeline writes."""
    split = settings.ROOT / settings.TEST_SPLIT
    out = [(name, '-', '-', '-', split / 'text' / f'{name}.txt') for name in ('bart-1', 't5-1', 'gector-1')]
    for k in settings.KS:
        out.append(('esc', k, '-', 'yes', split / f'k{k}' / 'esc' / 'pred.txt'))
    for name, k, vote, esc in settings.runs():
        out.append(('greco', k, f'{vote:g}', 'yes' if esc else 'no', settings.pred_path(k, name)))
    return out


def table(rows):
    """Markdown: one line per k, ERRANT F0.5 (P / R) for each combination."""
    columns = [('greco', '0', 'no', 'GRECO, no vote'), ('greco', 'tuned', 'no', 'GRECO, vote'),
               ('greco', '0', 'yes', 'GRECO + ESC, no vote'), ('greco', 'tuned', 'yes', 'GRECO + ESC, vote'),
               ('esc', '-', 'yes', 'ESC alone')]
    lines = ['| k | vote_coef | ' + ' | '.join(c[3] for c in columns) + ' |',
             '|---|---|' + '---|' * len(columns)]
    for k in settings.KS:
        cells = []
        for system, vote, esc, _ in columns:
            want = f'{settings.VOTE[k]:g}' if vote == 'tuned' else vote
            match = [r for r in rows if r['system'] == system and r['k'] == k
                     and r['vote_coef'] == want and r['esc'] == esc]
            cells.append(f"{match[0]['errant_f0_5']:.1f} ({match[0]['errant_p']:.1f} / {match[0]['errant_r']:.1f})"
                         if match else 'missing')
        lines.append(f'| {k} | {settings.VOTE[k]:g} | ' + ' | '.join(cells) + ' |')
    singles = [r for r in rows if r['k'] == '-']
    lines += ['', 'Single systems (top-1): ' + ', '.join(
        f"{r['system']} {r['errant_f0_5']:.1f} ({r['errant_p']:.1f} / {r['errant_r']:.1f})" for r in singles)]
    return '\n'.join(lines) + '\n'


def main():
    from training.metrics import score  # needs ERRANT, so imported only when scoring
    src = settings.DATA / f'{settings.TEST_SPLIT}.src'
    gold = settings.DATA / f'{settings.TEST_SPLIT}.m2'
    n = len(read_lines(src))
    rows, missing = [], []
    for system, k, vote, esc, path in outputs():
        if not path.exists():
            missing.append(path)
            continue
        hyps = read_lines(path)
        if len(hyps) != n:
            raise ValueError(f'{path}: {len(hyps)} lines, expected {n}')
        inputs = 1 if k == '-' else len(settings.inputs(k))
        rows.append(row(system, k, vote, esc, inputs, score(src, hyps, gold)))
        print(f"{system:<6} k={k!s:<2} vote={vote:<4} esc={esc:<3} "
              f"ERRANT F0.5 {rows[-1]['errant_f0_5']:.2f}  M2 F0.5 {rows[-1]['m2_f0_5']:.2f}")
    out = settings.ROOT / settings.TEST_SPLIT
    with open(out / 'results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: f'{v:.2f}' if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    (out / 'results.md').write_text(table(rows), encoding='utf-8')
    print('\n' + table(rows))
    for path in missing:
        print('missing:', path)
    print(f'wrote {out / "results.csv"} and {out / "results.md"}')


if __name__ == '__main__':
    main()
