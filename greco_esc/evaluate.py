"""Score every GRECO + ESC output, next to the single systems and ESC alone.

Scores each prediction with ERRANT (primary) and the M2 scorer (training/metrics.py), then writes

  greco_esc/runs/<split>/results.csv   one row per output
  greco_esc/runs/<split>/results.md    ERRANT F0.5 by k: GRECO with and without ESC and voting,
                                       then the three vote modes (hypothesis, system, rank)

<split> is dev_test (the default, against data/curriculum/dev_test.m2), dev_tune, or dev: the full
dev set, data/dev.src. For dev, each output's dev_tune and dev_test halves are merged into
data/dev.src order (data/curriculum/<half>.ids), and the reference is rebuilt the same way from
the two halves' M2 files (data/dev.m2 is in Git LFS and may be a pointer file on the cluster).

Run in gec-train from the repository root (run.sh does this as its last step):

    python3 -m greco_esc.evaluate          # dev_test
    python3 -m greco_esc.evaluate dev      # the full dev set
"""

import csv
import sys
from pathlib import Path

from greco_esc import settings
from greco_setup.prepare import read_lines

FIELDS = ['system', 'k', 'vote_mode', 'vote_coef', 'esc', 'inputs',
          'errant_p', 'errant_r', 'errant_f0_5', 'm2_p', 'm2_r', 'm2_f0_5']


def row(system, k, mode, vote, esc, inputs, result):
    e, m = result['errant']['correction'], result['m2']
    return {'system': system, 'k': k, 'vote_mode': mode, 'vote_coef': vote, 'esc': esc, 'inputs': inputs,
            'errant_p': e['precision'] * 100, 'errant_r': e['recall'] * 100, 'errant_f0_5': e['f0_5'] * 100,
            'm2_p': m['precision'] * 100, 'm2_r': m['recall'] * 100, 'm2_f0_5': m['f0_5'] * 100}


def outputs():
    """(system, k, vote_mode, vote_coef, esc, path) for every output, path relative to runs/<split>."""
    out = [(name, '-', '-', '-', '-', Path('text') / f'{name}.txt') for name in ('bart-1', 't5-1', 'gector-1')]
    for k in settings.KS:
        out.append(('esc', k, '-', '-', 'yes', Path(f'k{k}') / 'esc' / 'pred.txt'))
    for name, k, vote, esc, mode in settings.runs():
        out.append(('greco', k, mode if vote else '-', f'{vote:g}', 'yes' if esc else 'no',
                    Path(f'k{k}') / name / 'pred.txt'))
    return out


def ids(half):
    return [int(i) for i in read_lines(settings.DATA / f'{half}.ids')]


def merge(parts):
    """Lines from each half, put back in data/dev.src order. parts: {half: lines}."""
    order = {half: ids(half) for half in parts}
    merged = [None] * sum(len(o) for o in order.values())
    for half, lines in parts.items():
        if len(lines) != len(order[half]):
            raise ValueError(f'{half}: {len(lines)} lines, expected {len(order[half])}')
        for i, line in zip(order[half], lines):
            merged[i] = line
    if None in merged:
        raise ValueError('the dev_tune and dev_test ids do not cover the full dev set')
    return merged


def m2_blocks(path):
    return Path(path).read_text(encoding='utf-8').strip().split('\n\n')


def reference(split):
    """(source path, gold M2 path) for split; for dev, written to runs/dev from the two halves."""
    if split != settings.FULL_SPLIT:
        return settings.DATA / f'{split}.src', settings.DATA / f'{split}.m2'
    out = settings.ROOT / split
    out.mkdir(parents=True, exist_ok=True)
    src, gold = out / 'source.txt', out / 'reference.m2'
    sources = merge({h: read_lines(settings.DATA / f'{h}.src') for h in settings.HALVES})
    blocks = merge({h: m2_blocks(settings.DATA / f'{h}.m2') for h in settings.HALVES})
    if [b.split('\n')[0][2:] for b in blocks] != sources:
        raise ValueError('merged M2 sources differ from the merged source sentences')
    src.write_text('\n'.join(sources) + '\n', encoding='utf-8')
    gold.write_text('\n\n'.join(blocks) + '\n\n', encoding='utf-8')
    return src, gold


def load(split, rel):
    """An output's lines on split, or None if it (or, for dev, either half) is missing."""
    if split == settings.FULL_SPLIT:
        parts = {h: load(h, rel) for h in settings.HALVES}
        return None if None in parts.values() else merge(parts)
    path = settings.ROOT / split / rel
    return read_lines(path) if path.exists() else None


def cell(rows, **want):
    """ERRANT F0.5 (P / R) of the one row matching want, or 'missing'."""
    match = [r for r in rows if all(r[key] == value for key, value in want.items())]
    return f"{match[0]['errant_f0_5']:.1f} ({match[0]['errant_p']:.1f} / {match[0]['errant_r']:.1f})" \
        if match else 'missing'


def table(rows):
    """Markdown: one line per k, ERRANT F0.5 (P / R) for each combination."""
    columns = [('greco', '0', 'no', 'GRECO, no vote'), ('greco', 'tuned', 'no', 'GRECO, vote'),
               ('greco', '0', 'yes', 'GRECO + ESC, no vote'), ('greco', 'tuned', 'yes', 'GRECO + ESC, vote'),
               ('esc', '-', 'yes', 'ESC alone')]
    lines = ['| k | vote_coef | ' + ' | '.join(c[3] for c in columns) + ' |',
             '|---|---|' + '---|' * len(columns)]
    for k in settings.KS:
        cells = [cell(rows, system=system, k=k, esc=esc,
                      vote_coef=f'{settings.VOTE[k]:g}' if vote == 'tuned' else vote,
                      vote_mode='hypothesis' if vote == 'tuned' else '-')
                 for system, vote, esc, _ in columns]
        lines.append(f'| {k} | {settings.VOTE[k]:g} | ' + ' | '.join(cells) + ' |')
    lines += ['', mode_table(rows)]
    singles = [r for r in rows if r['k'] == '-']
    lines += ['', 'Single systems (top-1): ' + ', '.join(
        f"{r['system']} {r['errant_f0_5']:.1f} ({r['errant_p']:.1f} / {r['errant_r']:.1f})" for r in singles)]
    return '\n'.join(lines) + '\n'


def mode_table(rows):
    """Markdown: GRECO with each vote mode at its tuned alpha, without and with ESC.

    At k = 1 the modes give the same output, so only "hypothesis" is run there.
    """
    modes = settings.VOTE_MODES
    head = [f'GRECO, {m}' for m in modes] + [f'GRECO + ESC, {m}' for m in modes]
    lines = ['Vote modes, each with its tuned vote_coef (in brackets in the header row of each k):', '',
             '| k | ' + ' | '.join(head) + ' |', '|---|' + '---|' * len(head)]
    for k in settings.KS:
        cells = []
        for esc in ('no', 'yes'):
            for mode in modes:
                if k == 1 and mode != 'hypothesis':
                    cells.append('= hypothesis')
                    continue
                vote = settings.VOTES[mode][k]
                cells.append(f'{cell(rows, system="greco", k=k, esc=esc, vote_mode=mode, vote_coef=f"{vote:g}")}'
                             f' [α {vote:g}]')
        lines.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(split=settings.TEST_SPLIT):
    from training.metrics import score  # needs ERRANT, so imported only when scoring
    src, gold = reference(split)
    n = len(read_lines(src))
    rows, missing = [], []
    for system, k, mode, vote, esc, rel in outputs():
        hyps = load(split, rel)
        if hyps is None:
            missing.append(rel)
            continue
        if len(hyps) != n:
            raise ValueError(f'{split}/{rel}: {len(hyps)} lines, expected {n}')
        inputs = 1 if k == '-' else len(settings.inputs(k))
        rows.append(row(system, k, mode, vote, esc, inputs, score(src, hyps, gold)))
        print(f"{system:<6} k={k!s:<2} mode={mode:<10} vote={vote:<4} esc={esc:<3} "
              f"ERRANT F0.5 {rows[-1]['errant_f0_5']:.2f}  M2 F0.5 {rows[-1]['m2_f0_5']:.2f}")
    out = settings.ROOT / split
    print(f'{split}: {n} sentences')
    with open(out / 'results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: f'{v:.2f}' if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    (out / 'results.md').write_text(table(rows), encoding='utf-8')
    print('\n' + table(rows))
    for rel in missing:
        print('missing:', rel)
    print(f'wrote {out / "results.csv"} and {out / "results.md"}')


if __name__ == '__main__':
    splits = (settings.FULL_SPLIT,) + settings.TEST_SPLITS
    if len(sys.argv) > 2 or sys.argv[1:] and sys.argv[1] not in splits:
        sys.exit(f'usage: python3 -m greco_esc.evaluate [{" | ".join(splits)}]')
    main(*sys.argv[1:])
