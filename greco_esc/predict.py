"""Helpers for predict.sh: run one GRECO (+ ESC) combination on one split or file.

predict.sh calls these between its steps (each in the environment it needs):

    python3 -m greco_esc.predict preset <name>               # k vote_mode vote_coef esc
    python3 -m greco_esc.predict inputs <split> <out> [limit] # writes <out>/source.txt (+ gold.m2)
    python3 -m greco_esc.predict m2 <out> <k>                 # needs ERRANT (gec-train)
    python3 -m greco_esc.predict esc_model <split> <k>        # the committed ESC model to use
    python3 -m greco_esc.predict clip <raw> <out> <n>
    python3 -m greco_esc.predict score <out>                  # needs ERRANT (gec-train)

Splits are the ones training.evaluate uses: dev_tune and dev_test (data/curriculum/, with gold),
conll14_test (gold M2 in Git LFS) and bea19_test (no public gold; submit to Codabench). A path to
any tokenized file of source sentences also works, without scoring.

ESC model: each split is scored with an ESC that never saw its gold edits, as in the reported
results. dev_test uses esc_weights/k<k> (trained on dev_tune), dev_tune uses
esc_weights/k<k>/for-dev_tune (trained on dev_test), and every other input uses
esc_weights/final/k<k> (trained on the full dev set).
"""

import json
import os
import sys
from pathlib import Path

from greco_esc import prepare, settings
from greco_setup.prepare import read_lines

SPLITS = {
    'dev_tune': (settings.DATA / 'dev_tune.src', settings.DATA / 'dev_tune.m2'),
    'dev_test': (settings.DATA / 'dev_test.src', settings.DATA / 'dev_test.m2'),
    'conll14_test': settings.TEST_SETS['conll14'],
    'bea19_test': settings.TEST_SETS['bea19'],
}


def source_and_gold(split):
    if split in SPLITS:
        return SPLITS[split]
    if not Path(split).is_file():
        sys.exit(f'{split}: not a split ({", ".join(SPLITS)}) or a file')
    return Path(split), None


def write_inputs(split, out, limit=None):
    """<out>/source.txt and, where the split has public gold, <out>/gold.m2 (first `limit` sentences)."""
    src, gold = source_and_gold(split)
    out.mkdir(parents=True, exist_ok=True)
    lines = read_lines(src)[:limit]
    (out / 'source.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if gold is not None and Path(gold).is_file() and not is_lfs_pointer(gold):
        blocks = Path(gold).read_text(encoding='utf-8').strip().split('\n\n')
        (out / 'gold.m2').write_text('\n\n'.join(blocks[:len(lines)]) + '\n\n', encoding='utf-8')
    elif gold is not None:
        print(f'{gold} is missing or a Git LFS pointer (git lfs pull); predictions will not be scored', file=sys.stderr)
    print(len(lines))


def is_lfs_pointer(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read(40).startswith('version https://git-lfs')


def build_m2(out, k):
    """<out>/text/<name>.txt and <out>/m2/<name>.m2 for the 1 + 2k inputs, from <out>/pool/*.tgt."""
    src = out / 'source.txt'
    n = len(read_lines(src))
    for sub in ('text', 'm2'):
        (out / sub).mkdir(exist_ok=True)
    for name in settings.inputs(k):
        system, rank = name.rsplit('-', 1)
        hyps = prepare.ranked(out / 'pool' / f'{system}.tgt', int(rank) - 1, n)
        (out / 'text' / f'{name}.txt').write_text('\n'.join(hyps) + '\n', encoding='utf-8')
        prepare.to_m2(src, hyps, out / 'm2' / f'{name}.m2')
        if len(prepare.m2_sources(out / 'm2' / f'{name}.m2')) != n:
            sys.exit(f'{out / "m2" / name}.m2: not {n} sentences')
    # ESC reads every file in its data folder as one system, plus source.txt
    esc = out / 'esc'
    for name in settings.inputs(k):
        prepare.link(out / 'text' / f'{name}.txt', esc / 'text' / f'{name}.txt')
        prepare.link(out / 'm2' / f'{name}.m2', esc / 'm2' / f'{name}.m2')
    prepare.link(src, esc / 'text' / 'source.txt')
    print(f'{len(settings.inputs(k))} inputs for k = {k}: {" ".join(settings.inputs(k))}')


def esc_model(split, k):
    rel = {'dev_test': f'k{k}', 'dev_tune': f'k{k}/for-dev_tune'}.get(split, f'final/k{k}')
    return prepare.WEIGHTS / rel


def clip(raw, out, n):
    lines = Path(raw).read_text(encoding='utf-8').rstrip('\n').split('\n')
    scores = prepare.clip(lines, n)
    Path(out).write_text('\n'.join(json.dumps(s) for s in scores) + '\n', encoding='utf-8')


def score(out):
    from training.metrics import score as metric  # needs ERRANT
    hyps = read_lines(out / 'pred.txt')
    result = metric(out / 'source.txt', hyps, out / 'gold.m2')
    e, m = result['errant']['correction'], result['m2']
    summary = {'errant': {key: e[key] for key in ('precision', 'recall', 'f0_5')}, 'm2': m}
    (out / 'scores.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"ERRANT P {e['precision'] * 100:.2f} R {e['recall'] * 100:.2f} F0.5 {e['f0_5'] * 100:.2f} | "
          f"M2 P {m['precision'] * 100:.2f} R {m['recall'] * 100:.2f} F0.5 {m['f0_5'] * 100:.2f}")


def main(argv):
    cmd, args = (argv[0], argv[1:]) if argv else ('', [])
    if cmd == 'preset' and len(args) == 1 and args[0] in settings.PRESETS:
        name, k = settings.PRESETS[args[0]]
        mode, vote, esc = next((m, v, e) for n, kk, v, e, m in settings.runs() if n == name and kk == k)
        print(k, mode, f'{vote:g}', int(esc))
    elif cmd == 'inputs' and len(args) in (2, 3):
        write_inputs(args[0], Path(args[1]), int(args[2]) if len(args) == 3 and args[2] else None)
    elif cmd == 'm2' and len(args) == 2:
        build_m2(Path(args[0]), int(args[1]))
    elif cmd == 'esc_model' and len(args) == 2:
        print(esc_model(args[0], int(args[1])))
    elif cmd == 'clip' and len(args) == 3:
        clip(args[0], args[1], int(args[2]))
    elif cmd == 'score' and len(args) == 1:
        score(Path(args[0]))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    main(sys.argv[1:])
