"""Turn the candidate pools into the files ESC and GRECO read.

    python3 -m greco_esc.prepare m2 <split>     # needs ERRANT (gec-train)
    python3 -m greco_esc.prepare esc            # the ESC folders for every k
    python3 -m greco_esc.prepare scores <k> [split]   # check and clip ESC's edit scores for GRECO
    python3 -m greco_esc.prepare final_esc <test set>...   # ESC on the full dev set, for the test sets

m2: reads greco_esc/runs/<split>/pool/{bart,t5}.tgt (5 lines per source, best first) and
gector.tgt (one per source). Writes one text file and one ERRANT M2 file per ranked output,
greco_esc/runs/<split>/{text,m2}/<name>.{txt,m2}: gector-1, bart-1..5, t5-1..5. GRECO and ESC
both read these M2 files, so ESC scores exactly the edits GRECO combines.

esc: ESC treats every file in its data folder as one system and learns a weight for each
(system, edit type). For each k and test split, <esc_home>/<split>/{text,m2} links the k inputs
GRECO combines (settings.inputs), plus the source, and for the training split the gold
corrections (target.txt, and the gold M2 as target.m2, which ESC labels the edits with).
esc_home is runs/esc/k<k> for dev_test (ESC trained on dev_tune) and runs/esc/k<k>/for-dev_tune
for dev_tune (ESC trained on dev_test), see settings.esc_home.

final_esc: for the test sets (test.sh), ESC is trained once per k on the full dev set. Its training folder,
runs/esc/final/k<k>/dev/{text,m2}, holds each input, the source, target.txt and target.m2 as
dev_tune followed by dev_test (written, not linked: the halves are concatenated). Each test set's
folder, runs/esc/final/k<k>/<test set>, links that set's inputs and source, without gold.

scores: ESC's --score output has one JSON list of [start, end, correction, probability] per
sentence. GRECO adds log(p) and log(1 - p) for each edit, so a probability of exactly 0 or 1
would stop it; those are clipped to [1e-6, 1 - 1e-6].
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from greco_esc import settings
from greco_setup.prepare import read_lines, read_system

EPS = 1e-6


def ranked(path, rank, n_sources):
    """The rank-th hypothesis (0 = best) of every source, from a k-lines-per-source file."""
    _, per_source = read_system(f'system={path}', n_sources)
    return [hyps[rank] for hyps in per_source]


def to_m2(src, hyps, out):
    """Write the ERRANT M2 edits from src to hyps, one hypothesis per source line."""
    tools = Path(sys.executable).parent
    with tempfile.TemporaryDirectory() as tmp:
        cor = Path(tmp, 'hyp.txt')
        cor.write_text('\n'.join(hyps) + '\n', encoding='utf-8')
        subprocess.run([str(tools / 'errant_parallel'), '-orig', str(src), '-cor', str(cor), '-out', str(out)],
                       check=True)


def m2_sources(path):
    blocks = Path(path).read_text(encoding='utf-8').strip().split('\n\n')
    return [b.split('\n')[0][2:] for b in blocks]


def build_m2(split):
    src = settings.source(split)
    sources = read_lines(src)
    pool = settings.ROOT / split / 'pool'
    text, m2 = settings.ROOT / split / 'text', settings.ROOT / split / 'm2'
    text.mkdir(parents=True, exist_ok=True)
    m2.mkdir(parents=True, exist_ok=True)
    outputs = {'gector-1': ranked(pool / 'gector.tgt', 0, len(sources))}
    for r in range(1, settings.MAX_RANK + 1):
        for system in settings.NBEST:
            outputs[f'{system}-{r}'] = ranked(pool / f'{system}.tgt', r - 1, len(sources))
    # Dev splits: ESC checks every M2 file against the gold one sentence by sentence; fail here
    # instead. Test sets have no usable gold here, so only the sentence count is checked.
    gold = None if split in settings.TEST_SETS else m2_sources(settings.DATA / f'{split}.m2')
    for name, hyps in outputs.items():
        (text / f'{name}.txt').write_text('\n'.join(hyps) + '\n', encoding='utf-8')
        to_m2(src, hyps, m2 / f'{name}.m2')
        written = m2_sources(m2 / f'{name}.m2')
        if (gold is not None and written != gold) or len(written) != len(sources):
            sys.exit(f'{m2 / name}.m2: source lines differ from {src}')
    print(f'wrote {len(outputs)} text and M2 files for {len(sources)} sentences to {settings.ROOT / split}')


def link(target, link_path):
    """A relative symlink, so the runs folder can be moved or copied as a whole."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def esc_dir(k, split, test_split=None):
    """ESC's data folder for split, in the folders used to score test_split (default dev_test)."""
    return settings.esc_home(k, test_split or settings.TEST_SPLIT) / split


def build_esc():
    for k in settings.KS:
        for test_split in settings.TEST_SPLITS:
            train_split = settings.ESC_TRAIN[test_split]
            for split in (train_split, test_split):
                out = esc_dir(k, split, test_split)
                for sub in ('text', 'm2'):
                    if (out / sub).exists():
                        for f in (out / sub).iterdir():
                            f.unlink()
                for name in settings.inputs(k):
                    link(settings.ROOT / split / 'text' / f'{name}.txt', out / 'text' / f'{name}.txt')
                    link(settings.ROOT / split / 'm2' / f'{name}.m2', out / 'm2' / f'{name}.m2')
                link(settings.DATA / f'{split}.src', out / 'text' / 'source.txt')
                if split == train_split:
                    link(settings.DATA / f'{split}.tgt', out / 'text' / 'target.txt')
                    link(settings.DATA / f'{split}.m2', out / 'm2' / 'target.m2')
            (settings.esc_home(k, test_split) / 'model').mkdir(parents=True, exist_ok=True)
    print(f'ESC folders for k = {", ".join(map(str, settings.KS))} and test splits '
          f'{", ".join(settings.TEST_SPLITS)} in {settings.ROOT / "esc"}')


def m2_text(path):
    """An M2 file's blocks as text ending in one blank line, so two files can be concatenated."""
    return '\n\n'.join(Path(path).read_text(encoding='utf-8').strip().split('\n\n')) + '\n\n'


def build_final_esc(test_sets):
    """ESC folders for the test sets, for every k: training on the full dev set (dev_tune + dev_test)."""
    for k in settings.KS:
        _build_final_esc(k, test_sets)


def _build_final_esc(k, test_sets):
    home = settings.esc_home(k, test_sets[0])
    dev = home / settings.FULL_SPLIT
    for sub in ('text', 'm2'):
        (dev / sub).mkdir(parents=True, exist_ok=True)
    halves = settings.HALVES
    for name in settings.inputs(k):
        (dev / 'text' / f'{name}.txt').write_text(
            ''.join((settings.ROOT / h / 'text' / f'{name}.txt').read_text(encoding='utf-8') for h in halves),
            encoding='utf-8')
        (dev / 'm2' / f'{name}.m2').write_text(
            ''.join(m2_text(settings.ROOT / h / 'm2' / f'{name}.m2') for h in halves), encoding='utf-8')
    for ext, name in (('src', 'source.txt'), ('tgt', 'target.txt')):
        (dev / 'text' / name).write_text(
            ''.join((settings.DATA / f'{h}.{ext}').read_text(encoding='utf-8') for h in halves), encoding='utf-8')
    (dev / 'm2' / 'target.m2').write_text(''.join(m2_text(settings.DATA / f'{h}.m2') for h in halves),
                                          encoding='utf-8')
    n = len(read_lines(dev / 'text' / 'source.txt'))
    for name in settings.inputs(k) + ['target']:
        if len(m2_sources(dev / 'm2' / f'{name}.m2')) != n:
            sys.exit(f'{dev / "m2" / name}.m2: not {n} sentences')
    for test_set in test_sets:
        out = home / test_set
        for name in settings.inputs(k):
            link(settings.ROOT / test_set / 'text' / f'{name}.txt', out / 'text' / f'{name}.txt')
            link(settings.ROOT / test_set / 'm2' / f'{name}.m2', out / 'm2' / f'{name}.m2')
        link(settings.source(test_set), out / 'text' / 'source.txt')
    (home / 'model').mkdir(parents=True, exist_ok=True)
    print(f'full-dev ESC folders for k = {k} ({n} training sentences) and {", ".join(test_sets)} in {home}')


def clip(raw_lines, n_sources):
    """Check ESC's per-sentence edit scores and clip each probability to [EPS, 1 - EPS]."""
    if len(raw_lines) != n_sources:
        raise ValueError(f'{len(raw_lines)} lines of ESC scores, expected {n_sources}')
    out = []
    for line in raw_lines:
        edits = json.loads(line) if line.strip() else []
        out.append([[s, e, cor, min(max(p, EPS), 1 - EPS)] for s, e, cor, p in edits])
    return out


def fix_scores(k, split=None):
    split = split or settings.TEST_SPLIT
    path = settings.ROOT / split / f'k{k}' / 'esc.scores.raw.jsonl'
    n = len(read_lines(settings.source(split)))
    # ESC writes one JSON list per sentence ('[]' when it has no edits), joined by '\n'
    raw = path.read_text(encoding='utf-8').rstrip('\n').split('\n')
    scores = clip(raw, n)
    out = path.with_name('esc.scores.jsonl')
    out.write_text('\n'.join(json.dumps(s) for s in scores) + '\n', encoding='utf-8')
    print(f'k{k}: {sum(len(s) for s in scores)} edit scores for {n} sentences in {out}')


def main(argv):
    if argv[:1] == ['m2'] and len(argv) == 2:
        build_m2(argv[1])
    elif argv == ['esc']:
        build_esc()
    elif argv[:1] == ['scores'] and len(argv) in (2, 3):
        fix_scores(int(argv[1]), *argv[2:])
    elif argv[:1] == ['final_esc'] and len(argv) > 1 and set(argv[1:]) <= set(settings.TEST_SETS):
        build_final_esc(argv[1:])
    else:
        sys.exit('usage: python3 -m greco_esc.prepare m2 <split> | esc | scores <k> [split] '
                 '| final_esc <test set>...')


if __name__ == '__main__':
    main(sys.argv[1:])
