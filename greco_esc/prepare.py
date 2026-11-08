"""Turn the candidate pools into the files ESC and GRECO read.

    python3 -m greco_esc.prepare m2 <split>     # needs ERRANT (gec-train)
    python3 -m greco_esc.prepare esc            # the ESC folders for every k
    python3 -m greco_esc.prepare scores <k>     # check and clip ESC's edit scores for GRECO

m2: reads greco_esc/runs/<split>/pool/{bart,t5}.tgt (5 lines per source, best first) and
gector.tgt (one per source). Writes one text file and one ERRANT M2 file per ranked output,
greco_esc/runs/<split>/{text,m2}/<name>.{txt,m2}: gector-1, bart-1..5, t5-1..5. GRECO and ESC
both read these M2 files, so ESC scores exactly the edits GRECO combines.

esc: ESC treats every file in its data folder as one system and learns a weight for each
(system, edit type). For each k, greco_esc/runs/esc/k<k>/<split>/{text,m2} links the k inputs
GRECO combines (settings.inputs), plus the source, and for the training split the gold
corrections (target.txt, and the gold M2 as target.m2, which ESC labels the edits with).

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
    src = settings.DATA / f'{split}.src'
    sources = read_lines(src)
    pool = settings.ROOT / split / 'pool'
    text, m2 = settings.ROOT / split / 'text', settings.ROOT / split / 'm2'
    text.mkdir(parents=True, exist_ok=True)
    m2.mkdir(parents=True, exist_ok=True)
    outputs = {'gector-1': ranked(pool / 'gector.tgt', 0, len(sources))}
    for r in range(1, settings.MAX_RANK + 1):
        for system in settings.NBEST:
            outputs[f'{system}-{r}'] = ranked(pool / f'{system}.tgt', r - 1, len(sources))
    gold = m2_sources(settings.DATA / f'{split}.m2')
    for name, hyps in outputs.items():
        (text / f'{name}.txt').write_text('\n'.join(hyps) + '\n', encoding='utf-8')
        to_m2(src, hyps, m2 / f'{name}.m2')
        # ESC checks every M2 file against the gold one sentence by sentence; fail here instead
        if m2_sources(m2 / f'{name}.m2') != gold:
            sys.exit(f'{m2 / name}.m2: source lines differ from {split}.m2')
    print(f'wrote {len(outputs)} text and M2 files for {len(sources)} sentences to {settings.ROOT / split}')


def link(target, link_path):
    """A relative symlink, so the runs folder can be moved or copied as a whole."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def esc_dir(k, split):
    return settings.ROOT / 'esc' / f'k{k}' / split


def build_esc():
    for k in settings.KS:
        for split in (settings.TRAIN_SPLIT, settings.TEST_SPLIT):
            out = esc_dir(k, split)
            for sub in ('text', 'm2'):
                if (out / sub).exists():
                    for f in (out / sub).iterdir():
                        f.unlink()
            for name in settings.inputs(k):
                link(settings.ROOT / split / 'text' / f'{name}.txt', out / 'text' / f'{name}.txt')
                link(settings.ROOT / split / 'm2' / f'{name}.m2', out / 'm2' / f'{name}.m2')
            link(settings.DATA / f'{split}.src', out / 'text' / 'source.txt')
            if split == settings.TRAIN_SPLIT:
                link(settings.DATA / f'{split}.tgt', out / 'text' / 'target.txt')
                link(settings.DATA / f'{split}.m2', out / 'm2' / 'target.m2')
        (settings.ROOT / 'esc' / f'k{k}' / 'model').mkdir(parents=True, exist_ok=True)
    print(f'ESC folders for k = {", ".join(map(str, settings.KS))} in {settings.ROOT / "esc"}')


def clip(raw_lines, n_sources):
    """Check ESC's per-sentence edit scores and clip each probability to [EPS, 1 - EPS]."""
    if len(raw_lines) != n_sources:
        raise ValueError(f'{len(raw_lines)} lines of ESC scores, expected {n_sources}')
    out = []
    for line in raw_lines:
        edits = json.loads(line) if line.strip() else []
        out.append([[s, e, cor, min(max(p, EPS), 1 - EPS)] for s, e, cor, p in edits])
    return out


def fix_scores(k):
    path = settings.ROOT / settings.TEST_SPLIT / f'k{k}' / 'esc.scores.raw.jsonl'
    n = len(read_lines(settings.DATA / f'{settings.TEST_SPLIT}.src'))
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
    elif argv[:1] == ['scores'] and len(argv) == 2:
        fix_scores(int(argv[1]))
    else:
        sys.exit('usage: python3 -m greco_esc.prepare m2 <split> | esc | scores <k>')


if __name__ == '__main__':
    main(sys.argv[1:])
