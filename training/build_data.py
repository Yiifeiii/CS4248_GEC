"""Build the two-stage training data and the dev split in data/curriculum/.

    python -m training.build_data        # from the repo root, with transformers installed

Stage 1 (stage1.*): data/train.src/.tgt as released, the errorful sentences of
    FCE train+dev, Lang-8, NUCLE and W&I A/B/C.
Stage 2 (stage2.*): W&I A/B/C train including its correct sentences, from the
    original M2 files at commit SOURCE_REV, converted with the provided
    m2_to_parallel.py (unchanged).
Dev split (dev_tune.*, dev_test.*): a seeded random split of data/dev.* by sentence.

Default cleaning is the minimum training needs:
- pairs over MAX_TOKENS tokens are dropped (truncation would cut source and target
  at different points);
- empty sources or targets are dropped (whole-sentence deletions);
- stage 2 drops sentences whose only edit is UNK: an error was marked but no
  correction given, so the "correct" target still contains the error.
Opt-in cleaning for ablations (write those builds to another --out directory):
--normalize-quotes, --ratio-filter, --dedupe.

Every count, filter and hash is written to manifest.json; training/check_data.py verifies it.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from transformers import AutoTokenizer

from training.common import CURRICULUM, MAX_TOKENS, PREFIX, ROOT, git_commit, read_lines, write_lines

sys.path.insert(0, str(ROOT))
from m2_to_parallel import m2_to_parallel  # noqa: E402  (provided script, used unchanged)

# data/train.* is these files from commit SOURCE_REV joined in this order (checked by sha256)
SOURCE_REV = '3e50997'
TRAIN_CORPORA = [
    ('fce.train', 17715), ('fce.dev', 1370), ('lang8', 498360), ('nucle', 21353),
    ('wi.A', 8194), ('wi.B', 9138), ('wi.C', 5405),
]
WI_TRAIN_M2 = {
    'wi.A': 'wi+locness/m2/A.train.gold.bea19.m2',
    'wi.B': 'wi+locness/m2/B.train.gold.bea19.m2',
    'wi.C': 'wi+locness/m2/C.train.gold.bea19.m2',
}
TOKENIZER = 'google-t5/t5-base'
SEED = 4248
DEV_TUNE = 2000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_file(rev, path):
    """Contents of a file at a revision, resolving Git LFS pointers from the local LFS store."""
    blob = subprocess.run(['git', 'show', f'{rev}:{path}'], cwd=ROOT, check=True, capture_output=True).stdout
    if blob.startswith(b'version https://git-lfs'):
        blob = subprocess.run(['git', 'lfs', 'smudge'], cwd=ROOT, input=blob, check=True,
                              capture_output=True).stdout
        if blob.startswith(b'version https://git-lfs'):
            raise SystemExit(f'{path}: LFS object missing; run `git lfs fetch origin {rev}`')
    return blob.decode('utf-8')


def normalize_quotes(sentence):
    # Lang-8 writes double quotes PTB-style as `` and '' (0 lines have "); dev and test use "
    return ' '.join('"' if token in ('``', "''") else token for token in sentence.split())


class Cleaner:
    """The filters applied to every training pair, with counts of what each drops."""

    def __init__(self, tokenizer, ratio_filter=False, dedupe=False):
        self.tokenizer = tokenizer
        self.ratio_filter = ratio_filter
        self.dedupe = dedupe
        self.dropped = Counter()

    def too_long(self, src, tgt):
        return (len(self.tokenizer(PREFIX['t5'] + src)['input_ids']) > MAX_TOKENS
                or len(self.tokenizer(tgt)['input_ids']) > MAX_TOKENS)

    def reason(self, src, tgt):
        n, m = len(src.split()), len(tgt.split())
        if not n or not m:
            return 'empty'
        if self.ratio_filter and (m > 2 * n + 2 or m < 0.5 * n - 2):
            return 'length_ratio'
        if self.too_long(src, tgt):
            return f'over_{MAX_TOKENS}_tokens'
        return None

    def clean(self, pairs):
        """pairs: [(src, tgt, corpus)] -> the pairs that pass every filter."""
        self.dropped = Counter()
        kept, seen = [], set()
        for src, tgt, corpus in pairs:
            reason = self.reason(src, tgt)
            if reason is None and self.dedupe and (src, tgt) in seen:
                reason = 'duplicate'
            if reason:
                self.dropped[reason] += 1
                continue
            seen.add((src, tgt))
            kept.append((src, tgt, corpus))
        return kept


def write_stage(out, name, pairs, manifest):
    random.Random(SEED).shuffle(pairs)
    for i, ext in enumerate(('src', 'tgt', 'corpus')):
        write_lines(out / f'{name}.{ext}', [p[i] for p in pairs])
    manifest[name] = {
        'pairs': len(pairs),
        'unchanged_pairs': sum(src == tgt for src, tgt, _ in pairs),
        'by_corpus': dict(sorted(Counter(c for _, _, c in pairs).items())),
        'sha256': {ext: sha256(out / f'{name}.{ext}') for ext in ('src', 'tgt')},
    }


def build_stage1(out, cleaner, normalize, manifest):
    src, tgt = read_lines(ROOT / 'data/train.src'), read_lines(ROOT / 'data/train.tgt')
    assert len(src) == len(tgt) == sum(n for _, n in TRAIN_CORPORA)
    corpus = [name for name, n in TRAIN_CORPORA for _ in range(n)]
    pairs, quotes, unchanged = [], 0, 0
    for s, t, c in zip(src, tgt, corpus):
        if normalize and c == 'lang8':
            s2, t2 = normalize_quotes(s), normalize_quotes(t)
            quotes += (s2, t2) != (s, t)
            s, t = s2, t2
            if s == t:
                # Stage 1 is errorful only; a quote-only edit disappears after normalization
                unchanged += 1
                continue
        pairs.append((s, t, c))
    pairs = cleaner.clean(pairs)
    dropped = dict(cleaner.dropped)
    if normalize:
        dropped['unchanged_after_quote_normalization'] = unchanged
    write_stage(out, 'stage1', pairs, manifest)
    manifest['stage1'].update(
        source={'files': ['data/train.src', 'data/train.tgt'],
                'sha256': {'src': sha256(ROOT / 'data/train.src'), 'tgt': sha256(ROOT / 'data/train.tgt')},
                'lines': len(src), 'corpora_in_order': dict(TRAIN_CORPORA)},
        dropped=dropped)
    if normalize:
        manifest['stage1']['lang8_quote_normalized_pairs'] = quotes


def wi_entries(name):
    """Convert one W&I train M2 with the provided m2_to_parallel (all sentences, annotator 0).

    Returns [(src, tgt, kind)], kind in errorful / correct / unk_only, one per M2 entry.
    """
    text = git_file(SOURCE_REV, WI_TRAIN_M2[name])
    with tempfile.TemporaryDirectory() as tmp:
        m2 = Path(tmp, 'in.m2')
        m2.write_text(text, encoding='utf-8')
        m2_to_parallel([str(m2)], str(m2.with_suffix('.src')), str(m2.with_suffix('.tgt')), False, False)
        src, tgt = read_lines(m2.with_suffix('.src')), read_lines(m2.with_suffix('.tgt'))
    entries = text.strip().split('\n\n')
    # Without --erroneous_only the converter writes exactly one line per entry
    assert len(entries) == len(src) == len(tgt), name
    out = []
    for entry, s, t in zip(entries, src, tgt):
        assert entry.split('\n')[0][2:] == s
        types = [line.split('|||')[1] for line in entry.split('\n')[1:] if line.split('|||')[-1] == '0']
        if s != t:
            kind = 'errorful'
        elif 'UNK' in types:
            kind = 'unk_only'
        else:
            kind = 'correct'
        out.append((s, t, kind))
    return out


def build_stage2(out, cleaner, manifest):
    pairs, kinds = [], Counter()
    for name in WI_TRAIN_M2:
        for s, t, kind in wi_entries(name):
            kinds[kind] += 1
            if kind != 'unk_only':
                pairs.append((s, t, name))
    pairs = cleaner.clean(pairs)
    write_stage(out, 'stage2', pairs, manifest)
    manifest['stage2'].update(
        source={'rev': SOURCE_REV, 'm2': WI_TRAIN_M2, 'converter': 'm2_to_parallel.py (annotator 0, all sentences)'},
        m2_sentences=dict(kinds), dropped={'unk_only': kinds['unk_only'], **cleaner.dropped})


def build_dev_split(out, manifest):
    src, tgt = read_lines(ROOT / 'data/dev.src'), read_lines(ROOT / 'data/dev.tgt')
    blocks = (ROOT / 'data/dev.m2').read_text(encoding='utf-8').strip().split('\n\n')
    assert len(src) == len(tgt) == len(blocks)
    assert all(b.split('\n')[0][2:] == s for b, s in zip(blocks, src))
    tune = set(random.Random(SEED).sample(range(len(src)), DEV_TUNE))
    for name, ids in (('dev_tune', sorted(tune)), ('dev_test', [i for i in range(len(src)) if i not in tune])):
        write_lines(out / f'{name}.src', [src[i] for i in ids])
        write_lines(out / f'{name}.tgt', [tgt[i] for i in ids])
        write_lines(out / f'{name}.ids', [str(i) for i in ids])
        (out / f'{name}.m2').write_text('\n\n'.join(blocks[i] for i in ids) + '\n\n', encoding='utf-8')
        manifest[name] = {'sentences': len(ids), 'unchanged': sum(src[i] == tgt[i] for i in ids),
                          'ids': f'{name}.ids (0-based lines of data/dev.src)',
                          'sha256': {ext: sha256(out / f'{name}.{ext}') for ext in ('src', 'tgt', 'm2')}}


def overlap(out, manifest):
    """Training sources that also occur verbatim in dev/test (boilerplate and W&I prompts when checked)."""
    report = {}
    for stage in ('stage1', 'stage2'):
        train = set(read_lines(out / f'{stage}.src'))
        for split in ('dev.src', 'test.bea19.src', 'test.conll14.src'):
            shared = train & set(read_lines(ROOT / 'data' / split))
            report[f'{stage} vs {split}'] = {
                'sentences': len(shared), 'max_words': max((len(s.split()) for s in shared), default=0)}
    manifest['overlap_with_eval_sources'] = report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=CURRICULUM)
    parser.add_argument('--normalize-quotes', action='store_true',
                        help='Lang-8 only: map `` and \'\' to " on both sides')
    parser.add_argument('--ratio-filter', action='store_true',
                        help='drop pairs whose target has > 2*n+2 or < 0.5*n-2 words for an n-word source')
    parser.add_argument('--dedupe', action='store_true', help='drop exact (source, target) repeats within a stage')
    args = parser.parse_args()
    out = args.out.resolve()
    if out != CURRICULUM and not (args.normalize_quotes or args.ratio_filter or args.dedupe):
        print(f'note: default cleaning written to {out}, not {CURRICULUM}', flush=True)
    if out == CURRICULUM and (args.normalize_quotes or args.ratio_filter or args.dedupe):
        raise SystemExit(f'opt-in cleaning would overwrite the default data in {CURRICULUM}; pass --out')
    out.mkdir(parents=True, exist_ok=True)
    cleaner = Cleaner(AutoTokenizer.from_pretrained(TOKENIZER), args.ratio_filter, args.dedupe)
    manifest = {
        'built_by': 'training/build_data.py', 'git_commit': git_commit(), 'seed': SEED,
        'options': {'normalize_quotes': args.normalize_quotes, 'ratio_filter': args.ratio_filter,
                    'dedupe': args.dedupe},
        'filters': {
            'empty': 'empty source or target',
            f'over_{MAX_TOKENS}_tokens': f'{TOKENIZER} tokens of "{PREFIX["t5"]}"+source or of target > '
                                         f'{MAX_TOKENS} (same pairs dropped for BART)',
            'unk_only': 'stage 2: sentence whose only annotator-0 edit is UNK',
            **({'lang8_quotes': '`` and \'\' -> " (Lang-8 only, both sides)'} if args.normalize_quotes else {}),
            **({'length_ratio': 'target words > 2*source+2 or < 0.5*source-2'} if args.ratio_filter else {}),
            **({'duplicate': 'exact (source, target) repeat within a stage'} if args.dedupe else {}),
        },
    }
    build_stage1(out, cleaner, args.normalize_quotes, manifest)
    print('stage1', manifest['stage1']['pairs'], 'pairs, dropped', manifest['stage1']['dropped'], flush=True)
    build_stage2(out, cleaner, manifest)
    print('stage2', manifest['stage2']['pairs'], 'pairs, dropped', manifest['stage2']['dropped'], flush=True)
    build_dev_split(out, manifest)
    overlap(out, manifest)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {out}/', flush=True)


if __name__ == '__main__':
    main()
