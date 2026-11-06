"""Verify a build of training/build_data.py against its sources; exits non-zero on any failure.

    python -m training.check_data [--data data/curriculum]

Checks line counts and hashes against manifest.json, that every stage-1 pair is
a pair of data/train.* and every stage-2 errorful pair a W&I pair of it, that
the dev split partitions data/dev.*, the length limit, and that both tokenizers
(T5 with restore_from_source) reproduce every dev and stage-2 sentence exactly.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

from training.build_data import TOKENIZER, TRAIN_CORPORA, normalize_quotes, sha256
from training.common import CURRICULUM, MAX_TOKENS, PREFIX, ROOT, Postprocessor, read_lines

failures = []


def check(ok, message):
    print(('ok    ' if ok else 'FAIL  ') + message, flush=True)
    if not ok:
        failures.append(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=CURRICULUM)
    data = parser.parse_args().data
    manifest = json.loads((data / 'manifest.json').read_text())
    read = {f'{name}.{ext}': read_lines(data / f'{name}.{ext}')
            for name in ('stage1', 'stage2', 'dev_tune', 'dev_test') for ext in ('src', 'tgt')}

    for name in ('stage1', 'stage2'):
        n = manifest[name]['pairs']
        check(all(len(read[f'{name}.{e}']) == n for e in ('src', 'tgt')) and
              len(read_lines(data / f'{name}.corpus')) == n, f'{name}: {n} lines in .src/.tgt/.corpus')
        check(all(sha256(data / f'{name}.{e}') == manifest[name]['sha256'][e] for e in ('src', 'tgt')),
              f'{name}: sha256 matches manifest')
        check(all(s.strip() and t.strip() for s, t in zip(read[f'{name}.src'], read[f'{name}.tgt'])),
              f'{name}: no empty lines')

    # Stage 1: every pair comes from data/train (after quote normalization if enabled); none unchanged
    train_src, train_tgt = read_lines(ROOT / 'data/train.src'), read_lines(ROOT / 'data/train.tgt')
    corpus = [c for c, n in TRAIN_CORPORA for _ in range(n)]
    normalize = manifest['options']['normalize_quotes']
    train = Counter()
    for s, t, c in zip(train_src, train_tgt, corpus):
        train[(normalize_quotes(s), normalize_quotes(t)) if normalize and c == 'lang8' else (s, t)] += 1
    stage1 = Counter(zip(read['stage1.src'], read['stage1.tgt']))
    check(not stage1 - train, 'stage1: every pair is a pair of data/train.*')
    check(all(s != t for s, t in stage1), 'stage1: errorful only')
    dropped = sum(v for k, v in manifest['stage1']['dropped'].items())
    check(manifest['stage1']['pairs'] + dropped == len(train_src),
          f'stage1: {manifest["stage1"]["pairs"]} kept + {dropped} dropped = {len(train_src)}')

    # Stage 2: errorful pairs are W&I pairs of data/train; the rest are unchanged sentences
    wi_start = sum(n for c, n in TRAIN_CORPORA if not c.startswith('wi.'))
    wi = Counter(zip(train_src[wi_start:], train_tgt[wi_start:]))
    stage2 = list(zip(read['stage2.src'], read['stage2.tgt']))
    errorful = Counter(p for p in stage2 if p[0] != p[1])
    check(not errorful - wi, f'stage2: all {sum(errorful.values())} errorful pairs are W&I pairs of data/train.*')
    check(sum(errorful.values()) + manifest['stage2']['unchanged_pairs'] == manifest['stage2']['pairs'],
          f'stage2: {manifest["stage2"]["unchanged_pairs"]} correct (unchanged) sentences')
    kinds = manifest['stage2']['m2_sentences']
    check(kinds['errorful'] == sum(wi.values()), f'stage2: M2 errorful count {kinds["errorful"]} = W&I lines of data/train')

    # Dev split
    dev_src, dev_tgt = read_lines(ROOT / 'data/dev.src'), read_lines(ROOT / 'data/dev.tgt')
    ids = {name: [int(i) for i in read_lines(data / f'{name}.ids')] for name in ('dev_tune', 'dev_test')}
    check(not set(ids['dev_tune']) & set(ids['dev_test']) and
          sorted(ids['dev_tune'] + ids['dev_test']) == list(range(len(dev_src))),
          f'dev split: {len(ids["dev_tune"])} + {len(ids["dev_test"])} partition data/dev ({len(dev_src)})')
    for name in ids:
        m2_sources = [line[2:] for line in (data / f'{name}.m2').read_text().splitlines() if line.startswith('S ')]
        check(read[f'{name}.src'] == [dev_src[i] for i in ids[name]] == m2_sources and
              read[f'{name}.tgt'] == [dev_tgt[i] for i in ids[name]], f'{name}: .src/.tgt/.m2 match data/dev at .ids')

    # Tokenization
    t5 = AutoTokenizer.from_pretrained(TOKENIZER)
    for name in ('stage1', 'stage2'):
        longest = max(max(len(t5(PREFIX['t5'] + s)['input_ids']) for s in read[f'{name}.src']),
                      max(len(t5(t)['input_ids']) for t in read[f'{name}.tgt']))
        check(longest <= MAX_TOKENS, f'{name}: longest sequence {longest} <= {MAX_TOKENS} T5 tokens')
    bart = AutoTokenizer.from_pretrained('facebook/bart-base')
    post = Postprocessor(t5, 't5')
    sentences = [s for name in ('dev_tune', 'dev_test', 'stage2') for e in ('src', 'tgt') for s in read[f'{name}.{e}']]
    bart_bad = [s for s in sentences if bart.decode(bart(s)['input_ids'], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False) != s]
    check(not bart_bad, f'BART reproduces all {len(sentences)} dev/stage2 sentences exactly')
    raw_bad = [s for s in sentences if t5.decode(t5(s)['input_ids'], skip_special_tokens=True) != s]
    t5_bad = [s for s in raw_bad if post(s, t5.decode(t5(s)['input_ids'], skip_special_tokens=True)) != s]
    check(not t5_bad, f'T5 + restore_from_source reproduces all of them '
                      f'({len(raw_bad)} would differ without restoring)')
    for s in t5_bad[:5]:
        print('      ', s)

    if failures:
        sys.exit(f'{len(failures)} check(s) failed')
    print('all checks passed')


if __name__ == '__main__':
    main()
