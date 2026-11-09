"""Decode a split with a fine-tuned model and score it with ERRANT (primary) and M2.

    python -m training.evaluate --model ${CKPT_ROOT}/bart-base.s1/best --split dev_test \
        --out experiments/training/bart-base.s1

Splits: dev_tune, dev_test (data/curriculum/, with gold M2) and bea19_test,
conll14_test (data/test.*; bea19 has no public gold, so only the predictions are
written, for Codabench). Writes <out>/<split>.pred and <out>/<split>.scores.json.

Reference checkpoints (not trained by us) also work:
    --model gotutiyan/gec-bart-base
    --model vennify/t5-base-grammar-correction --prefix "grammar: " --retokenize
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer

from training.common import (CURRICULUM, ROOT, Postprocessor, decode, expand, family, git_commit, pick_device,
                             read_lines, write_lines)
from training.metrics import m2_subset, score

sys.path.insert(0, str(ROOT / 'bart'))
from generate_nbest import load_model  # noqa: E402  (restores gotutiyan's tied embedding)

SPLITS = {
    'dev_tune': (CURRICULUM / 'dev_tune.src', CURRICULUM / 'dev_tune.m2'),
    'dev_test': (CURRICULUM / 'dev_test.src', CURRICULUM / 'dev_test.m2'),
    'conll14_test': (ROOT / 'data/test.conll14.src', ROOT / 'data/test.conll14.m2'),
    'bea19_test': (ROOT / 'data/test.bea19.src', None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='checkpoint directory or Hugging Face name')
    parser.add_argument('--split', choices=SPLITS, nargs='+', default=['dev_tune', 'dev_test'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--beams', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, help='only the first N sentences (smoke tests)')
    parser.add_argument('--prefix', help="override the model family's source prefix")
    parser.add_argument('--retokenize', action='store_true',
                        help='spaCy-tokenize outputs like the corpora (models trained on untokenized text)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    args = parser.parse_args()
    name = expand(args.model)
    device = pick_device() if args.device == 'auto' else torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = load_model(name).to(device).eval()
    post = Postprocessor(tokenizer, family(model.config))
    if args.retokenize:
        sys.path.insert(0, str(ROOT / 't5'))
        from retokenize import retokenize
        base = post
        post = lambda source, text: base(source, retokenize(text))  # noqa: E731
    args.out.mkdir(parents=True, exist_ok=True)

    for split in args.split:
        src_path, gold = SPLITS[split]
        sources = read_lines(src_path)[:args.limit]
        started = time.time()
        hyps = decode(model, tokenizer, sources, num_beams=args.beams, batch_size=args.batch_size,
                      prefix=args.prefix, post=post)
        seconds = round(time.time() - started)
        write_lines(args.out / f'{split}.pred', hyps)
        report = {'model': name, 'split': split, 'sentences': len(sources), 'beams': args.beams,
                  'prefix': args.prefix, 'retokenize': args.retokenize, 'git_commit': git_commit(),
                  'decode_seconds': seconds, 'device': device.type}
        if gold and gold.read_text(encoding='utf-8')[:40].startswith('version https://git-lfs'):
            # data/test.conll14.m2 is in Git LFS; without git-lfs (the cluster) it is a pointer file
            print(f'{split:<12} {gold} is a Git LFS pointer: predictions only, score where git-lfs '
                  'has pulled it (greco_setup/test_report.py)', flush=True)
            gold = None
        if gold:
            if args.limit:
                src_path = args.out / f'{split}.{args.limit}.src'
                write_lines(src_path, sources)
                gold = m2_subset(gold, args.limit, args.out / f'{split}.{args.limit}.m2')
            report.update(score(src_path, hyps, gold))
            e, m = report['errant']['correction'], report['m2']
            print(f"{split:<12} ERRANT P {e['precision'] * 100:5.1f} R {e['recall'] * 100:5.1f} "
                  f"F0.5 {e['f0_5'] * 100:5.1f}   M2 P {m['precision'] * 100:5.1f} R {m['recall'] * 100:5.1f} "
                  f"F0.5 {m['f0_5'] * 100:5.1f}   ({seconds}s)", flush=True)
        else:
            print(f'{split:<12} predictions only (no public gold) ({seconds}s)', flush=True)
        (args.out / f'{split}.scores.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
