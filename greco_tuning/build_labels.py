"""Build GRECO training records from n-best candidates, labelled with ERRANT.

Each source sentence gets a group of n hypotheses (n = 4, as in the GRECO paper). The
group is one JSON line holding one record per hypothesis, in the layout that the
upstream loader (utils/data.py, TrainDataset) reads:

  source, hyp   word lists
  labels        [[F0.5 of the hypothesis], [source word labels], [hypothesis word labels]]
  gap           one label per hypothesis word, plus one for the end of the sentence
  masks         [source edit mask, hypothesis edit mask]
  gap_mask      edit mask for gap

Labels: 1 means keep (correct), 0 means change. A gap label of 0 means a word must be
inserted there. The edit masks mark the positions that have an edit.

Candidates come from the k-lines-per-source files that bart/export_candidates.py writes.
They are merged by rank, so the best candidate of every system comes first. Candidates
equal to the source (no edit) or to the gold target (already perfect) are dropped, as in
the paper. Duplicates are dropped too. A source with fewer than n usable candidates is skipped.

    python3 -m greco_tuning.build_labels \\
      --source data/greco/stage1.sample.src --gold data/greco/stage1.sample.tgt \\
      --system bart=outputs/greco-s1/bart.tgt --system t5=outputs/greco-s1/t5.tgt \\
      --k 5 --out data/greco/stage1.train.jsonl

Needs ERRANT and en_core_web_sm in an environment other than the GRECO one (see README).
"""

import argparse
import json
from pathlib import Path

from greco_setup.prepare import read_lines, read_system

GROUP_SIZE = 4
MAX_LENGTH = 200  # the upstream loader pads and truncates to this many subword tokens


def f05(hyp_edits, gold_edits):
    """Sentence F0.5 of a hypothesis: its edits matched against the gold edits.

    An empty side counts as precision 1 or recall 1, so a perfect hypothesis scores 1.
    """
    gold_keys = [edit_key(e) for e in gold_edits]
    hyp_keys = [edit_key(e) for e in hyp_edits]
    tp = sum(key in gold_keys for key in hyp_keys)
    p = tp / len(hyp_keys) if hyp_keys else 1.0
    r = tp / len(gold_keys) if gold_keys else 1.0
    if p + r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def edit_key(edit):
    return (edit.o_start, edit.o_end, edit.c_str)


def token_labels(gold_edits, wrong_edits, n_src, n_hyp):
    """Word and gap labels for one hypothesis, plus the edit masks.

    Source labels come from the gold edits, since the gold correction says which source
    words are wrong. Hypothesis labels come from the edits that turn the hypothesis into
    the gold (wrong_edits), so a hypothesis word is 0 when it differs from the gold there.
    """
    src_w, src_m = [1] * n_src, [False] * n_src
    for edit in gold_edits:
        for i in range(edit.o_start, edit.o_end):
            src_w[i], src_m[i] = 0, True

    hyp_w, hyp_m = [1] * n_hyp, [False] * n_hyp
    gap, gap_m = [1] * (n_hyp + 1), [False] * (n_hyp + 1)
    for edit in wrong_edits:
        for j in range(edit.o_start, edit.o_end):
            hyp_w[j], hyp_m[j] = 0, True
        if edit.o_start == edit.o_end:
            # A missing word is an insertion at this position in the hypothesis
            gap[edit.o_start], gap_m[edit.o_start] = 0, True
    return src_w, src_m, hyp_w, hyp_m, gap, gap_m


def hyp_record(annotator, src_doc, gold_doc, hyp_doc, src_words, hyp_words, gold_edits):
    hyp_edits = annotator.annotate(src_doc, hyp_doc)
    wrong_edits = annotator.annotate(hyp_doc, gold_doc)
    src_w, src_m, hyp_w, hyp_m, gap, gap_m = token_labels(
        gold_edits, wrong_edits, len(src_words), len(hyp_words))
    return {
        'source': src_words,
        'hyp': hyp_words,
        'labels': [[f05(hyp_edits, gold_edits)], src_w, hyp_w],
        'gap': gap,
        'masks': [src_m, hyp_m],
        'gap_mask': gap_m,
    }


def merged_candidates(systems, index, k):
    """Candidates for one source in rank order, alternating between the systems."""
    for rank in range(k):
        for _, per_source in systems:
            hyps = per_source[index]
            if rank < len(hyps):
                yield hyps[rank]


class Builder:
    def __init__(self, annotator, tokenizer=None, group_size=GROUP_SIZE, max_length=MAX_LENGTH):
        self.annotator = annotator
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_length = max_length
        self.stats = {'sources': 0, 'groups': 0, 'skipped_few_candidates': 0,
                      'skipped_parse_mismatch': 0}

    def parse(self, text):
        doc = self.annotator.parse(text, tokenise=False)
        return doc if [t.text for t in doc] == text.split() else None

    def fits(self, src_words, hyp_words):
        if self.tokenizer is None:
            return True
        enc = self.tokenizer(src_words, hyp_words, is_split_into_words=True, truncation=False)
        return len(enc['input_ids']) <= self.max_length

    def group(self, source, gold, systems, index, k):
        """One JSON line's worth of records for a source, or None if it can't be used."""
        self.stats['sources'] += 1
        src_words = source.split()
        src_doc, gold_doc = self.parse(source), self.parse(gold)
        if src_doc is None or gold_doc is None:
            self.stats['skipped_parse_mismatch'] += 1
            return None

        gold_edits = self.annotator.annotate(src_doc, gold_doc)
        records, seen = [], set()
        for cand in merged_candidates(systems, index, k):
            hyp_words = cand.split()
            key = ' '.join(hyp_words)
            if not hyp_words or key in seen or hyp_words == src_words or hyp_words == gold.split():
                continue
            seen.add(key)
            hyp_doc = self.parse(key)
            if hyp_doc is None or not self.fits(src_words, hyp_words):
                continue
            records.append(hyp_record(self.annotator, src_doc, gold_doc, hyp_doc,
                                      src_words, hyp_words, gold_edits))
            if len(records) == self.group_size:
                self.stats['groups'] += 1
                return records
        self.stats['skipped_few_candidates'] += 1
        return None


def build(source_path, gold_path, system_specs, k, out_path, annotator, tokenizer=None,
          max_length=MAX_LENGTH):
    sources = read_lines(Path(source_path))
    golds = read_lines(Path(gold_path))
    if len(sources) != len(golds):
        raise ValueError('source and gold files have different line counts')
    systems = [read_system(spec, len(sources)) for spec in system_specs]

    builder = Builder(annotator, tokenizer, max_length=max_length)
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, (source, gold) in enumerate(zip(sources, golds)):
            group = builder.group(source, gold, systems, i, k)
            if group is not None:
                out.write(json.dumps(group, ensure_ascii=False) + '\n')
    return builder.stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--gold', required=True, type=Path)
    parser.add_argument('--system', required=True, action='append', metavar='NAME=PATH')
    parser.add_argument('--k', type=int, default=5, help='candidates per system per source')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--tokenizer', default=None,
                        help='e.g. microsoft/deberta-v3-large: drop groups longer than --max-length')
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    import errant
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stats = build(args.source, args.gold, args.system, args.k, args.out, errant.load('en'),
                  tokenizer, args.max_length)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
