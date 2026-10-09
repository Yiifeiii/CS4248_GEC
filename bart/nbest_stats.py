"""Diversity and pooled-edit statistics of n-best lists against a gold M2 file.

For each k, the top-k candidates of a sentence are pooled and their ERRANT
edits compared with the gold edits (annotator 0). At k=1 the precision, recall
and F0.5 are the ordinary 1-best scores.
"""
import argparse
import json

import errant
from tqdm import tqdm


def read_nbest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("header"):
                rows.append(row)
    return rows


def read_gold(path, annotator_id=0):
    """Return [(source, {(start, end, correction)})] from an M2 file."""
    gold = []
    with open(path, encoding="utf-8") as f:
        entries = f.read().strip().split("\n\n")
    for entry in entries:
        lines = entry.split("\n")
        edits = set()
        for line in lines[1:]:
            fields = line[2:].split("|||")
            # UNK edits have no usable correction; ERRANT's scorer skips them too.
            if fields[1] in ("noop", "UNK") or int(fields[-1]) != annotator_id:
                continue
            start, end = map(int, fields[0].split())
            edits.add((start, end, " ".join(fields[2].split())))
        gold.append((lines[0][2:], edits))
    return gold


def f_beta(p, r, beta=0.5):
    return (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0


def main(args):
    rows = read_nbest(args.nbest)
    gold = read_gold(args.gold)
    assert len(rows) == len(gold), f"{len(rows)} n-best rows vs {len(gold)} gold sentences"
    annotator = errant.load("en")
    ks = sorted(args.k)
    kmax = ks[-1]

    stats = {k: dict(unique=0, short=0, pooled=0, tp=0, n_gold=0) for k in ks}
    for row, (gold_src, gold_edits) in zip(tqdm(rows), gold):
        assert " ".join(row["src"].split()) == " ".join(gold_src.split()), \
            f"source mismatch at id {row['id']}"
        orig = annotator.parse(row["src"])
        cand_edits = []
        for cand in row["cands"][:kmax]:
            if cand["is_source"]:
                cand_edits.append(set())
                continue
            cor = annotator.parse(cand["text"])
            cand_edits.append({(e.o_start, e.o_end, e.c_str)
                               for e in annotator.annotate(orig, cor)})
        for k in ks:
            pooled = set().union(*cand_edits[:k]) if cand_edits else set()
            s = stats[k]
            s["unique"] += min(k, len(cand_edits))
            s["short"] += len(cand_edits) < k
            s["pooled"] += len(pooled)
            s["tp"] += len(pooled & gold_edits)
            s["n_gold"] += len(gold_edits)

    n = len(rows)
    print(f"{args.nbest}  ({n} sentences)")
    print(f"{'k':>2} {'uniq/sent':>9} {'<k cands':>9} {'edits/sent':>10} "
          f"{'pool P':>7} {'pool R':>7} {'F0.5':>6}")
    for k in ks:
        s = stats[k]
        p = s["tp"] / s["pooled"] if s["pooled"] else 1.0
        r = s["tp"] / s["n_gold"] if s["n_gold"] else 1.0
        print(f"{k:>2} {s['unique'] / n:>9.2f} {s['short'] / n:>9.1%} {s['pooled'] / n:>10.2f} "
              f"{p:>7.1%} {r:>7.1%} {f_beta(p, r):>6.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest", required=True, help="JSONL from generate_nbest.py")
    parser.add_argument("--gold", required=True, help="gold M2 file for the same sources")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    main(parser.parse_args())
