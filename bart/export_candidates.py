"""Turn an n-best JSONL into the shared prediction format for GRECO.

Writes one .tgt file with k consecutive lines per source sentence, best first:
lines k*(i-1)+1 .. k*i belong to source line i. Every system (BART, T5, GECToR)
uses this format, and greco_setup/prepare.py reads it.

    python bart/export_candidates.py --nbest outputs/bart/dev30.beam.nbest.jsonl \
        --out outputs/bart/dev30.beam.tgt --k 5
"""
import argparse
import json
import os


def read_nbest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("header"):
                rows.append(row)
    return rows


def main(args):
    rows = read_nbest(args.nbest)
    lines = []
    padded = 0
    for row in rows:
        cands = row["cands"]
        for r in range(args.k):
            if r < len(cands):
                text = cands[r]["text"]
            else:
                # Keep exactly k lines per source so the file stays aligned
                padded += 1
                if args.pad == "top1" and cands:
                    text = cands[0]["text"]
                else:
                    text = " ".join(row["src"].split())
            lines.append(text)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))
    print(f"wrote {args.k} candidates for each of {len(rows)} sources to {args.out} "
          f"({padded} slots padded with {args.pad})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest", required=True, help="JSONL from generate_nbest.py")
    parser.add_argument("--out", required=True, help="output .tgt file")
    parser.add_argument("--k", type=int, default=5, help="candidates per source sentence")
    parser.add_argument("--pad", choices=["top1", "source"], default="top1",
                        help="filler when a sentence has fewer than k unique candidates")
    main(parser.parse_args())
