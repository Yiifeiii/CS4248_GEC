"""Turn an n-best JSONL into per-rank files for GRECO.

Writes <out_dir>/texts/rank1.txt ... rankK.txt (one sentence per source line;
this folder is what GRECO's rerank.py reads) and, with --m2,
<out_dir>/m2/<name>-1.m2 ... for GRECO's run_combination.py, where each rank
acts as one system.
"""
import argparse
import json
import os
import subprocess
import sys


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
    texts_dir = os.path.join(args.out_dir, "texts")
    os.makedirs(texts_dir, exist_ok=True)

    src_path = os.path.join(args.out_dir, "source.txt")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write("".join(row["src"] + "\n" for row in rows))

    padded = 0
    rank_paths = []
    for r in range(args.k):
        path = os.path.join(texts_dir, f"rank{r + 1}.txt")
        rank_paths.append(path)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                cands = row["cands"]
                if r < len(cands):
                    text = cands[r]["text"]
                else:
                    padded += 1
                    if args.pad == "top1" and cands:
                        text = cands[0]["text"]
                    else:
                        text = " ".join(row["src"].split())
                f.write(text + "\n")
    print(f"wrote {args.k} rank files for {len(rows)} sources to {texts_dir} "
          f"({padded} slots padded with {args.pad})")

    if args.m2:
        m2_dir = os.path.join(args.out_dir, "m2")
        os.makedirs(m2_dir, exist_ok=True)
        errant_parallel = os.path.join(os.path.dirname(sys.executable), "errant_parallel")
        for r, path in enumerate(rank_paths):
            out = os.path.join(m2_dir, f"{args.name}-{r + 1}.m2")
            subprocess.run([errant_parallel, "-orig", src_path, "-cor", path, "-out", out],
                           check=True)
            print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest", required=True, help="JSONL from generate_nbest.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k", type=int, default=5, help="number of ranks to export")
    parser.add_argument("--pad", choices=["top1", "source"], default="top1",
                        help="filler when a sentence has fewer than k unique candidates")
    parser.add_argument("--m2", action="store_true", help="also write one ERRANT M2 file per rank")
    parser.add_argument("--name", default="BART", help="system name prefix for the M2 files")
    main(parser.parse_args())
