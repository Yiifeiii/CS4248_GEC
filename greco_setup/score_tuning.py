"""Score tune_vote.py predictions with ERRANT and pick the best alpha per (k, vote mode).

Needs errant (e.g. the gec-train environment), not the GRECO environment:
    python -m greco_setup.score_tuning --split dev_tune.errant

Writes <out>/<split>/scores.csv (every setting) and, for dev_tune, best.json: the alpha
with the highest ERRANT F0.5 for each k and vote mode, which tune_vote.py --best-from
then runs on dev_test. Reference edits: data/curriculum/<split>.m2.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

TOOLS = Path(sys.executable).parent


def errant_scores(source, hyp, ref):
    """ERRANT span-based correction TP, FP, FN, P, R, F0.5 of one prediction file."""
    with tempfile.TemporaryDirectory() as tmp:
        m2 = Path(tmp, "hyp.m2")
        subprocess.run([TOOLS / "errant_parallel", "-orig", source, "-cor", hyp, "-out", m2],
                       check=True, capture_output=True)
        out = subprocess.run([TOOLS / "errant_compare", "-hyp", m2, "-ref", ref],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    values = out[next(i for i, line in enumerate(out) if line.startswith("TP")) + 1].split()
    return dict(zip(["tp", "fp", "fn", "precision", "recall", "f0_5"], map(float, values)))


def reference_subset(split, out_dir, n):
    """The split's reference M2, cut to the first n sentences for --limit runs."""
    ref = Path(f"data/curriculum/{split.split('.')[0]}.m2")
    blocks = ref.read_text(encoding="utf-8").strip().split("\n\n")
    if n == len(blocks):
        return ref
    path = out_dir / "reference.m2"
    path.write_text("\n\n".join(blocks[:n]) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev_tune.errant",
                        help="folder under --out, e.g. dev_tune.errant or dev_test.errant")
    parser.add_argument("--out", type=Path, default=Path("outputs/greco/tuning"))
    parser.add_argument("--jobs", type=int, default=8, help="files scored in parallel")
    args = parser.parse_args()

    out_dir = args.out / args.split
    source = out_dir / "source.txt"
    n = len(source.read_text(encoding="utf-8").splitlines())
    ref = reference_subset(args.split, out_dir, n)
    files = sorted(out_dir.glob("k*/*/alpha*.tgt"))
    if not files:
        sys.exit(f"no predictions under {out_dir}; run greco_setup.tune_vote first")

    with ProcessPoolExecutor(args.jobs) as pool:
        scores = list(pool.map(errant_scores, [source] * len(files), files, [ref] * len(files)))
    rows = []
    for path, s in zip(files, scores):
        k = int(path.parts[-3][1:])
        mode = path.parts[-2]
        alpha = float(re.fullmatch(r"alpha([0-9.]+)\.tgt", path.name).group(1))
        rows.append({"k": k, "mode": mode, "alpha": alpha, **s})
    rows.sort(key=lambda r: (r["k"], r["mode"], r["alpha"]))
    with (out_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'k':>2} {'mode':<10} {'alpha':>5} | {'P':>5} {'R':>5} {'F0.5':>5}")
    for r in rows:
        print(f"{r['k']:>2} {r['mode']:<10} {r['alpha']:>5.1f} | {r['precision']*100:5.1f} "
              f"{r['recall']*100:5.1f} {r['f0_5']*100:5.1f}")

    # Best alpha per (k, mode); alpha 0 ('none') competes in every mode
    best = {}
    for k in sorted({r["k"] for r in rows}):
        base = [r for r in rows if r["k"] == k and r["mode"] == "none"]
        best[f"k{k}"] = {}
        for mode in sorted({r["mode"] for r in rows if r["k"] == k} - {"none"}):
            candidates = base + [r for r in rows if r["k"] == k and r["mode"] == mode]
            winner = max(candidates, key=lambda r: (r["f0_5"], -r["alpha"]))
            best[f"k{k}"][mode] = winner["alpha"]
    print("\nbest alpha per (k, mode):", json.dumps(best))
    if not args.split.startswith("dev_test"):
        (out_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_dir / 'best.json'}")


if __name__ == "__main__":
    main()
