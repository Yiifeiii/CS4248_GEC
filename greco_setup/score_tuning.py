"""Score tune_vote.py predictions with ERRANT and pick the best alpha per (k, vote mode).

Needs errant (e.g. the gec-train environment), not the GRECO environment:
    python -m greco_setup.score_tuning --split dev_tune.errant

Writes <out>/<split>/scores.csv (every setting), report.md (the best alpha per k and
vote mode, and the best setting overall) and, unless the split is dev_test, best.json,
which tune_vote.py --best-from then runs on another split. Reference edits:
data/curriculum/<split>.m2, or data/dev.m2 for the full dev set (--split dev...).
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


def full_dev_reference():
    """The M2 of all of dev in data/dev.src order, rebuilt from the dev_tune and dev_test
    halves. data/dev.m2 is stored in Git LFS, so checkouts without git-lfs (the
    cluster) only have a pointer file."""
    blocks = {}
    for half in ("dev_tune", "dev_test"):
        ids = [int(x) for x in Path(f"data/curriculum/{half}.ids").read_text().split()]
        half_blocks = Path(f"data/curriculum/{half}.m2").read_text(encoding="utf-8").strip().split("\n\n")
        if len(ids) != len(half_blocks):
            raise ValueError(f"{half}: {len(ids)} ids but {len(half_blocks)} M2 blocks")
        blocks.update(zip(ids, half_blocks))
    if sorted(blocks) != list(range(len(blocks))):
        raise ValueError("dev_tune and dev_test ids do not cover dev exactly once")
    return [blocks[i] for i in range(len(blocks))]


def reference_subset(split, out_dir, n):
    """The split's reference M2, cut to the first n sentences for --limit runs."""
    base = split.split(".")[0]
    if base == "dev":
        blocks = full_dev_reference()
    else:
        ref = Path(f"data/curriculum/{base}.m2")
        blocks = ref.read_text(encoding="utf-8").strip().split("\n\n")
        if n == len(blocks):
            return ref
    sources = (out_dir / "source.txt").read_text(encoding="utf-8").splitlines()
    if [b.split("\n")[0][2:] for b in blocks[:n]] != sources:
        raise ValueError(f"reference M2 does not match {out_dir / 'source.txt'}")
    path = out_dir / "reference.m2"
    path.write_text("\n\n".join(blocks[:n]) + "\n", encoding="utf-8")
    return path


def pick(rows):
    """Best row by F0.5; ties go to the smaller alpha."""
    return max(rows, key=lambda r: (r["f0_5"], -r["alpha"]))


def scores(r):
    return f"{r['f0_5']*100:.1f} ({r['precision']*100:.1f} / {r['recall']*100:.1f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev_tune.errant",
                        help="folder under --out, e.g. dev_tune.errant, dev_test.errant or dev.errant")
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
        results = list(pool.map(errant_scores, [source] * len(files), files, [ref] * len(files)))
    rows = []
    for path, s in zip(files, results):
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
    ks = sorted({r["k"] for r in rows})
    modes = sorted({r["mode"] for r in rows} - {"none"})
    best = {}
    lines = [f"# GRECO tuning on {args.split} ({n} sentences)", "",
             "ERRANT F0.5 (P / R) at the best α per k and vote mode; α 0 is no voting.", "",
             "| k | no voting | " + " | ".join(modes) + " |",
             "|---|---|" + "---|" * len(modes)]
    for k in ks:
        base = [r for r in rows if r["k"] == k and r["mode"] == "none"]
        best[f"k{k}"] = {}
        cells = [scores(base[0]) if base else "–"]
        for mode in modes:
            winner = pick(base + [r for r in rows if r["k"] == k and r["mode"] == mode])
            best[f"k{k}"][mode] = winner["alpha"]
            cells.append(f"{scores(winner)}, α {winner['alpha']:.1f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    top = pick(rows)
    lines += ["", f"**Best setting:** k = {top['k']}, {top['mode']}, α {top['alpha']:.1f}: F0.5 {scores(top)}."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    print("\nbest alpha per (k, mode):", json.dumps(best))
    if not args.split.startswith("dev_test"):
        (out_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_dir / 'best.json'}")


if __name__ == "__main__":
    main()
