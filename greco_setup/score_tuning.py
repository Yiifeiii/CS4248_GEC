"""Score tune_vote.py predictions with ERRANT and pick the best alpha per (k, vote mode).

Needs errant (e.g. the gec-train environment), not the GRECO environment:
    python -m greco_setup.score_tuning --split dev_tune.errant

Writes <out>/<split>/scores.csv (every setting), report.md (the best alpha per k and
vote mode, and the best setting overall) and, for dev_tune and dev, best.json, which
tune_vote.py --best-from then runs on another split. Reference edits:
data/curriculum/<split>.m2, or data/dev.m2 for the full dev set (--split dev...).

The CoNLL-2014 test set (--split conll14...) is scored against data/test.conll14.m2 with
the NUS M2 scorer (its official metric, converted to Python 3 with lib2to3, so Python <=
3.12) as well as ERRANT. Its report lists every setting rather than the best
alpha, since the alphas were fixed on dev; --system NAME=PATH adds the top hypothesis of
a single system (a file with k lines per source) as a baseline:
    python -m greco_setup.score_tuning --split conll14.errant.beam \
        --system bart=outputs/bart/conll14.tgt --system t5=outputs/t5/conll14.tgt \
        --system gector=outputs/gector/conll14.tgt
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
# Test sets with a gold M2 (BEA-2019 has none: score it on Codabench)
TEST_REFS = {"conll14": Path("data/test.conll14.m2")}


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


def m2_scorer(directory):
    """A Python 3 copy of the bundled (Python 2) M2 scorer in directory, made with lib2to3
    as in training/metrics.py, which this does not import because it needs torch."""
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package
    converter = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    for name in ("util.py", "levenshtein.py", "m2scorer.py"):
        text = str(converter.refactor_string(Path("m2scorer/scripts", name).read_text(), name))
        text = text.replace("puffer.decode('utf8')", "puffer").replace('line.decode("utf8")', "line")
        Path(directory, name).write_text(text)
    return Path(directory, "m2scorer.py")


def m2_scores(scorer, hyp, ref):
    """NUS M2 scorer precision, recall and F0.5 of one prediction file."""
    out = subprocess.run([sys.executable, scorer, hyp, ref], check=True, capture_output=True,
                         text=True).stdout
    values = [float(v) for v in re.findall(r":\s*([0-9.]+)", out)]
    if len(values) != 3:
        raise ValueError(f"unexpected M2 scorer output for {hyp}: {out}")
    return dict(zip(["m2_precision", "m2_recall", "m2_f0_5"], values))


def top_hypotheses(spec, n, path):
    """Write the first hypothesis per source of NAME=PATH (k lines per source) to path."""
    name, src = spec.split("=", 1)
    lines = Path(src).read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or len(lines) % n:
        raise ValueError(f"{src}: {len(lines)} lines is not a multiple of {n} sources")
    k = len(lines) // n
    path.write_text("\n".join(lines[::k]) + "\n", encoding="utf-8")
    return name, path


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
    if base in TEST_REFS:
        blocks = TEST_REFS[base].read_text(encoding="utf-8").strip().split("\n\n")
        if blocks[0].startswith("version https://git-lfs"):
            raise ValueError(f"{TEST_REFS[base]} is a Git LFS pointer; run git lfs pull")
    elif base == "dev":
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


def setting(path):
    """k, vote mode and alpha of outputs/greco/tuning/<split>/k<k>/<mode>/alpha<a>.tgt"""
    return {"k": int(path.parts[-3][1:]), "mode": path.parts[-2],
            "alpha": float(re.fullmatch(r"alpha([0-9.]+)\.tgt", path.name).group(1))}


def score_test(args, out_dir, source, n, ref, files):
    """M2 and ERRANT scores of every setting and the --system baselines on a test split."""
    with tempfile.TemporaryDirectory() as tmp:
        scorer = m2_scorer(tmp)
        baselines = [top_hypotheses(spec, n, Path(tmp, f"top{i}.txt"))
                     for i, spec in enumerate(args.system)]
        labels = [{"output": name, "k": "", "mode": "", "alpha": ""} for name, _ in baselines]
        labels += [{"output": "greco", **setting(p)} for p in files]
        paths = [p for _, p in baselines] + files
        with ProcessPoolExecutor(args.jobs) as pool:
            e = list(pool.map(errant_scores, [source] * len(paths), paths, [ref] * len(paths)))
            m = list(pool.map(m2_scores, [scorer] * len(paths), paths, [ref] * len(paths)))
    rows = [{**label, **a, **b} for label, a, b in zip(labels, e, m)]
    with (out_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def cell(r, prefix):
        return (f"{r[prefix + 'f0_5']*100:.1f} ({r[prefix + 'precision']*100:.1f} / "
                f"{r[prefix + 'recall']*100:.1f})")
    lines = [f"# {args.split} ({n} sentences)", "",
             "Settings (α) fixed on dev; nothing is tuned on this set. The M2 scorer is the "
             "official CoNLL-2014 metric. F0.5 (P / R).", "",
             "| output | k | vote | α | M2 | ERRANT |", "|---|---|---|---|---|---|"]
    for r in rows:
        alpha = "" if r["alpha"] == "" else f"{r['alpha']:.1f}"
        lines.append(f"| {r['output']} | {r['k']} | {r['mode']} | {alpha} | {cell(r, 'm2_')} | "
                     f"{cell(r, '')} |")
    top = max((r for r in rows if r["output"] == "greco"), key=lambda r: r["m2_f0_5"])
    lines += ["", f"**Best GRECO setting by M2:** k = {top['k']}, {top['mode']}, α {top['alpha']:.1f}: "
              f"M2 {cell(top, 'm2_')}, ERRANT {cell(top, '')}."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


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
    parser.add_argument("--system", action="append", default=[],
                        help="NAME=PATH: also score this system's top hypothesis (test splits)")
    args = parser.parse_args()

    out_dir = args.out / args.split
    source = out_dir / "source.txt"
    n = len(source.read_text(encoding="utf-8").splitlines())
    ref = reference_subset(args.split, out_dir, n)
    files = sorted(out_dir.glob("k*/*/alpha*.tgt"))
    if not files:
        sys.exit(f"no predictions under {out_dir}; run greco_setup.tune_vote first")

    test = args.split.split(".")[0] in TEST_REFS
    if test:
        return score_test(args, out_dir, source, n, ref, files)

    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(errant_scores, [source] * len(files), files, [ref] * len(files)))
    rows = [{**setting(path), **s} for path, s in zip(files, results)]
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
    if args.split.split(".")[0] in ("dev_tune", "dev"):
        (out_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_dir / 'best.json'}")


if __name__ == "__main__":
    main()
