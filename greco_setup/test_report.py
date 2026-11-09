"""Test-set results of the three enhancements in one table, and the BEA-2019 Codabench zips.

Rows, in the order the project builds them up:
1. curriculum training: BART and T5 after stage 1 only (*.s1) and after stage 1 + stage 2
   (*.s1s2), from cluster/test_singles.sh (experiments/test/<run>/);
2. GRECO combination of BART + T5 + GECToR at k = 1, with GECToR alone for reference;
3. GRECO at k = 2, 3, 5 (outputs/greco/tuning/conll14.errant.beam/, cluster/greco_tune.sh).

CoNLL-2014 is scored here against data/test.conll14.m2 with the NUS M2 scorer (its official
metric) and ERRANT, so it needs errant, Python <= 3.12 and git-lfs's copy of the gold M2:
    python -m greco_setup.test_report

BEA-2019 has no public gold. --zips writes one zip per output to outputs/greco/codabench_bea19/
for Codabench (one file, no folders); its F0.5 scores go into experiments/test/codabench.json
by name (the script writes a template with the names), and the table then shows them.

Writes experiments/test/README.md and experiments/test/scores.csv. Outputs not yet produced
are listed as missing.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import tempfile
import zipfile

from greco_setup.score_tuning import errant_scores, m2_scorer, m2_scores

TEST = Path("experiments/test")
GRECO = Path("outputs/greco/tuning")
SOURCES = {"conll14": Path("data/test.conll14.src"), "bea19": Path("data/test.bea19.src")}
GOLD = Path("data/test.conll14.m2")
ZIPS = Path("outputs/greco/codabench_bea19")
BEST = Path("outputs/greco/tuning/dev.errant.beam/best.json")
SINGLES = [("bart-base.s1", "BART, stage 1 only (no curriculum)", "1. curriculum"),
           ("bart-base.s1s2", "BART, stage 1 + 2 (curriculum)", "1. curriculum"),
           ("t5-base.s1", "T5, stage 1 only (no curriculum)", "1. curriculum"),
           ("t5-base.s1s2", "T5, stage 1 + 2 (curriculum)", "1. curriculum")]


def outputs():
    """(name, label, enhancement, {test set: prediction path}) of every output in the table."""
    rows = [(run, label, group, {t: TEST / run / f"{t}_test.pred" for t in SOURCES})
            for run, label, group in SINGLES]
    rows.append(("gector", "GECToR (official RoBERTa checkpoint)", "2. GRECO k = 1",
                 {t: Path(f"outputs/gector/{t}.tgt") for t in SOURCES}))
    best = json.loads(BEST.read_text(encoding="utf-8"))
    for k in (1, 2, 3, 5):
        group = "2. GRECO k = 1" if k == 1 else "3. larger k"
        # k = 1: every vote mode gives the same output, so one voted row
        modes = ["none", "rank"] if k == 1 else ["none", "hypothesis", "system", "rank"]
        for mode in modes:
            alpha = 0.0 if mode == "none" else float(best[f"k{k}"][mode])
            vote = "no voting" if mode == "none" else f"{mode} vote, α {alpha:.1f}"
            rows.append((f"greco.k{k}.{mode}", f"GRECO BART + T5 + GECToR, k = {k}, {vote}", group,
                         {t: GRECO / f"{t}.errant.beam/k{k}/{mode}/alpha{alpha:.1f}.tgt" for t in SOURCES}))
    return rows


def line_count(path):
    return len(path.read_text(encoding="utf-8").splitlines())


def check(path, test):
    """Whether path is a complete prediction file for test (one line per source)."""
    return path.is_file() and line_count(path) == line_count(SOURCES[test])


def write_zips(rows):
    ZIPS.mkdir(parents=True, exist_ok=True)
    for name, _, _, paths in rows:
        if check(paths["bea19"], "bea19"):
            with zipfile.ZipFile(ZIPS / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                z.write(paths["bea19"], f"{name}.txt")
    print(f"Codabench zips in {ZIPS}/")


def codabench_scores(rows):
    """{name: BEA-2019 F0.5} from experiments/test/codabench.json; writes a template if missing."""
    path = TEST / "codabench.json"
    if not path.exists():
        TEST.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({name: None for name, *_ in rows}, indent=2) + "\n")
        print(f"wrote {path}: fill in the Codabench F0.5 of each zip (e.g. 59.8)")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zips", action="store_true", help="also write the BEA-2019 Codabench zips")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    rows = outputs()
    if GOLD.read_text(encoding="utf-8")[:40].startswith("version https://git-lfs"):
        raise SystemExit(f"{GOLD} is a Git LFS pointer; run git lfs pull")
    if args.zips:
        write_zips(rows)
    bea = codabench_scores(rows)

    ready = [(name, paths["conll14"]) for name, _, _, paths in rows if check(paths["conll14"], "conll14")]
    with tempfile.TemporaryDirectory() as tmp:
        scorer = m2_scorer(tmp)
        files = [p for _, p in ready]
        n = len(files)
        with ProcessPoolExecutor(args.jobs) as pool:
            e = list(pool.map(errant_scores, [SOURCES["conll14"]] * n, files, [GOLD] * n))
            m = list(pool.map(m2_scores, [scorer] * n, files, [GOLD] * n))
    conll = {name: {**a, **b} for (name, _), a, b in zip(ready, e, m)}

    def cell(s, prefix=""):
        return f"{s[prefix + 'f0_5']*100:.1f} ({s[prefix + 'precision']*100:.1f} / {s[prefix + 'recall']*100:.1f})"

    lines = ["# Test-set results", "",
             "Nothing is tuned on these sets: the checkpoints were chosen on dev_tune and GRECO's "
             "vote coefficients α on the full W&I+LOCNESS dev set (outputs/greco/tuning/dev.errant.beam). "
             "F0.5 (P / R). CoNLL-2014 (1,312 sentences) is scored with the M2 scorer, its official "
             "metric, and ERRANT; BEA-2019 (4,477) with ERRANT on Codabench. Single BART and T5 "
             "rows are beam-5 1-best (training.evaluate); GRECO pools the top-k of 10-beam lists "
             "(cluster/nbest_beam.sh). Written by greco_setup/test_report.py.", "",
             "| Enhancement | Output | CoNLL-14 M2 | CoNLL-14 ERRANT | BEA-19 ERRANT (Codabench) |",
             "|---|---|---|---|---|"]
    table = []
    for name, label, group, paths in rows:
        s = conll.get(name)
        b = bea.get(name)
        lines.append(f"| {group} | {label} | {cell(s, 'm2_') if s else 'missing'} | "
                     f"{cell(s) if s else 'missing'} | {b if b is not None else '–'} |")
        table.append({"name": name, "group": group, "label": label, "bea19_f0_5": b,
                      **({k: s[k] for k in ("m2_precision", "m2_recall", "m2_f0_5", "precision",
                                            "recall", "f0_5")} if s else {})})
    (TEST / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = ["name", "group", "label", "m2_precision", "m2_recall", "m2_f0_5", "precision", "recall",
              "f0_5", "bea19_f0_5"]
    with (TEST / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print("\n".join(lines))
    missing = [name for name, *_ in rows if name not in conll]
    if missing:
        print("\nnot produced yet (CoNLL-14):", ", ".join(missing))


if __name__ == "__main__":
    main()
