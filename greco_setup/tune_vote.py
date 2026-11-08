"""Run GRECO over a grid of k, vote mode and vote coefficient (alpha) on part of dev.

Writes one prediction file per setting; greco_setup/score_tuning.py scores them with
ERRANT (it needs a different environment) and picks the best alpha per (k, vote mode).

Tuning on dev_tune (GRECO environment, repository root):
    python -m greco_setup.tune_vote --split dev_tune --device cuda

Or the whole grid on all of dev (--split dev); score_tuning.py then also reports a
two-fold estimate that chooses settings on one half and scores them on the other.

Then the chosen settings, once, on dev_test:
    python -m greco_setup.tune_vote --split dev_test --best-from outputs/greco/tuning/dev_tune/best.json

Edits come from ERRANT by default (--edits errant: run greco_setup.extract_edits first,
in an environment with errant) or from difflib (--edits difflib). ERRANT runs are written
to <split>.errant, difflib runs to <split>; --tag T appends .T (e.g. for other --system
files) and --limit N appends .first<N>.

Output: <out>/<folder>/source.txt and <out>/<folder>/k<k>/<mode>/alpha<a>.tgt, aligned with
data/curriculum/<split>.ids. alpha 0 does not depend on the vote mode and is written
once per k as k<k>/none/alpha0.0.tgt. Finished files are skipped, so a job that hits
its time limit can be resubmitted. GRECO scores are cached in <out>/score_cache.jsonl
and shared by every setting and split.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from greco_setup.combine import VOTE_MODES, CachedScorer, GrecoScorer, combine, edit_support, read_record
from greco_setup.prepare import build, read_edits, read_lines, read_system

DEFAULT_SYSTEMS = ["bart=outputs/bart/full/diverse.tgt", "t5=outputs/t5/full/diverse.tgt",
                   "gector=outputs/gector/dev.tgt"]
DEFAULT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def settings(args):
    """(k, mode, alpha) to run. alpha 0 is mode-independent, so it is run once as 'none'."""
    if args.best_from:
        best = json.loads(Path(args.best_from).read_text(encoding="utf-8"))
        runs = []
        for k in args.k:
            runs.append((k, "none", 0.0))
            runs += [(k, mode, float(best[f"k{k}"][mode])) for mode in args.modes
                     if float(best[f"k{k}"][mode]) > 0]
        return list(dict.fromkeys(runs))
    runs = []
    for k in args.k:
        if 0.0 in args.alphas:
            runs.append((k, "none", 0.0))
        runs += [(k, mode, a) for mode in args.modes for a in args.alphas if a > 0]
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev_tune",
                        help="data/curriculum/<split>.ids selects the sentences; 'dev' is all of data/dev.src")
    parser.add_argument("--source", type=Path, default=Path("data/dev.src"))
    parser.add_argument("--system", action="append", help="NAME=PATH (k lines per source); default BART, T5, GECToR")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--modes", nargs="+", choices=VOTE_MODES, default=list(VOTE_MODES))
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--best-from", type=Path, help="best.json from score_tuning.py: run only those alphas")
    parser.add_argument("--out", type=Path, default=Path("outputs/greco/tuning"))
    parser.add_argument("--limit", type=int, help="only the first N sentences of the split (test runs)")
    parser.add_argument("--edits", choices=["errant", "difflib"], default="errant",
                        help="how hypotheses are split into edits")
    parser.add_argument("--tag", help="suffix for the output folder, e.g. beam for plain beam search lists")
    parser.add_argument("--greco-repo", default="external/greco")
    parser.add_argument("--checkpoint", default="models/greco/checkpoint.bin")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    sources_all = read_lines(args.source)
    if args.split == "dev":
        ids = list(range(len(sources_all)))
    else:
        ids = [int(x) for x in Path(f"data/curriculum/{args.split}.ids").read_text().split()]
    if args.limit:
        ids = ids[:args.limit]
    specs = args.system or DEFAULT_SYSTEMS
    systems = [read_system(spec, len(sources_all)) for spec in specs]
    sources = [sources_all[i] for i in ids]
    systems = [(name, [hyps[i] for i in ids]) for name, hyps in systems]
    edits = None
    if args.edits == "errant":
        edits = {name: [per[i] for i in ids]
                 for name, per in (read_edits(spec, len(sources_all)) for spec in specs)}

    folder = args.split + (".errant" if args.edits == "errant" else "") + (f".{args.tag}" if args.tag else "")
    out = args.out / (folder if not args.limit else f"{folder}.first{args.limit}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
    (out / "ids.txt").write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")

    runs = settings(args)
    todo = [(k, m, a) for k, m, a in runs if not (out / f"k{k}/{m}/alpha{a:.1f}.tgt").is_file()]
    print(f"{len(sources)} sentences, {len(runs)} settings, {len(todo)} to run", flush=True)
    if not todo:
        return
    scorer = CachedScorer(GrecoScorer(args.greco_repo, args.checkpoint, args.device),
                          args.out / "score_cache.jsonl")
    print(f"score cache: {len(scorer.cache)} entries", flush=True)

    records = {}
    for k, mode, alpha in todo:
        if k not in records:
            records[k] = build(sources, systems, k, edits)
        started, cached = time.time(), len(scorer.cache)
        best = []
        for i, record in enumerate(records[k], 1):
            source, pool = read_record(record)
            support = edit_support(record, mode) if alpha > 0 else None
            result = combine(source, pool, scorer, args.beam_size, args.batch_size,
                             support=support, vote_coef=alpha)
            best.append(result["best_sentence"])
            if i % 200 == 0:
                print(f"  k={k} {mode} alpha={alpha:.1f}: {i}/{len(sources)}", flush=True)
        path = out / f"k{k}/{mode}/alpha{alpha:.1f}.tgt"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(best) + "\n", encoding="utf-8")
        tmp.rename(path)
        print(f"k={k} {mode} alpha={alpha:.1f}: {time.time() - started:.0f}s, "
              f"{len(scorer.cache) - cached} new scores -> {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
