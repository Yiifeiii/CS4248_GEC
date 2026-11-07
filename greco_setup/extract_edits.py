"""Extract the ERRANT edits of every hypothesis, for GRECO to combine.

Needs errant and its spaCy model (e.g. the gec-train environment), not the GRECO one:
    python -m greco_setup.extract_edits

For each prediction file (k lines per line of --source) writes <file>.errant.jsonl:
a header with the file's sha256, then one line per hypothesis holding its edits as
[{"start", "end", "replacement"}, ...] in source-token offsets. These are the same
edits ERRANT scores against the gold M2, so a correction is split into the same units
in every hypothesis and system (difflib, used otherwise, merges adjacent changes).
Up-to-date files are skipped.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from greco_setup.combine import proposal_edits, render
from greco_setup.prepare import edits_path, file_hash, read_lines, read_system

DEFAULT_SYSTEMS = ["bart=outputs/bart/full/diverse.tgt", "t5=outputs/t5/full/diverse.tgt",
                   "gector=outputs/gector/dev.tgt"]

annotator = None


def load():
    global annotator
    import errant
    annotator = errant.load("en")


def annotate(job):
    """ERRANT edits of each hypothesis of one source sentence."""
    source, hypotheses = job
    orig = annotator.parse(source)
    result = []
    for hyp in hypotheses:
        if not hyp.strip():
            result.append([])
            continue
        edits = []
        for e in annotator.annotate(orig, annotator.parse(hyp)):
            last = edits[-1] if edits else None
            # ERRANT can split one insertion into several at the same position ("their" ->
            # "his or her": +his, +or, their->her); combine.py treats insertions at one
            # position as alternatives, so join them in order
            if last and e.o_start == e.o_end == last["start"] == last["end"]:
                last["replacement"] = f"{last['replacement']} {e.c_str}"
            else:
                edits.append({"start": e.o_start, "end": e.o_end, "replacement": e.c_str})
        # The edits must rebuild the hypothesis and be usable by combine.py
        if render(source.split(), proposal_edits(source.split(), {"edits": edits})) != " ".join(hyp.split()):
            raise ValueError(f"ERRANT edits do not rebuild {hyp!r} from {source!r}")
        result.append(edits)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("data/dev.src"))
    parser.add_argument("--system", action="append", help="NAME=PATH; default BART, T5, GECToR on dev")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    sources = read_lines(args.source)
    with ProcessPoolExecutor(args.jobs, initializer=load) as pool:
        for spec in args.system or DEFAULT_SYSTEMS:
            path = spec.partition("=")[2]
            out, digest = edits_path(path), file_hash(path)
            if out.is_file():
                with out.open(encoding="utf-8") as f:
                    if json.loads(f.readline()).get("sha256") == digest:
                        print(f"{out} is up to date")
                        continue
            _, hyps = read_system(spec, len(sources))
            per_source = pool.map(annotate, zip(sources, hyps), chunksize=32)
            tmp = out.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"tgt": path, "sha256": digest, "source": str(args.source)}) + "\n")
                n = 0
                for edits in per_source:
                    for e in edits:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                        n += 1
            tmp.rename(out)
            print(f"wrote {n} hypotheses' edits to {out}", flush=True)


if __name__ == "__main__":
    main()
