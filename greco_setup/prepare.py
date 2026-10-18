"""Connect system output files to combine.py.

build    source file + one prediction file per system  ->  combine.py input JSON
collect  combine.py output JSON                         ->  one best sentence per line

A prediction file (outputs/<system>/<split>.tgt) is plain text with k consecutive
lines per source sentence, best hypothesis first (k = 1 for GECToR, k = 5 for
BART/T5 top-5). k is inferred from the line count. All files must use the same
tokenization as the source.

    python -m greco_setup.prepare build --source outputs/dev30.src \
        --system gector=outputs/gector/dev30.tgt \
        --system bart=outputs/bart/dev30.beam.tgt \
        --system t5=outputs/t5/dev30.tgt \
        --k 3 --output outputs/greco/dev30.input.json

    python -m greco_setup.combine --input outputs/greco/dev30.input.json \
        --output outputs/greco/dev30.output.json --device cuda

    python -m greco_setup.prepare collect --input outputs/greco/dev30.output.json \
        --output outputs/greco/dev30.tgt
"""

import argparse
import json
from pathlib import Path


def read_lines(path):
    text = path.read_text(encoding="utf-8")
    # Drop only the final newline: a trailing empty line is a real (empty) hypothesis
    return (text[:-1] if text.endswith("\n") else text).split("\n")


def read_system(spec, n_sources):
    """Parse NAME=PATH and split the file into k hypotheses per source sentence."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise ValueError(f"--system must look like NAME=PATH, got {spec!r}")
    lines = read_lines(Path(path))
    if len(lines) % n_sources:
        raise ValueError(f"{path}: {len(lines)} lines is not a multiple of {n_sources} source sentences")
    k = len(lines) // n_sources
    return name, [lines[i * k:(i + 1) * k] for i in range(n_sources)]


def build(sources, systems, k=None):
    """One combine.py record per source sentence, with proposals from every system."""
    records = []
    for i, source in enumerate(sources):
        proposals = []
        for name, hypotheses in systems:
            for rank, sentence in enumerate(hypotheses[i][:k], 1):
                # Some generators return an empty string; combine.py rejects those
                if sentence.strip():
                    proposals.append({"system": name, "rank": rank, "sentence": sentence})
        if not proposals:
            # No usable hypothesis: proposing the source itself means "no edits"
            proposals.append({"system": "source", "rank": 0, "sentence": source})
        records.append({"source": source, "proposals": proposals})
    return records


def collect(results):
    # A dry run has no best sentence; fall back to the source so lines stay aligned
    return [r["best_sentence"] if r["best_sentence"] is not None else r["source"] for r in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", required=True, type=Path)
    b.add_argument("--system", required=True, action="append", help="NAME=PATH, repeat for each system")
    b.add_argument("--k", type=int, help="keep at most the top-k hypotheses per system")
    b.add_argument("--output", required=True, type=Path)
    c = sub.add_parser("collect")
    c.add_argument("--input", required=True, type=Path)
    c.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "build":
            if args.k is not None and args.k < 1:
                raise ValueError("--k must be positive")
            sources = read_lines(args.source)
            systems = [read_system(spec, len(sources)) for spec in args.system]
            records = build(sources, systems, args.k)
            args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            counts = ", ".join(f"{name}: k={len(hyps[0])}" for name, hyps in systems)
            print(f"Wrote {len(records)} records ({counts}) to {args.output}")
        else:
            sentences = collect(json.loads(args.input.read_text(encoding="utf-8")))
            args.output.write_text("\n".join(sentences) + "\n", encoding="utf-8")
            print(f"Wrote {len(sentences)} sentences to {args.output}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
