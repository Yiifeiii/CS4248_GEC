"""Run with python -m greco_setup.combine --input PATH [--dry-run]."""

import argparse
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import importlib.util
import json
import math
from pathlib import Path


@dataclass(frozen=True, order=True)
class Edit:
    start: int
    end: int
    replacement: tuple


def conflicts(a, b):
    if a.start == a.end and b.start == b.end:
        return a.start == b.start
    if a.start == a.end:
        return b.start < a.start < b.end
    if b.start == b.end:
        return a.start < b.start < a.end
    return max(a.start, b.start) < min(a.end, b.end)


def render(tokens, edits):
    result = []
    cursor = 0
    for edit in sorted(edits):
        result.extend(tokens[cursor:edit.start])
        result.extend(edit.replacement)
        cursor = edit.end
    result.extend(tokens[cursor:])
    return " ".join(result)


def read_record(record):
    source = record.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a nonempty tokenized sentence")
    source = " ".join(source.split())
    tokens = source.split()
    proposals = record.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("proposals must be a nonempty list")
    pool = set()
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("each proposal must be an object")
        if ("sentence" in proposal) == ("edits" in proposal):
            raise ValueError("each proposal needs exactly one of sentence or edits")
        edits = []
        if "sentence" in proposal:
            if not isinstance(proposal["sentence"], str) or not proposal["sentence"].strip():
                raise ValueError("proposed sentence must be a nonempty string")
            target = proposal["sentence"].split()
            for tag, start, end, j, k in SequenceMatcher(
                a=tokens, b=target, autojunk=False
            ).get_opcodes():
                if tag != "equal":
                    edits.append(Edit(start, end, tuple(target[j:k])))
        else:
            if not isinstance(proposal["edits"], list):
                raise ValueError("edits must be a list")
            for item in proposal["edits"]:
                if not isinstance(item, dict):
                    raise ValueError("each edit must be an object")
                start, end = item.get("start"), item.get("end")
                replacement = item.get("replacement")
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start <= end <= len(tokens)
                        or not isinstance(replacement, str)):
                    raise ValueError("edit requires valid start/end offsets and a replacement string")
                edit = Edit(start, end, tuple(replacement.split()))
                if tuple(tokens[start:end]) != edit.replacement:
                    edits.append(edit)
        edits = sorted(set(edits))
        if any(conflicts(a, b) for i, a in enumerate(edits) for b in edits[i + 1:]):
            raise ValueError("a proposal contains conflicting edits")
        pool.update(edits)
    return source, sorted(pool)


class GrecoScorer:
    """Load the upstream class and released state dict, without copying its code."""

    def __init__(self, repo, checkpoint, device="cpu"):
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        if not (repo / "models.py").is_file() or not checkpoint.is_file():
            raise ValueError("GRECO source/checkpoint missing; follow greco_setup/README.md")
        import torch
        spec = importlib.util.spec_from_file_location("cs4248_upstream_greco", repo / "models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.GRECO("microsoft/deberta-v3-large")
        self.model.device = torch.device(device)
        self.model.to(self.model.device)
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def score(self, source, candidates):
        return self.model.score(source, candidates).detach().cpu().reshape(-1).tolist()


def combine(source, edits, scorer=None, beam_size=16, batch_size=8, max_candidates=4096):
    """Include/skip every pooled edit; score candidates at every beam step.

    Without a scorer, enumerate all compatible combinations under a hard cap.
    Scores are cached because unchanged beam states recur at subsequent steps.
    """
    if beam_size < 1 or batch_size < 1 or max_candidates < 1:
        raise ValueError("beam_size, batch_size and max_candidates must be positive")
    tokens = source.split()
    beam = [()]
    cache = {}

    def rank(states):
        missing = list(dict.fromkeys(render(tokens, s) for s in states
                                     if render(tokens, s) not in cache))
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            scores = list(scorer.score(source, batch))
            if len(scores) != len(batch) or any(not math.isfinite(float(s)) for s in scores):
                raise ValueError("scorer returned invalid scores or an incorrect score count")
            cache.update(zip(batch, map(float, scores)))
        return sorted(states, key=lambda s: (-cache[render(tokens, s)], len(s), s))[:beam_size]

    for edit in edits:
        expanded = list(beam)
        for state in beam:
            if all(not conflicts(edit, chosen) for chosen in state):
                expanded.append(tuple(sorted((*state, edit))))
        expanded = list(dict.fromkeys(expanded))
        if scorer is None:
            if len(expanded) > max_candidates:
                raise ValueError("dry-run candidate limit exceeded; reduce proposals or increase --max-candidates")
            beam = expanded
        else:
            beam = rank(expanded)
    if scorer is not None:
        beam = rank(beam)
    candidates = []
    seen = set()
    for state in beam:
        sentence = render(tokens, state)
        if sentence not in seen:
            candidates.append({"sentence": sentence, "score": cache.get(sentence),
                               "edits": [asdict(e) for e in state]})
            seen.add(sentence)
    return {"source": source, "scorer": "greco" if scorer else "none",
            "best_sentence": candidates[0]["sentence"] if scorer else None,
            "pooled_edits": [asdict(e) for e in edits], "candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="enumerate edits without model scores")
    parser.add_argument("--greco-repo", default="external/greco")
    parser.add_argument("--checkpoint", default="models/greco/checkpoint.bin")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=4096)
    args = parser.parse_args()
    try:
        if min(args.beam_size, args.batch_size, args.max_candidates) < 1:
            raise ValueError("size arguments must be positive")
        records = json.loads(args.input.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or not records or any(not isinstance(r, dict) for r in records):
            raise ValueError("input must be an object or a nonempty list of objects")
        prepared = [read_record(record) for record in records]
        scorer = None if args.dry_run else GrecoScorer(args.greco_repo, args.checkpoint, args.device)
        results = [combine(source, edits, scorer, args.beam_size, args.batch_size,
                           args.max_candidates) for source, edits in prepared]
        output = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
