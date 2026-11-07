"""Run with python -m greco_setup.combine --input PATH [--dry-run].

Candidates are ranked by log Q + vote_coef * log V (Qorib & Ng, 2023, Eq. 10), where
Q is the GRECO score and V the average support of the candidate's edits. Support
(--vote-mode) is the share of hypotheses ("hypothesis", as in the official code),
of systems ("system", the paper's definition) or of systems weighted by the best
rank at which each proposes the edit ("rank"). vote_coef 0 ranks by Q alone.
"""

import argparse
from dataclasses import asdict, dataclass
from collections import defaultdict
from difflib import SequenceMatcher
import importlib.util
import json
import math
from pathlib import Path

VOTE_MODES = ("hypothesis", "system", "rank")


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
        pool.update(proposal_edits(tokens, proposal))
    return source, sorted(pool)


def proposal_edits(tokens, proposal):
    """The edits one proposal makes to the source tokens, as a sorted list."""
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
    return edits


def edit_support(record, mode):
    """Support of each pooled edit in [0, 1], and the support of the unchanged sentence.

    hypothesis: share of proposals containing the edit (the official run_combination.py,
                given one input per hypothesis).
    system:     share of systems with at least one proposal containing it (the paper's
                count(e) / c), so a system repeating an edit across its beams votes once.
    rank:       each system votes 1 / (best rank of a proposal containing the edit),
                divided by the number of systems. A rank-1 edit keeps the same vote at
                any k; edits found only in lower beams count less.
    The unchanged sentence gets one voter's share (1 / hypotheses or 1 / systems), as the
    official code does, so candidates without edits still have a finite log V.
    """
    if mode not in VOTE_MODES:
        raise ValueError(f"vote mode must be one of {VOTE_MODES}")
    source, _ = read_record(record)
    tokens = source.split()
    proposals = record["proposals"]
    systems = sorted({str(p.get("system", "")) for p in proposals})
    hyps = defaultdict(int)
    best_rank = defaultdict(dict)
    for proposal in proposals:
        system, rank = str(proposal.get("system", "")), proposal.get("rank", 1)
        if type(rank) is not int or rank < 1:
            raise ValueError("proposal rank must be a positive integer")
        for edit in proposal_edits(tokens, proposal):
            hyps[edit] += 1
            best_rank[edit][system] = min(rank, best_rank[edit].get(system, rank))
    if mode == "hypothesis":
        n = len(proposals)
        return {e: hyps[e] / n for e in hyps}, 1 / n
    c = len(systems)
    if mode == "system":
        return {e: len(best_rank[e]) / c for e in hyps}, 1 / c
    return {e: sum(1 / r for r in best_rank[e].values()) / c for e in hyps}, 1 / c


def vote(state, support, baseline):
    """Average support V of a candidate's edits, smoothed with the baseline voter."""
    return (baseline + sum(support[e] for e in state)) / (1 + len(state))


def attention_mask_patched(repo):
    """True if upstream GRECO.score() passes the attention mask (greco-attention-mask.patch).

    Without it, padded positions are attended to and a candidate's score depends on
    the other candidates in its batch.
    """
    source = (Path(repo) / "models.py").read_text(encoding="utf-8")
    return "attention_mask=texts['attention_mask']" in source


class GrecoScorer:
    """Load the upstream class and released state dict, without copying its code."""

    def __init__(self, repo, checkpoint, device="cpu"):
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        if not (repo / "models.py").is_file() or not checkpoint.is_file():
            raise ValueError("GRECO source/checkpoint missing; follow greco_setup/README.md")
        if not attention_mask_patched(repo):
            raise ValueError("GRECO models.py lacks the attention-mask fix; rerun greco_setup/setup.sh")
        import torch
        spec = importlib.util.spec_from_file_location("cs4248_upstream_greco", repo / "models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.GRECO("microsoft/deberta-v3-large")
        self.model.device = torch.device(device)
        self.model.to(self.model.device)
        state = torch.load(checkpoint, map_location="cpu")
        # Newer transformers no longer register position_ids as a buffer; the pinned 4.23 does
        if "lm.embeddings.position_ids" not in self.model.state_dict():
            state.pop("lm.embeddings.position_ids", None)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def score(self, source, candidates):
        return self.model.score(source, candidates).detach().cpu().reshape(-1).tolist()


class CachedScorer:
    """Wrap a scorer with a persistent cache of GRECO scores (one JSON line per score).

    Scores depend only on (source, candidate), not on the vote settings, so runs with
    different vote modes and coefficients reuse each other's scores. Only valid for
    the attention-mask-patched scorer, whose scores do not depend on the batch.
    """

    def __init__(self, scorer, path):
        self.scorer, self.path, self.cache = scorer, Path(path), {}
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    self.cache[(item["s"], item["h"])] = item["q"]
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def score(self, source, candidates):
        missing = [c for c in dict.fromkeys(candidates) if (source, c) not in self.cache]
        if missing:
            scores = list(self.scorer.score(source, missing))
            if len(scores) != len(missing):
                raise ValueError("scorer returned an incorrect score count")
            with self.path.open("a", encoding="utf-8") as f:
                for c, q in zip(missing, scores):
                    self.cache[(source, c)] = float(q)
                    f.write(json.dumps({"s": source, "h": c, "q": float(q)}, ensure_ascii=False) + "\n")
        return [self.cache[(source, c)] for c in candidates]


def combine(source, edits, scorer=None, beam_size=16, batch_size=8, max_candidates=4096,
            support=None, vote_coef=0.0):
    """Include/skip every pooled edit; score candidates at every beam step.

    Without a scorer, enumerate all compatible combinations under a hard cap.
    Scores are cached because unchanged beam states recur at subsequent steps.
    With vote_coef > 0, support = edit_support(...) adds the voting bias.
    """
    if beam_size < 1 or batch_size < 1 or max_candidates < 1:
        raise ValueError("beam_size, batch_size and max_candidates must be positive")
    if vote_coef < 0 or (vote_coef > 0 and support is None):
        raise ValueError("vote_coef must be >= 0, and voting needs edit support")
    support_of, baseline = support if support is not None else ({}, 1.0)
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
        return sorted(states, key=key)[:beam_size]

    def key(state):
        q = cache[render(tokens, state)]
        if vote_coef > 0:
            q = math.log(max(q, 1e-12)) + vote_coef * math.log(vote(state, support_of, baseline))
        return (-q, len(state), state)

    for edit in edits:
        expanded = list(beam)
        for state in beam:
            if all(not conflicts(edit, chosen) for chosen in state):
                state = tuple(sorted((*state, edit)))
                # Deletions from different hypotheses can combine into an empty sentence
                if render(tokens, state):
                    expanded.append(state)
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
            item = {"sentence": sentence, "score": cache.get(sentence),
                    "edits": [asdict(e) for e in state]}
            if vote_coef > 0:
                item["vote"] = vote(state, support_of, baseline)
            candidates.append(item)
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
    parser.add_argument("--vote-mode", choices=VOTE_MODES, default="system")
    parser.add_argument("--vote-coef", type=float, default=0.0,
                        help="voting bias weight (alpha); 0 ranks by the GRECO score alone")
    parser.add_argument("--score-cache", type=Path, help="JSON-lines cache of GRECO scores")
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
        supports = [edit_support(r, args.vote_mode) if args.vote_coef > 0 else None for r in records]
        scorer = None if args.dry_run else GrecoScorer(args.greco_repo, args.checkpoint, args.device)
        if scorer is not None and args.score_cache:
            scorer = CachedScorer(scorer, args.score_cache)
        results = [combine(source, edits, scorer, args.beam_size, args.batch_size,
                           args.max_candidates, support, args.vote_coef)
                   for (source, edits), support in zip(prepared, supports)]
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
