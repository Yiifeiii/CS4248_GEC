"""Reference-edit coverage and diversity of an n-best list; no system combination."""
from difflib import SequenceMatcher
from itertools import combinations


def pool_metrics(texts, candidate_edits, gold_edits, k):
    """Compute diagnostics for the first k available unique candidates per source.

    Gold coverage unions edits across candidates. It is not attainable correction
    recall in general: edits may conflict or appear only in otherwise bad outputs.
    """
    if not (len(texts) == len(candidate_edits) == len(gold_edits)) or not texts or k < 1:
        raise ValueError('Aligned nonempty source lists and positive k required')
    covered = total = proposed = full = 0
    count_sum = 0
    distances = []
    for choices, edits, gold in zip(texts, candidate_edits, gold_edits):
        if not choices or len(choices) != len(edits) or len(set(choices)) != len(choices):
            raise ValueError('Each source needs aligned, nonempty, unique candidates')
        chosen = choices[:k]
        pool = set().union(*edits[:k])
        covered += len(pool & gold)
        total += len(gold)
        proposed += len(pool)
        full += len(chosen) == k
        count_sum += len(chosen)
        distances.extend(1 - SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()
                         for a, b in combinations(chosen, 2))
    return dict(k=k, sources=len(texts), mean_unique_candidates=count_sum/len(texts),
                sources_with_full_k=full, gold_edits_covered=covered, gold_edits_total=total,
                gold_edit_coverage=covered/total if total else None, pooled_candidate_edits=proposed,
                mean_pairwise_token_dissimilarity=sum(distances)/len(distances) if distances else None)
