import json
from pathlib import Path
import tempfile
import unittest

from greco_setup.combine import CachedScorer, Edit, combine, edit_support, read_record, vote

SOURCE = "a b c d e"


def record(bart, t5=(), gector=None):
    """bart/t5: sentences by rank; gector: one sentence or None."""
    proposals = [{"system": "bart", "rank": r, "sentence": s} for r, s in enumerate(bart, 1)]
    proposals += [{"system": "t5", "rank": r, "sentence": s} for r, s in enumerate(t5, 1)]
    if gector is not None:
        proposals.append({"system": "gector", "rank": 1, "sentence": gector})
    return {"source": SOURCE, "proposals": proposals}


# e1 is in every BART beam, e2 only in beam 1, e3 only in beam 5 (the report example)
E1, E2, E3 = Edit(0, 1, ("A",)), Edit(2, 3, ("C",)), Edit(4, 5, ("E",))
FIVE = record(["A b C d e", "A b c d e", "A b c d e", "A b c d e", "A b c d E"],
              ["a b c d e"] * 5, "a b c d e")


class SupportTests(unittest.TestCase):
    def test_hypothesis_counts_every_beam(self):
        support, baseline = edit_support(FIVE, "hypothesis")
        self.assertAlmostEqual(support[E1], 5 / 11)
        self.assertAlmostEqual(support[E2], 1 / 11)
        self.assertAlmostEqual(baseline, 1 / 11)

    def test_system_votes_once_per_system(self):
        support, baseline = edit_support(FIVE, "system")
        for e in (E1, E2, E3):
            self.assertAlmostEqual(support[e], 1 / 3)
        self.assertAlmostEqual(baseline, 1 / 3)

    def test_rank_uses_best_beam(self):
        support, _ = edit_support(FIVE, "rank")
        self.assertAlmostEqual(support[E1], 1 / 3)
        self.assertAlmostEqual(support[E2], 1 / 3)
        self.assertAlmostEqual(support[E3], (1 / 5) / 3)

    def test_rank_vote_of_top_beam_does_not_depend_on_k(self):
        top1 = record(["A b C d e"], ["a b c d e"], "a b c d e")
        self.assertAlmostEqual(edit_support(top1, "rank")[0][E2], edit_support(FIVE, "rank")[0][E2])

    def test_all_modes_agree_at_k1(self):
        top1 = record(["A b c d e"], ["A b C d e"], "a b c d e")
        results = [edit_support(top1, m) for m in ("hypothesis", "system", "rank")]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_agreement_across_systems_adds_up(self):
        support, _ = edit_support(record(["A b c d e"], ["A b c d e"], "A b c d e"), "system")
        self.assertAlmostEqual(support[E1], 1.0)

    def test_vote_averages_with_baseline(self):
        support = {E1: 1.0, E2: 0.0}
        self.assertAlmostEqual(vote((), support, 0.25), 0.25)
        self.assertAlmostEqual(vote((E1, E2), support, 0.25), (0.25 + 1.0) / 3)


class VotingSearchTests(unittest.TestCase):
    class Scorer:
        """Prefers the lone edit E2 slightly over the agreed edit E1."""
        def score(self, source, candidates):
            return [{"a b C d e": 0.95, "A b c d e": 0.94}.get(c, 0.5) for c in candidates]

    def test_vote_changes_the_choice(self):
        rec = record(["a b C d e"], ["A b c d e"], "A b c d e")
        source, edits = read_record(rec)
        support = edit_support(rec, "system")
        no_vote = combine(source, edits, self.Scorer(), beam_size=4, batch_size=2)
        with_vote = combine(source, edits, self.Scorer(), beam_size=4, batch_size=2,
                            support=support, vote_coef=0.4)
        self.assertEqual(no_vote["best_sentence"], "a b C d e")
        self.assertEqual(with_vote["best_sentence"], "A b c d e")
        self.assertIn("vote", with_vote["candidates"][0])

    def test_zero_coef_matches_plain_search(self):
        source, edits = read_record(FIVE)
        plain = combine(source, edits, self.Scorer(), beam_size=4, batch_size=2)
        zero = combine(source, edits, self.Scorer(), beam_size=4, batch_size=2,
                       support=edit_support(FIVE, "rank"), vote_coef=0.0)
        self.assertEqual(plain, zero)

    def test_voting_needs_support(self):
        source, edits = read_record(FIVE)
        with self.assertRaises(ValueError):
            combine(source, edits, self.Scorer(), vote_coef=0.4)


class CachedScorerTests(unittest.TestCase):
    def test_scores_are_reused_across_instances(self):
        calls = []

        class Scorer:
            def score(self, source, candidates):
                calls.append(list(candidates))
                return [len(c) / 10 for c in candidates]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "cache.jsonl")
            first = CachedScorer(Scorer(), path).score("s", ["x", "yy"])
            second = CachedScorer(Scorer(), path).score("s", ["yy", "x", "zzz"])
            self.assertEqual(first, [0.1, 0.2])
            self.assertEqual(second, [0.2, 0.1, 0.3])
            self.assertEqual(calls, [["x", "yy"], ["zzz"]])
            self.assertEqual(len(path.read_text().splitlines()), 3)
            self.assertEqual(json.loads(path.read_text().splitlines()[0])["h"], "x")

    def test_broken_cache_lines_are_skipped(self):
        # Lines left by two jobs appending at once: invalid JSON, or valid JSON missing fields
        class Scorer:
            def score(self, source, candidates):
                return [0.5] * len(candidates)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "cache.jsonl")
            path.write_text('{"s": "s", "h": "x", "q": 0.9}\n{"s": "s", "h"\n'
                            '{"s": "s", "q": 0.1}\n42\n{"s": "s", "h": "y", "q": "nan"}\n')
            scorer = CachedScorer(Scorer(), path)
            self.assertEqual(scorer.cache, {("s", "x"): 0.9})
            self.assertEqual(scorer.score("s", ["x", "y"]), [0.9, 0.5])


if __name__ == "__main__":
    unittest.main()
