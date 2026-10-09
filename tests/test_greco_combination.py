import json
from pathlib import Path
import subprocess
import sys
import unittest

from greco_setup.combine import Edit, combine, conflicts, read_record, render


class CombinationTests(unittest.TestCase):
    def test_replacement_insertion_deletion_use_source_offsets(self):
        self.assertEqual(render("He eat red apple .".split(), [
            Edit(1, 2, ("eats",)), Edit(2, 3, ()), Edit(3, 3, ("an",))
        ]), "He eats an apple .")

    def test_conflicts_and_boundaries(self):
        replacement = Edit(1, 3, ("new",))
        self.assertTrue(conflicts(replacement, Edit(2, 2, ("a",))))
        self.assertTrue(conflicts(replacement, Edit(2, 4, ())))
        self.assertFalse(conflicts(replacement, Edit(1, 1, ("a",))))
        self.assertFalse(conflicts(replacement, Edit(3, 3, ("a",))))
        self.assertTrue(conflicts(Edit(2, 2, ("a",)), Edit(2, 2, ("the",))))

    def test_pooling_and_real_search_contract(self):
        source, edits = read_record({"source": "He eat apple .", "proposals": [
            {"sentence": "He eats apple ."}, {"sentence": "He eat an apple ."},
            {"sentence": "He eats apple ."}
        ]})
        self.assertEqual(len(edits), 2)

        class Scorer:
            def score(self, source, candidates):
                return [1.0 if s == "He eats an apple ." else 0.1 for s in candidates]

        result = combine(source, edits, Scorer(), beam_size=4, batch_size=1)
        self.assertEqual(result["best_sentence"], "He eats an apple .")
        self.assertEqual(len(result["candidates"]), 4)
        self.assertIsNone(combine(source, edits)["best_sentence"])

    def test_conflicting_alternatives_never_combine(self):
        result = combine("He eat .", [Edit(1, 2, ("eats",)), Edit(1, 2, ("ate",))])
        self.assertEqual({c["sentence"] for c in result["candidates"]},
                         {"He eat .", "He eats .", "He ate ."})

    def test_unchanged_and_invalid_scores(self):
        class Scorer:
            def score(self, source, candidates):
                return [0.9 if c == source else 0.1 for c in candidates]
        self.assertEqual(combine("He eats .", [Edit(1, 2, ("eat",))], Scorer())["best_sentence"],
                         "He eats .")
        self.assertEqual(combine("Fine .", [], Scorer())["best_sentence"], "Fine .")
        class BadScorer:
            def score(self, source, candidates):
                return [float("nan")] * len(candidates)
        with self.assertRaises(ValueError):
            combine("Fine .", [], BadScorer())

    def test_validation_and_candidate_cap(self):
        with self.assertRaises(ValueError):
            read_record({"source": "Hi .", "proposals": [{"edits": [
                {"start": 0, "end": 3, "replacement": "Hello"}]}]})
        with self.assertRaises(ValueError):
            combine("a b", [Edit(0, 1, ("x",)), Edit(1, 2, ("y",))], max_candidates=2)

    def test_example_cli(self):
        root = Path(__file__).resolve().parents[1]
        run = subprocess.run([sys.executable, "-m", "greco_setup.combine", "--input",
                              "greco_setup/examples/manual.json", "--dry-run"],
                             cwd=root, capture_output=True, text=True, check=True)
        results = json.loads(run.stdout)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["best_sentence"] is None for r in results))
        self.assertIn("He eats an apple .", {c["sentence"] for c in results[1]["candidates"]})


if __name__ == "__main__":
    unittest.main()
