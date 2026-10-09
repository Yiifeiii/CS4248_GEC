from pathlib import Path
import tempfile
import unittest

from greco_setup.combine import combine, read_record
from greco_setup.prepare import build, collect, read_system


class PrepareTests(unittest.TestCase):
    def test_build_pools_systems_and_respects_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            top2 = Path(tmp, "bart.top2")
            top2.write_text("He eats apple .\nHe eat an apple .\nI am fine .\nI am fine .\n")
            one = Path(tmp, "gector.pred")
            one.write_text("He eats an apple .\n\n")
            sources = ["He eat apple .", "I is fine ."]
            systems = [read_system(f"bart={top2}", 2), read_system(f"gector={one}", 2)]

        records = build(sources, systems)
        self.assertEqual([p["rank"] for p in records[0]["proposals"]], [1, 2, 1])
        # The empty GECToR line is skipped, BART's proposals remain
        self.assertEqual({p["system"] for p in records[1]["proposals"]}, {"bart"})
        self.assertEqual(len(build(sources, systems, k=1)[0]["proposals"]), 2)
        # eat->eats, +an, and GECToR's adjacent change merged into one edit: eat->"eats an"
        _, edits = read_record(records[0])
        self.assertEqual(len(edits), 3)

    def test_misaligned_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "t5.top5")
            path.write_text("a\nb\nc\n")
            with self.assertRaises(ValueError):
                read_system(f"t5={path}", 2)

    def test_source_fallback_and_collect(self):
        records = build(["Fine ."], [("t5", [[""]])])
        self.assertEqual(records[0]["proposals"][0]["system"], "source")
        result = combine(*read_record(records[0]))
        self.assertEqual(collect([result]), ["Fine ."])


if __name__ == "__main__":
    unittest.main()
