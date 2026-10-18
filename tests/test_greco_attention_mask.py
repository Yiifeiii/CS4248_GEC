from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

from greco_setup.combine import GrecoScorer, attention_mask_patched

REPO = Path("external/greco")
CHECKPOINT = Path("models/greco/checkpoint.bin")
# GRECO's pinned dependencies only install in the Python 3.10 GRECO environment
HAVE_GRECO = sys.version_info[:2] == (3, 10) and (REPO / "models.py").is_file() \
    and CHECKPOINT.is_file() and importlib.util.find_spec("torch") is not None


class AttentionMaskGuardTests(unittest.TestCase):
    def test_unpatched_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "models.py").write_text(
                "output = self.__call__(texts['input_ids'].to(self.device))\n")
            Path(tmp, "checkpoint.bin").write_bytes(b"x")
            self.assertFalse(attention_mask_patched(tmp))
            with self.assertRaisesRegex(ValueError, "attention-mask"):
                GrecoScorer(tmp, Path(tmp, "checkpoint.bin"))

    def test_patch_file_matches_guard(self):
        patch = Path("greco_setup/greco-attention-mask.patch").read_text(encoding="utf-8")
        self.assertIn("+                                   attention_mask=texts['attention_mask']", patch)


@unittest.skipUnless(HAVE_GRECO, "needs the GRECO environment (.venv-greco)")
class PaddingInvarianceTests(unittest.TestCase):
    def test_score_does_not_depend_on_batch(self):
        scorer = GrecoScorer(REPO, CHECKPOINT)
        source = "She go to school yesterday ."
        short = ["She went to school yesterday .", "She go to school yesterday ."]
        long = ("She went to the school that is very far away from her house yesterday "
                "because she wanted to learn many new things about grammar .")
        alone = scorer.score(source, short)
        batched = scorer.score(source, short + [long])[:2]
        for a, b in zip(alone, batched):
            self.assertAlmostEqual(a, b, places=4)


if __name__ == "__main__":
    unittest.main()
