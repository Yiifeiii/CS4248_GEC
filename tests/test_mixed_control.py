from collections import Counter
import unittest
import tempfile
import hashlib
import json
from pathlib import Path
from training.build_mixed_control import mixed_pairs, build


class MixedControlTests(unittest.TestCase):
    def test_exact_exposure_and_overlap_is_preserved(self):
        a, b, c = ('a', 'A'), ('b', 'B'), ('c', 'c')
        pairs = mixed_pairs([a, b], [b, c])
        self.assertEqual(Counter(pairs), {a: 3, b: 8, c: 5})
        self.assertEqual(len(pairs), 16)

    def test_deterministic_mixing_without_stage_boundary(self):
        a, b = ('stage1', 'x'), ('stage2', 'y')
        first = mixed_pairs([a]*10, [b]*10)
        self.assertEqual(first, mixed_pairs([a]*10, [b]*10))
        self.assertNotEqual(first, mixed_pairs([a]*10, [b]*10, seed=43))
        self.assertIn(a, first[:20]); self.assertIn(b, first[:20])

    def test_build_checks_hashes_and_never_needs_dev_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {}
            for name in ('stage1', 'stage2'):
                hashes = {}
                for suffix in ('src', 'tgt'):
                    p = root / f'{name}.{suffix}'
                    p.write_text(name + '\n')
                    hashes[suffix] = hashlib.sha256(p.read_bytes()).hexdigest()
                manifest[name] = {'pairs': 1, 'sha256': hashes}
            (root/'manifest.json').write_text(json.dumps(manifest))
            record = build(root, root/'mixed/train')
            self.assertEqual(record['pairs'], 8)
            self.assertEqual(len((root/'mixed/train.src').read_text().splitlines()), 8)
            (root/'stage2.src').write_text('tampered\n')
            with self.assertRaises(ValueError): build(root, root/'bad/train')


if __name__ == '__main__': unittest.main()
