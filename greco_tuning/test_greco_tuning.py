from pathlib import Path
import json
import tempfile
import unittest

from greco_tuning.build_labels import build, f05, hyp_record, token_labels
from greco_tuning.prepare_stage1 import prepare

try:
    import errant
    ANNOTATOR = errant.load('en')
except (ImportError, OSError):
    ANNOTATOR = None


class Edit:
    """Stands in for an ERRANT edit where only the span and replacement matter."""
    def __init__(self, o_start, o_end, c_str):
        self.o_start, self.o_end, self.c_str = o_start, o_end, c_str


class F05Tests(unittest.TestCase):
    def test_perfect_hypothesis_scores_one(self):
        gold = [Edit(1, 2, 'eats')]
        self.assertEqual(f05(gold, gold), 1.0)

    def test_no_edits_on_either_side_scores_one(self):
        self.assertEqual(f05([], []), 1.0)

    def test_missing_every_gold_edit_scores_zero(self):
        self.assertEqual(f05([], [Edit(1, 2, 'eats')]), 0.0)

    def test_only_wrong_edits_scores_zero(self):
        self.assertEqual(f05([Edit(0, 1, 'She')], [Edit(1, 2, 'eats')]), 0.0)

    def test_partial_match_uses_f_half(self):
        # precision 1/2, recall 1/1: F0.5 = 1.25 * 0.5 * 1 / (0.25 * 0.5 + 1)
        hyp = [Edit(1, 2, 'eats'), Edit(0, 1, 'She')]
        gold = [Edit(1, 2, 'eats')]
        self.assertAlmostEqual(f05(hyp, gold), 1.25 * 0.5 / 1.125)


@unittest.skipIf(ANNOTATOR is None, 'errant and en_core_web_sm are not installed')
class LabelTests(unittest.TestCase):
    def labels_for(self, source, gold, hyp):
        src_doc = ANNOTATOR.parse(source, tokenise=False)
        gold_doc = ANNOTATOR.parse(gold, tokenise=False)
        hyp_doc = ANNOTATOR.parse(hyp, tokenise=False)
        gold_edits = ANNOTATOR.annotate(src_doc, gold_doc)
        return hyp_record(ANNOTATOR, src_doc, gold_doc, hyp_doc,
                          source.split(), hyp.split(), gold_edits)

    def test_wrong_word_is_zero_and_source_error_is_zero(self):
        record = self.labels_for('He eat apple .', 'He eats apple .', 'He eat apple .')
        self.assertEqual(record['labels'][1], [1, 0, 1, 1])
        self.assertEqual(record['labels'][2], [1, 0, 1, 1])
        self.assertEqual(record['masks'], [[False, True, False, False], [False, True, False, False]])
        self.assertEqual(record['gap'], [1, 1, 1, 1, 1])
        self.assertEqual(record['labels'][0], [0.0])

    def test_perfect_hypothesis_is_all_ones(self):
        record = self.labels_for('He eat apple .', 'He eats apple .', 'He eats apple .')
        self.assertEqual(record['labels'][0], [1.0])
        self.assertEqual(record['labels'][2], [1, 1, 1, 1])
        self.assertEqual(record['gap'], [1, 1, 1, 1, 1])

    def test_missing_word_is_a_zero_gap_at_its_position(self):
        # The gold inserts "to" before "school"; the hypothesis keeps the source, so the gap
        # before "school" (index 2 of the hypothesis gaps) needs an insertion.
        record = self.labels_for('He went school .', 'He went to school .', 'He went school .')
        self.assertEqual(record['gap'], [1, 1, 0, 1, 1])
        self.assertEqual(record['gap_mask'], [False, False, True, False, False])
        self.assertEqual(record['labels'][2], [1, 1, 1, 1])
        self.assertEqual(record['labels'][1], [1, 1, 1, 1])
        self.assertEqual(record['labels'][0], [0.0])


@unittest.skipIf(ANNOTATOR is None, 'errant and en_core_web_sm are not installed')
class BuildTests(unittest.TestCase):
    def test_group_has_loader_layout_and_drops_source_and_gold(self):
        source = 'He eat apple .'
        gold = 'He eats apple .'
        candidates = [
            'He eats apple .',          # equals gold: dropped
            'He eat apple .',           # equals source: dropped
            'She eats apple .',
            'He eats an apple .',
            'He eats apples .',
            'He eats the apple .',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'src').write_text(source + '\n', encoding='utf-8')
            (tmp / 'gold').write_text(gold + '\n', encoding='utf-8')
            (tmp / 'sys.tgt').write_text('\n'.join(candidates) + '\n', encoding='utf-8')
            out = tmp / 'out.jsonl'
            stats = build(tmp / 'src', tmp / 'gold', [f'sys={tmp / "sys.tgt"}'], 6, out, ANNOTATOR)

            lines = out.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), 1)
            group = json.loads(lines[0])
            self.assertEqual(len(group), 3)
            self.assertEqual(stats['groups'], 1)
            for record in group:
                n_src, n_hyp = len(record['source']), len(record['hyp'])
                self.assertEqual(record['source'], source.split())
                self.assertEqual(len(record['labels'][0]), 1)
                self.assertEqual(len(record['labels'][1]), n_src)
                self.assertEqual(len(record['labels'][2]), n_hyp)
                self.assertEqual(len(record['gap']), n_hyp + 1)
                self.assertEqual(len(record['gap_mask']), n_hyp + 1)
                self.assertEqual(len(record['masks'][0]), n_src)
                self.assertEqual(len(record['masks'][1]), n_hyp)
                self.assertNotIn(record['hyp'], (source.split(), gold.split()))

    def test_source_with_too_few_candidates_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'src').write_text('He eat apple .\n', encoding='utf-8')
            (tmp / 'gold').write_text('He eats apple .\n', encoding='utf-8')
            (tmp / 'sys.tgt').write_text('He eats apple .\nHe eat apple .\nShe eats apple .\n',
                                         encoding='utf-8')
            out = tmp / 'out.jsonl'
            stats = build(tmp / 'src', tmp / 'gold', [f'sys={tmp / "sys.tgt"}'], 3, out, ANNOTATOR)
            self.assertEqual(out.read_text(encoding='utf-8'), '')
            self.assertEqual(stats['skipped_few_candidates'], 1)


class PrepareStage1Tests(unittest.TestCase):
    def test_removes_wi_and_samples_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, out = tmp / 'data', tmp / 'out'
            data.mkdir()
            rows = [('a b .', 'a c .', 'lang8'), ('d e .', 'd f .', 'wi.A'),
                    ('g h .', 'g i .', 'nucle'), ('j k .', 'j l .', 'wi.C'),
                    ('m n .', 'm o .', 'fce.train')]
            for col, idx in (('src', 0), ('tgt', 1), ('corpus', 2)):
                (data / f'stage1.{col}').write_text(''.join(r[idx] + '\n' for r in rows),
                                                    encoding='utf-8')

            manifest = prepare(data, out, sample=2, seed=4248)
            self.assertEqual(manifest['clean']['pairs'], 3)
            clean_src = (out / 'stage1.clean.src').read_text(encoding='utf-8').splitlines()
            self.assertEqual(clean_src, ['a b .', 'g h .', 'm n .'])
            self.assertEqual(len((out / 'stage1.sample.src').read_text(encoding='utf-8').splitlines()), 2)

            first = (out / 'stage1.sample.src').read_text(encoding='utf-8')
            prepare(data, out, sample=2, seed=4248)
            self.assertEqual((out / 'stage1.sample.src').read_text(encoding='utf-8'), first)


if __name__ == '__main__':
    unittest.main()
