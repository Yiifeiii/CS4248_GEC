"""Tests for the GRECO + ESC pipeline.

    python3 -m unittest greco_esc.test_greco_esc

The end-to-end test trains upstream ESC on a toy pool and feeds its edit scores to upstream
GRECO's combiner, with a stand-in for the DeBERTa scorer. It needs external/esc and
external/greco (greco_esc/setup.sh, greco_setup/setup.sh) and torch, scikit-learn and
transformers, and is skipped without them.
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from greco_esc import evaluate, prepare, settings

REPO = Path(__file__).resolve().parent.parent


class SettingsTests(unittest.TestCase):
    def test_inputs_follow_the_mini_experiment_order(self):
        self.assertEqual(settings.inputs(1), ['gector-1', 'bart-1', 't5-1'])
        self.assertEqual(settings.inputs(3), ['gector-1', 'bart-1', 't5-1', 'bart-2', 't5-2', 'bart-3', 't5-3'])
        self.assertEqual(len(settings.inputs(5)), 11)

    def test_every_k_has_the_four_combinations(self):
        runs = settings.runs()
        self.assertEqual(len(runs), 4 * len(settings.KS))
        self.assertEqual(len({(name, k) for name, k, _, _ in runs}), len(runs))
        for k in settings.KS:
            self.assertEqual({(esc, vote) for _, kk, vote, esc in runs if kk == k},
                             {(False, 0), (False, settings.VOTE[k]), (True, 0), (True, settings.VOTE[k])})

    def test_every_k_has_a_tuned_vote(self):
        self.assertEqual(set(settings.VOTE), set(settings.KS))

    def test_the_shell_listing_matches(self):
        out = subprocess.run([sys.executable, '-m', 'greco_esc.settings', 'runs'], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split('\n')
        self.assertIn('greco+esc.vote0.1 3 0.1 1', out)
        self.assertIn('greco.vote0 1 0 0', out)


class ClipTests(unittest.TestCase):
    def test_probabilities_of_0_and_1_are_clipped(self):
        out = prepare.clip(['[[0, 1, "a", 1.0], [2, 2, "", 0.0], [3, 4, "b", 0.5]]', '[]'], 2)
        self.assertEqual([e[3] for e in out[0]], [1 - prepare.EPS, prepare.EPS, 0.5])
        self.assertEqual(out[1], [])

    def test_a_missing_sentence_fails(self):
        with self.assertRaises(ValueError):
            prepare.clip(['[]'], 2)


def write_m2(path, sources, edits):
    """M2 blocks for sources, with edits[i] = [(start, end, type, correction), ...]."""
    blocks = []
    for src, sent_edits in zip(sources, edits):
        lines = [f'S {src}'] + [f'A {s} {e}|||{t}|||{c}|||REQUIRED|||-NONE-|||0' for s, e, t, c in sent_edits]
        if not sent_edits:
            lines.append('A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0')
        blocks.append('\n'.join(lines))
    path.write_text('\n\n'.join(blocks) + '\n\n', encoding='utf-8')


def toy_pool(tmp, k=1):
    """A pool in which BART proposes the gold verb fix, T5 a wrong article and GECToR both."""
    data, root = Path(tmp, 'data'), Path(tmp, 'runs')
    data.mkdir()
    sources = [f'He go to school number {i} .' for i in range(20)]
    verb, article = (1, 2, 'R:VERB:SVA', 'goes'), (2, 2, 'M:DET', 'the')
    systems = {'gector-1': [[verb, article]] * 20, 'bart-1': [[verb]] * 20, 't5-1': [[article]] * 20}
    for split in (settings.TRAIN_SPLIT, settings.TEST_SPLIT):
        (data / f'{split}.src').write_text('\n'.join(sources) + '\n', encoding='utf-8')
        (data / f'{split}.tgt').write_text('\n'.join(s.replace(' go ', ' goes ') for s in sources) + '\n',
                                           encoding='utf-8')
        write_m2(data / f'{split}.m2', sources, [[verb]] * 20)
        for sub in ('text', 'm2'):
            (root / split / sub).mkdir(parents=True)
        for name, edits in systems.items():
            (root / split / 'text' / f'{name}.txt').write_text('x\n' * 20, encoding='utf-8')
            write_m2(root / split / 'm2' / f'{name}.m2', sources, edits)
    return data, root


class EscFolderTests(unittest.TestCase):
    def test_each_k_links_its_inputs_and_only_training_has_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, root = toy_pool(tmp)
            with mock.patch.object(settings, 'DATA', data), mock.patch.object(settings, 'ROOT', root), \
                    mock.patch.object(settings, 'KS', (1,)), contextlib.redirect_stdout(io.StringIO()):
                prepare.build_esc()
            train, test = prepare_dirs(root, 1)
            self.assertEqual(sorted(p.name for p in (train / 'text').iterdir()),
                             ['bart-1.txt', 'gector-1.txt', 'source.txt', 't5-1.txt', 'target.txt'])
            self.assertEqual(sorted(p.name for p in (test / 'm2').iterdir()), ['bart-1.m2', 'gector-1.m2', 't5-1.m2'])
            self.assertEqual((train / 'm2' / 'target.m2').read_text(encoding='utf-8'),
                             (data / f'{settings.TRAIN_SPLIT}.m2').read_text(encoding='utf-8'))


def prepare_dirs(root, k):
    with mock.patch.object(settings, 'ROOT', root):
        return prepare.esc_dir(k, settings.TRAIN_SPLIT), prepare.esc_dir(k, settings.TEST_SPLIT)


def has(*modules):
    return all(importlib.util.find_spec(m) for m in modules)


@unittest.skipUnless((REPO / 'external/esc/run.py').exists() and (REPO / 'external/greco/run_combination.py').exists()
                     and has('torch', 'sklearn', 'transformers'), 'needs external/esc, external/greco and torch')
class EndToEndTests(unittest.TestCase):
    """Upstream ESC trained on the toy pool, its scores read by upstream GRECO's combiner."""

    def test_esc_scores_reach_every_greco_edit_and_pick_the_gold_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, root = toy_pool(tmp)
            patches = [mock.patch.object(settings, 'DATA', data), mock.patch.object(settings, 'ROOT', root),
                       mock.patch.object(settings, 'KS', (1,))]
            for p in patches:
                p.start()
            self.addCleanup(lambda: [p.stop() for p in patches])
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.build_esc()
            train, test = prepare.esc_dir(1, settings.TRAIN_SPLIT), prepare.esc_dir(1, settings.TEST_SPLIT)
            model = root / 'esc' / 'k1' / 'model'
            esc = [sys.executable, str(REPO / 'external/esc/run.py')]
            quiet = {'check': True, 'capture_output': True}
            subprocess.run(esc + ['--train', '--data_dir', str(train / 'text'), '--m2_dir', str(train / 'm2'),
                                  '--model_path', str(model), '--vocab_path', str(model / 'vocab.idx')], **quiet)
            out = root / settings.TEST_SPLIT / 'k1'
            out.mkdir(parents=True)
            subprocess.run(esc + ['--score', '--data_dir', str(test / 'text'), '--m2_dir', str(test / 'm2'),
                                  '--model_path', str(model / 'model.pt'), '--vocab_path', str(model / 'vocab.idx'),
                                  '--output_path', str(out / 'esc.scores.raw.jsonl')], **quiet)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.fix_scores(1)
            scores = [json.loads(line) for line in (out / 'esc.scores.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(scores), 20)
            by_edit = {(s, e, c): p for s, e, c, p in scores[0]}
            # The gold verb fix is proposed by two systems and always right; the article never is
            self.assertGreater(by_edit[(1, 2, 'goes')], 0.5)
            self.assertLess(by_edit[(2, 2, 'the')], 0.5)

            # Upstream's combiner with these scores, ranking candidates by a constant text score
            # so that only ESC's scores decide; it warns for any edit without a score
            sys.path.insert(0, str(REPO / 'external/greco'))
            self.addCleanup(sys.path.remove, str(REPO / 'external/greco'))
            cwd = os.getcwd()
            os.chdir(REPO / 'external/greco')
            self.addCleanup(os.chdir, cwd)
            import run_combination
            from utils.data import SysCombDataset
            inputs = [str(root / settings.TEST_SPLIT / 'm2' / f'{n}.m2') for n in settings.inputs(1)]
            dataset = SysCombDataset(inputs, False, str(out / 'esc.scores.jsonl'))

            class Constant:
                def score(self, source, hyps):
                    return [0.5] * len(hyps)

            log = io.StringIO()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(io.StringIO()):
                result = run_combination.beam_search(dataset, Constant(), 16, 16, edit_score_w=settings.SCORE_RATIO)
            self.assertNotIn('not found', log.getvalue())
            self.assertEqual(result[0], 'He goes to school number 0 .')


class EvaluateTests(unittest.TestCase):
    def test_the_table_has_one_line_per_k_and_marks_missing_outputs(self):
        rows = [{'system': 'greco', 'k': 1, 'vote_coef': '0', 'esc': 'no',
                 'errant_f0_5': 57.6, 'errant_p': 65.0, 'errant_r': 39.5}]
        text = evaluate.table(rows)
        self.assertIn('| 1 | 0.4 | 57.6 (65.0 / 39.5) | missing |', text)
        self.assertEqual(sum(line.startswith('| ') and line[2].isdigit() for line in text.splitlines()),
                         len(settings.KS))

    def test_every_run_has_an_output_path(self):
        paths = [p for *_, p in evaluate.outputs()]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(paths), 3 + len(settings.KS) + len(settings.runs()))


if __name__ == '__main__':
    unittest.main()
