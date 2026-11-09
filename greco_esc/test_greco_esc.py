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

from greco_esc import combine, evaluate, prepare, settings
from greco_setup.prepare import read_lines

REPO = Path(__file__).resolve().parent.parent


class SettingsTests(unittest.TestCase):
    def test_inputs_follow_the_mini_experiment_order(self):
        self.assertEqual(settings.inputs(1), ['gector-1', 'bart-1', 't5-1'])
        self.assertEqual(settings.inputs(3), ['gector-1', 'bart-1', 't5-1', 'bart-2', 't5-2', 'bart-3', 't5-3'])
        self.assertEqual(len(settings.inputs(5)), 11)

    def test_every_k_has_each_combination(self):
        runs = settings.runs()
        self.assertEqual(len({(name, k) for name, k, *_ in runs}), len(runs))
        for k in settings.KS:
            modes = ('hypothesis',) if k == 1 else settings.VOTE_MODES
            want = {(esc, 0, 'hypothesis') for esc in (False, True)} | \
                   {(esc, settings.VOTES[m][k], m) for esc in (False, True) for m in modes}
            self.assertEqual({(esc, vote, mode) for _, kk, vote, esc, mode in runs if kk == k}, want)

    def test_the_hypothesis_runs_keep_their_names(self):
        # finished runs from before the vote modes were added are reused by name
        names = {(name, k) for name, k, *_ in settings.runs()}
        self.assertIn(('greco+esc.vote0.1', 3), names)
        self.assertIn(('greco.vote0.1', 1), names)

    def test_every_mode_and_k_has_a_tuned_vote(self):
        for mode in settings.VOTE_MODES:
            self.assertEqual(set(settings.VOTES[mode]), set(settings.KS))

    def test_the_shell_listing_matches(self):
        out = subprocess.run([sys.executable, '-m', 'greco_esc.settings', 'runs'], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split('\n')
        self.assertIn('greco+esc.vote0.1 3 0.1 1 hypothesis', out)
        self.assertIn('greco.vote0 1 0 0 hypothesis', out)
        self.assertIn('greco+esc.rank0.6 5 0.6 1 rank', out)
        self.assertIn('greco.rank0.2 2 0.2 0 rank', out)
        self.assertIn('greco.system0.1 2 0.1 0 system', out)


class VoteModeTests(unittest.TestCase):
    """combine.support, against the greco-voting branch's definitions."""
    LABELS = [('gector', 1), ('bart', 1), ('t5', 1), ('bart', 2), ('t5', 2)]
    A, B, C = (0, 1, 'x'), (2, 3, 'y'), (4, 4, 'z')
    # A: everyone; B: bart at ranks 1 and 2 only; C: t5 at rank 2 only
    EDITS = [{A}, {A, B}, {A}, {A, B}, {A, C}]

    def test_hypothesis_counts_inputs(self):
        sup, base = combine.support(self.EDITS, self.LABELS, 'hypothesis')
        self.assertEqual(sup, {self.A: 1.0, self.B: 2 / 5, self.C: 1 / 5})
        self.assertEqual(base, 1 / 5)

    def test_system_counts_a_repeated_edit_once(self):
        sup, base = combine.support(self.EDITS, self.LABELS, 'system')
        self.assertEqual(sup, {self.A: 1.0, self.B: 1 / 3, self.C: 1 / 3})
        self.assertEqual(base, 1 / 3)

    def test_rank_counts_each_system_at_its_best_rank(self):
        sup, base = combine.support(self.EDITS, self.LABELS, 'rank')
        self.assertEqual(sup, {self.A: 1.0, self.B: 1 / 3, self.C: (1 / 2) / 3})
        self.assertEqual(base, 1 / 3)

    def test_the_modes_agree_at_k_1(self):
        labels, edits = self.LABELS[:3], self.EDITS[:3]
        results = [combine.support(edits, labels, m) for m in settings.VOTE_MODES]
        self.assertTrue(all(r == results[0] for r in results))

    def test_input_names_give_system_and_rank(self):
        self.assertEqual(combine.system_rank('/x/m2/bart-2.m2'), ('bart', 2))
        self.assertEqual(combine.system_rank('gector-1.m2'), ('gector', 1))
        with self.assertRaises(ValueError):
            combine.system_rank('bart.m2')

    def test_vote_mode_is_removed_before_upstream_parses(self):
        argv, mode = combine._pop_vote_mode(['x', '--vote_coef', '0.1', '--vote_mode', 'rank', '--data', 'a'])
        self.assertEqual((argv, mode), (['x', '--vote_coef', '0.1', '--data', 'a'], 'rank'))
        self.assertEqual(combine._pop_vote_mode(['x'])[1], 'hypothesis')


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


class CrossFitTests(unittest.TestCase):
    """The reverse fold (ESC trained on dev_test, scoring dev_tune) used for the full dev set."""

    def test_the_reverse_fold_trains_on_dev_test_and_keeps_the_original_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, root = toy_pool(tmp)
            with mock.patch.object(settings, 'DATA', data), mock.patch.object(settings, 'ROOT', root), \
                    mock.patch.object(settings, 'KS', (1,)), contextlib.redirect_stdout(io.StringIO()):
                prepare.build_esc()
                train = prepare.esc_dir(1, settings.TEST_SPLIT, settings.TRAIN_SPLIT)
                test = prepare.esc_dir(1, settings.TRAIN_SPLIT, settings.TRAIN_SPLIT)
            self.assertEqual(train, root / 'esc' / 'k1' / 'for-dev_tune' / 'dev_test')
            self.assertEqual((train / 'm2' / 'target.m2').read_text(encoding='utf-8'),
                             (data / f'{settings.TEST_SPLIT}.m2').read_text(encoding='utf-8'))
            self.assertFalse((test / 'm2' / 'target.m2').exists())
            self.assertTrue((root / 'esc' / 'k1' / 'for-dev_tune' / 'model').is_dir())
            # the dev_test fold is where the first run left it, so its model and outputs are reused
            self.assertTrue((root / 'esc' / 'k1' / settings.TRAIN_SPLIT / 'm2' / 'target.m2').exists())
            self.assertFalse((root / 'esc' / 'k1' / settings.TEST_SPLIT / 'm2' / 'target.m2').exists())

    def test_each_split_lists_the_other_as_esc_training(self):
        out = subprocess.run([sys.executable, '-m', 'greco_esc.settings', 'splits'], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split('\n')
        self.assertEqual(out[:2], ['dev_test dev_tune', 'dev_tune dev_test'])


class FullDevTests(unittest.TestCase):
    """Merging the dev_tune and dev_test halves back into data/dev.src order."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data, self.root = Path(tmp.name, 'data'), Path(tmp.name, 'runs')
        self.data.mkdir()
        # full dev = s0 .. s4; dev_tune holds 1 and 3, dev_test 0, 2 and 4
        halves = {settings.TRAIN_SPLIT: [1, 3], settings.TEST_SPLIT: [0, 2, 4]}
        for half, order in halves.items():
            (self.data / f'{half}.ids').write_text(''.join(f'{i}\n' for i in order), encoding='utf-8')
            (self.data / f'{half}.src').write_text(''.join(f's{i}\n' for i in order), encoding='utf-8')
            write_m2(self.data / f'{half}.m2', [f's{i}' for i in order], [[(0, 1, 'R:NOUN', f'g{i}')] for i in order])
            pred = self.root / half / 'k1' / 'greco.vote0' / 'pred.txt'
            pred.parent.mkdir(parents=True)
            pred.write_text('\n'.join(f'h{i}' for i in order), encoding='utf-8')  # no final newline, as upstream
        patches = [mock.patch.object(settings, 'DATA', self.data), mock.patch.object(settings, 'ROOT', self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_outputs_and_reference_are_merged_in_dev_order(self):
        rel = Path('k1') / 'greco.vote0' / 'pred.txt'
        self.assertEqual(evaluate.load('dev', rel), [f'h{i}' for i in range(5)])
        src, gold = evaluate.reference('dev')
        self.assertEqual(read_lines(src), [f's{i}' for i in range(5)])
        self.assertEqual([b.split('\n')[1].split('|||')[2] for b in evaluate.m2_blocks(gold)],
                         [f'g{i}' for i in range(5)])

    def test_a_missing_half_makes_the_full_dev_output_missing(self):
        (self.root / settings.TRAIN_SPLIT / 'k1' / 'greco.vote0' / 'pred.txt').unlink()
        self.assertIsNone(evaluate.load('dev', Path('k1') / 'greco.vote0' / 'pred.txt'))
        self.assertIsNotNone(evaluate.load(settings.TEST_SPLIT, Path('k1') / 'greco.vote0' / 'pred.txt'))

    def test_a_half_with_the_wrong_length_fails(self):
        with self.assertRaises(ValueError):
            evaluate.merge({settings.TRAIN_SPLIT: ['a'], settings.TEST_SPLIT: ['b', 'c', 'd']})


class TestSetTests(unittest.TestCase):
    """test.sh: every combination, ESC trained on the full dev set, test sets never trained on."""

    def test_the_final_run_is_one_of_the_runs(self):
        name, k = settings.FINAL_RUN
        self.assertIn((name, k), {(n, kk) for n, kk, *_ in settings.runs()})

    def test_test_sets_share_one_esc_model_per_k_outside_the_dev_folders(self):
        self.assertEqual(settings.esc_home(3, 'conll14'), settings.esc_home(3, 'bea19'))
        self.assertEqual(settings.esc_home(3, 'conll14'), settings.ROOT / 'esc' / 'final' / 'k3')
        self.assertEqual(settings.source('conll14'), Path('data/test.conll14.src'))
        self.assertEqual(settings.source('dev_tune'), settings.DATA / 'dev_tune.src')
        self.assertEqual(evaluate.reference('conll14'), settings.TEST_SETS['conll14'])

    def test_the_full_dev_esc_trains_on_both_halves_and_the_test_set_has_no_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, root = toy_pool(tmp)
            # a 3-sentence toy test set with the k = 1 inputs
            test_src = Path(tmp, 'test.src')
            test_src.write_text('A b .\nC d .\nE f .\n', encoding='utf-8')
            for sub in ('text', 'm2'):
                (root / 'toy' / sub).mkdir(parents=True)
            for name in settings.inputs(1):
                (root / 'toy' / 'text' / f'{name}.txt').write_text('x\n' * 3, encoding='utf-8')
                write_m2(root / 'toy' / 'm2' / f'{name}.m2', ['A b .', 'C d .', 'E f .'], [[], [], []])
            patches = [mock.patch.object(settings, 'DATA', data), mock.patch.object(settings, 'ROOT', root),
                       mock.patch.object(settings, 'KS', (1,)),
                       mock.patch.dict(settings.TEST_SETS, {'toy': (test_src, None)})]
            for p in patches:
                p.start()
                self.addCleanup(p.stop)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.build_final_esc(['toy'])
            home = root / 'esc' / 'final' / 'k1'
            # 20 dev_tune + 20 dev_test sentences, in every file ESC reads
            self.assertEqual(len(read_lines(home / 'dev' / 'text' / 'source.txt')), 40)
            self.assertEqual(len(read_lines(home / 'dev' / 'text' / 'bart-1.txt')), 40)
            for name in settings.inputs(1) + ['target']:
                self.assertEqual(len(prepare.m2_sources(home / 'dev' / 'm2' / f'{name}.m2')), 40, name)
            self.assertEqual(sorted(p.name for p in (home / 'toy' / 'text').iterdir()),
                             ['bart-1.txt', 'gector-1.txt', 'source.txt', 't5-1.txt'])
            self.assertFalse((home / 'toy' / 'm2' / 'target.m2').exists())
            self.assertEqual(read_lines(home / 'toy' / 'text' / 'source.txt'), ['A b .', 'C d .', 'E f .'])

    def test_an_lfs_pointer_gold_is_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, gold = Path(tmp, 't.src'), Path(tmp, 't.m2')
            src.write_text('a\nb\n', encoding='utf-8')
            gold.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n', encoding='utf-8')
            root = Path(tmp, 'runs')
            with mock.patch.object(settings, 'ROOT', root), \
                    mock.patch.dict(settings.TEST_SETS, {'toy': (src, gold)}):
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    evaluate.main('toy')
            self.assertIn('Git LFS pointer', log.getvalue())
            self.assertFalse((root / 'toy' / 'results.md').exists())

    def test_a_tag_moves_only_the_greco_outputs(self):
        released = {p for *_, p in evaluate.outputs()}
        tagged = {p for *_, p in evaluate.outputs('s1')}
        self.assertIn(Path('k3') / 'greco+esc.rank0.4' / 'pred.txt', released)
        self.assertIn(Path('s1') / 'k3' / 'greco+esc.rank0.4' / 'pred.txt', tagged)
        self.assertIn(Path('text') / 'gector-1.txt', tagged)  # single systems do not depend on GRECO

    def test_the_shell_script_parses(self):
        subprocess.run(['bash', '-n', str(REPO / 'greco_esc' / 'test.sh')], check=True)


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


@unittest.skipUnless((REPO / 'external/greco/run_combination.py').exists() and has('torch', 'transformers'),
                     'needs external/greco and torch')
class VotedDatasetTests(unittest.TestCase):
    """combine.voted_dataset on upstream's SysCombDataset: the vote beam_search sees per mode."""

    def test_counts_per_mode(self):
        x = 'A 0 1|||R:NOUN|||A|||REQUIRED|||-NONE-|||0'
        y = 'A 2 3|||R:NOUN|||C|||REQUIRED|||-NONE-|||0'
        noop = 'A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0'
        # A from every system's top input; C only from bart at ranks 2 and 3
        edits = {'gector-1': x, 'bart-1': x, 't5-1': x, 'bart-2': y, 't5-2': noop, 'bart-3': y, 't5-3': noop}
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for name, edit in edits.items():
                path = Path(tmp, f'{name}.m2')
                path.write_text(f'S a b c d\n{edit}\n\n', encoding='utf-8')
                files.append(str(path))
            sys.path.insert(0, str(REPO / 'external/greco'))
            self.addCleanup(sys.path.remove, str(REPO / 'external/greco'))
            want = {'hypothesis': (7, {'A': 3 / 7, 'C': 2 / 7}), 'system': (3, {'A': 1, 'C': 1 / 3}),
                    'rank': (3, {'A': 1, 'C': 1 / 6})}
            for mode, (voters, support) in want.items():
                dataset = combine.voted_dataset(files, None, mode)
                got = {e[3]: count / dataset.num_hyp for e, count in dataset[0]['edits']}
                self.assertEqual(dataset.num_hyp, voters, mode)
                for edit, value in support.items():
                    self.assertAlmostEqual(got[edit], value, msg=mode)


class EvaluateTests(unittest.TestCase):
    def test_the_table_has_one_line_per_k_and_marks_missing_outputs(self):
        rows = [{'system': 'greco', 'k': 1, 'vote_mode': '-', 'vote_coef': '0', 'esc': 'no',
                 'errant_f0_5': 57.6, 'errant_p': 65.0, 'errant_r': 39.5},
                {'system': 'greco', 'k': 5, 'vote_mode': 'rank', 'vote_coef': f"{settings.VOTES['rank'][5]:g}", 'esc': 'yes',
                 'errant_f0_5': 59.0, 'errant_p': 70.0, 'errant_r': 36.0}]
        text = evaluate.table(rows)
        self.assertIn(f"| 1 | {settings.VOTE[1]:g} | 57.6 (65.0 / 39.5) | missing |", text)
        self.assertIn(f"59.0 (70.0 / 36.0) [α {settings.VOTES['rank'][5]:g}] |", text)
        self.assertIn('= hypothesis', text)
        # two tables, each with one line per k
        self.assertEqual(sum(line.startswith('| ') and line[2].isdigit() for line in text.splitlines()),
                         2 * len(settings.KS))

    def test_every_run_has_an_output_path(self):
        paths = [p for *_, p in evaluate.outputs()]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(paths), 3 + len(settings.KS) + len(settings.runs()))


if __name__ == '__main__':
    unittest.main()
