import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('tune', ROOT / 't5_tuning' / 'tune.py')
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)

RUN = 't5-tune.base.lr3e-5.seed42'


def curve(losses):
    return [{'step': 10 * (i + 1), 'loss': loss, 'grad_norm': 1.0} for i, loss in enumerate(losses)]


class ConfigTests(unittest.TestCase):
    def test_every_committed_config_matches_its_manifest_row(self):
        for row in tune.load_manifest().values():
            with self.subTest(run=row['run']):
                self.assertEqual(tune.config_problems(row), [])

    def test_a_changed_setting_is_reported(self):
        row = tune.load_manifest()[RUN]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = json.loads((ROOT / row['config']).read_text(encoding='utf-8'))
            cfg['learning_rate'] = 5e-5
            path = Path(tmp, 'config.json')
            path.write_text(json.dumps(cfg), encoding='utf-8')
            self.assertEqual(len(tune.config_problems(row, path)), 1)

    def test_numbers_compare_by_value(self):
        self.assertTrue(tune.same(3e-05, '3e-5'))
        self.assertFalse(tune.same('a', 'b'))


class StabilityTests(unittest.TestCase):
    def test_a_smooth_decreasing_curve_passes(self):
        self.assertEqual(tune.stability(curve([3.0, 2.5, 2.0, 1.8, 1.7, 1.6])), ([], []))

    def test_a_non_finite_loss_fails(self):
        failures, _ = tune.stability(curve([3.0, math.nan, 2.0]))
        self.assertEqual(len(failures), 1)

    def test_a_spike_after_the_start_is_a_warning(self):
        failures, warnings = tune.stability(curve([3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 0.9]))
        self.assertEqual(failures, [])
        self.assertTrue(any('spike' in w for w in warnings))

    def test_a_loss_that_does_not_fall_is_a_warning(self):
        _, warnings = tune.stability(curve([2.0, 2.1, 2.2]))
        self.assertTrue(any('did not decrease' in w for w in warnings))


class RunTests(unittest.TestCase):
    """A finished run, laid out as run_one.sh leaves it, checked against its committed config."""

    def make_run(self, tmp, **changes):
        row = tune.load_manifest()[RUN]
        results, ckpt = Path(tmp, 'runs'), Path(tmp, 'ckpt')
        (results / RUN).mkdir(parents=True)
        (ckpt / RUN).mkdir(parents=True)
        (results / RUN / 'dev_tune.scores.json').write_text('{}', encoding='utf-8')
        n = changes.pop('pred_lines', tune.SPLIT_SIZES['dev_tune'])
        (results / RUN / 'dev_tune.pred').write_text('x\n' * n, encoding='utf-8')
        config = json.loads((ROOT / row['config']).read_text(encoding='utf-8'))
        info = {
            'config': {**config, **changes.pop('config', {})},
            'training_pairs': json.loads(tune.DATA_MANIFEST.read_text(encoding='utf-8'))['stage2']['pairs'],
            'data_manifest_sha256': tune.sha256(tune.DATA_MANIFEST),
            'versions': tune.recorded_versions(),
            'finished': '2026-11-01 00:00:00',
            **changes,
        }
        (ckpt / RUN / 'run_info.json').write_text(json.dumps(info), encoding='utf-8')
        (ckpt / RUN / 'trainer_state.json').write_text(
            json.dumps({'log_history': curve([3.0, 2.0, 1.5])}), encoding='utf-8')
        return tune.run_problems(row, ckpt, results)

    def test_a_run_that_matches_its_config_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.make_run(tmp), ([], []))

    def test_a_different_learning_rate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            failures, _ = self.make_run(tmp, config={'learning_rate': 5e-5})
            self.assertTrue(any('learning_rate' in f for f in failures))

    def test_different_software_versions_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            failures, _ = self.make_run(tmp, versions={'torch': '0.0', 'transformers': '0.0'})
            self.assertTrue(any('recorded runs used' in f for f in failures))

    def test_a_short_prediction_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            failures, _ = self.make_run(tmp, pred_lines=1999)
            self.assertTrue(any('1999 lines' in f for f in failures))

    def test_an_unfinished_run_fails(self):
        # An interrupted run's run_info.json has no "finished" time
        with tempfile.TemporaryDirectory() as tmp:
            self.make_run(tmp)
            info_path = Path(tmp, 'ckpt', RUN, 'run_info.json')
            info = json.loads(info_path.read_text(encoding='utf-8'))
            del info['finished']
            info_path.write_text(json.dumps(info), encoding='utf-8')
            failures, _ = tune.run_problems(tune.load_manifest()[RUN], Path(tmp, 'ckpt'), Path(tmp, 'runs'))
            self.assertTrue(any('did not finish' in f for f in failures))


class FinalDirTests(unittest.TestCase):
    def test_untagged_stage_2_keeps_the_original_folder(self):
        self.assertEqual(tune.final_dir(2, None).name, 'final-dev_test')
        self.assertEqual(tune.final_dir(1, None).name, 'final-dev_test-s1')
        self.assertEqual(tune.final_dir(2, 's1lr3e-4').name, 'final-dev_test-s1lr3e-4')


if __name__ == '__main__':
    unittest.main()
