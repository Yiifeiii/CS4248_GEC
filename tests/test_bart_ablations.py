import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from training.dropout import configure_dropout


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'training/configs/bart-ablations-20261105'
CHANGES = {
    'baseline': {},
    'lr-low': {'learning_rate': 5e-6},
    'lr-high': {'learning_rate': 2e-5},
    'wd-zero': {'weight_decay': 0},
    'wd-high': {'weight_decay': 0.1},
    'dropout-zero': {'dropout': 0},
    'dropout-high': {'dropout': 0.3},
    'smoothing-zero': {'label_smoothing': 0},
    'smoothing-high': {'label_smoothing': 0.2},
}


def configs():
    return {name: json.loads((CONFIGS / f'{name}.json').read_text()) for name in CHANGES}


class AblationDesignTests(unittest.TestCase):
    def test_submission_defaults_to_preview_without_calling_slurm(self):
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / 'sbatch'
            stub.write_text('#!/bin/sh\necho UNEXPECTED_SUBMISSION >&2\nexit 99\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(ROOT / 'cluster/submit_bart_ablations.sh')],
                                    cwd=directory, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Preview only', result.stdout)
            self.assertNotIn('UNEXPECTED_SUBMISSION', result.stderr)

    def test_single_factor_changes_and_no_checkpoint_collisions(self):
        runs = configs()
        self.assertEqual((CONFIGS / 'runs.txt').read_text().splitlines(), list(CHANGES))
        self.assertEqual(len({c['output_dir'] for c in runs.values()}), 9)
        self.assertEqual(len({c['run'] for c in runs.values()}), 9)
        for name, change in CHANGES.items():
            with self.subTest(run=name):
                expected = copy.deepcopy(runs['baseline'])
                expected.update(change)
                expected['run'] = f'bart-ablations-20261105/{name}'
                expected['output_dir'] = '${CKPT_ROOT}/' + expected['run']
                self.assertEqual(runs[name], expected)
                self.assertEqual(runs[name]['train'], 'data/curriculum/stage2')
                self.assertEqual(runs[name]['init'], 'CS4248-GEC/gec-bart-base-s1')

    def test_baseline_retains_original_stage2_recipe(self):
        reference = json.loads((ROOT / 'experiments/training/bart-base.s1s2/run_info.json').read_text())
        original = reference['config']
        self.assertEqual(original, json.loads((ROOT / 'training/configs/bart-base.stage2.json').read_text()))
        baseline = configs()['baseline']
        for key, value in original.items():
            if key not in ('run', 'init', 'output_dir'):
                self.assertEqual(baseline[key], value, key)
        self.assertEqual(baseline['dropout_scope'], 'all')


class DropoutConfigurationTests(unittest.TestCase):
    def test_no_dropout_disables_all_bart_seq2seq_dropout(self):
        config = SimpleNamespace(model_type='bart', dropout=0.1,
                                 attention_dropout=0.1, activation_dropout=0.1)
        applied = configure_dropout(config, 0, 'all')
        self.assertEqual(applied, dict(dropout=0, attention_dropout=0, activation_dropout=0))

    def test_existing_configs_keep_their_original_scope(self):
        config = SimpleNamespace(model_type='bart', dropout=0.1,
                                 attention_dropout=0.1, activation_dropout=0.1)
        configure_dropout(config, 0.3)
        self.assertEqual((config.dropout, config.attention_dropout, config.activation_dropout),
                         (0.3, 0.1, 0.1))

    def test_t5_uses_its_dropout_rate(self):
        config = SimpleNamespace(model_type='t5', dropout_rate=0.1)
        self.assertEqual(configure_dropout(config, 0.3, 'all'), {'dropout_rate': 0.3})

    def test_invalid_probabilities_and_scopes_are_rejected(self):
        config = SimpleNamespace(model_type='bart')
        for rate in (-0.1, 1, float('nan'), float('inf')):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                configure_dropout(config, rate, 'all')
        with self.assertRaises(ValueError):
            configure_dropout(config, 0.1, 'unknown')


class TinyBartTrainingTests(unittest.TestCase):
    """Exercise real Trainer/AdamW/label smoothing offline on a tiny random BART.

    These are two-update CPU smoke tests, not GEC experiments or score estimates.
    """
    def test_all_nine_recipes_train_and_save_without_external_models(self):
        import torch
        from transformers import (BartConfig, BartForConditionalGeneration,
                                  Seq2SeqTrainer, Seq2SeqTrainingArguments)
        torch.set_num_threads(1)
        for name, cfg in configs().items():
            with self.subTest(run=name), tempfile.TemporaryDirectory() as folder:
                torch.manual_seed(cfg['seed'])
                model_config = BartConfig(vocab_size=32, d_model=16,
                    encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                    decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                    max_position_embeddings=32, pad_token_id=1, bos_token_id=0,
                    eos_token_id=2, decoder_start_token_id=2)
                configure_dropout(model_config, cfg['dropout'], cfg['dropout_scope'])
                model = BartForConditionalGeneration(model_config)
                for layer in [*model.model.encoder.layers, *model.model.decoder.layers]:
                    self.assertEqual(layer.dropout, cfg['dropout'])
                    self.assertEqual(layer.activation_dropout, cfg['dropout'])
                    self.assertEqual(layer.self_attn.dropout, cfg['dropout'])
                self.assertEqual(model.model.decoder.layers[0].encoder_attn.dropout, cfg['dropout'])
                before = model.model.shared.weight.detach().clone()
                dataset = [dict(input_ids=[0, 5, 6, 2], attention_mask=[1, 1, 1, 1],
                                labels=[7, 6, 2], decoder_input_ids=[2, 7, 6])] * 4
                arguments = Seq2SeqTrainingArguments(output_dir=folder, use_cpu=True,
                    max_steps=2, per_device_train_batch_size=2, report_to='none',
                    disable_tqdm=True, save_strategy='no', logging_strategy='no',
                    optim='adamw_torch', learning_rate=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'], label_smoothing_factor=cfg['label_smoothing'],
                    lr_scheduler_type='linear', warmup_steps=0, seed=cfg['seed'])
                trainer = Seq2SeqTrainer(model=model, args=arguments, train_dataset=dataset)
                result = trainer.train()
                self.assertEqual(result.global_step, 2)
                self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
                self.assertFalse(torch.equal(before, model.model.shared.weight))
                nonzero_decay = {g['weight_decay'] for g in trainer.optimizer.param_groups
                                 if g['weight_decay'] != 0}
                self.assertEqual(nonzero_decay, {cfg['weight_decay']} if cfg['weight_decay'] else set())
                trainer.save_model(folder)
                restored = BartForConditionalGeneration.from_pretrained(folder, local_files_only=True)
                self.assertTrue(torch.equal(restored.model.shared.weight, model.model.shared.weight))


if __name__ == '__main__':
    unittest.main()
