import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from training.candidate_quality import pool_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'training/configs/bart-decoding-20261106.json'


class CandidateCoverageTests(unittest.TestCase):
    def test_extra_candidates_cover_more_edits_without_padding_credit(self):
        a, b, wrong = (0, 1, 'He'), (1, 2, 'goes'), (2, 3, 'away')
        texts = [['He go home .', 'he goes home .', 'He goes away .'], ['Hello .']]
        edits = [[{a}, {b}, {a, b, wrong}], [set()]]
        gold = [{a, b}, set()]
        one = pool_metrics(texts, edits, gold, 1)
        three = pool_metrics(texts, edits, gold, 3)
        five = pool_metrics(texts, edits, gold, 5)
        self.assertEqual(one['gold_edit_coverage'], 0.5)
        self.assertEqual(three['gold_edit_coverage'], 1)
        self.assertEqual(five['gold_edit_coverage'], 1)
        self.assertEqual(three['pooled_candidate_edits'], 3)
        self.assertEqual(five['mean_unique_candidates'], 2)
        self.assertEqual(five['sources_with_full_k'], 0)
        self.assertNotIn('f0_5', five)  # Pool coverage is not a correction-system score.
        self.assertIsNone(one['mean_pairwise_token_dissimilarity'])
        self.assertGreater(three['mean_pairwise_token_dissimilarity'], 0)

    def test_no_gold_edits_does_not_divide_by_zero(self):
        result = pool_metrics([['Hello .']], [[set()]], [set()], 5)
        self.assertIsNone(result['gold_edit_coverage'])

    def test_misalignment_and_duplicate_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            pool_metrics([['a']], [], [set()], 1)
        with self.assertRaises(ValueError):
            pool_metrics([['a', 'a']], [[set(), set()]], [set()], 3)
        with self.assertRaises(ValueError):
            pool_metrics([[]], [[]], [set()], 1)


class BartDecodingGridTests(unittest.TestCase):
    def test_controlled_grid_and_divisibility(self):
        cfg = json.loads(CONFIG.read_text())
        self.assertEqual(cfg['model'], 'CS4248-GEC/gec-bart-base-s1s2')
        self.assertEqual(cfg['split'], 'dev_tune')
        self.assertEqual(cfg['sentences'], 2000)
        self.assertEqual(cfg['k'], [1, 3, 5])
        self.assertEqual(cfg['num_return'], 5)
        self.assertEqual(len({s['name'] for s in cfg['setups']}), 7)
        plain = [s['beams'] for s in cfg['setups'] if s['mode'] == 'beam']
        self.assertEqual(plain, [5, 10, 20])
        diverse = [s for s in cfg['setups'] if s['mode'] == 'diverse']
        self.assertEqual({(s['groups'], s['penalty']) for s in diverse},
                         {(2, 0.1), (2, 0.5), (2, 1), (5, 0.5)})
        for setup in cfg['setups']:
            self.assertEqual(setup['beams'] % setup['groups'], 0)
            self.assertGreaterEqual(setup['beams'], cfg['num_return'])

    def test_preview_does_not_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / 'sbatch'
            stub.write_text('#!/bin/sh\nexit 99\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(ROOT / 'cluster/submit_bart_decoding.sh')],
                                    cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Preview only', result.stdout)

    def test_grouped_generation_works_on_tiny_bart_offline(self):
        import torch
        from transformers import BartConfig, BartForConditionalGeneration
        from bart.generate_nbest import generate_kwargs
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = BartForConditionalGeneration(BartConfig(vocab_size=32, d_model=16,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
            decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
            pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2)).eval()
        for setup in json.loads(CONFIG.read_text())['setups']:
            with self.subTest(setup=setup['name']):
                args = SimpleNamespace(mode=setup['mode'], num_beams=setup['beams'], num_return=5,
                    num_beam_groups=setup['groups'], diversity_penalty=setup['penalty'],
                    length_penalty=1, no_repeat_ngram_size=0)
                kwargs = generate_kwargs(args, 6)
                self.assertFalse(kwargs['do_sample'])
                if setup['mode'] == 'beam':
                    self.assertNotIn('diversity_penalty', kwargs)
                else:
                    self.assertEqual(kwargs['num_beam_groups'], setup['groups'])
                with torch.no_grad():
                    output = model.generate(input_ids=torch.tensor([[0, 5, 6, 2]]),
                                            attention_mask=torch.ones((1, 4), dtype=torch.long), **kwargs)
                self.assertEqual(output.shape[0], 5)


if __name__ == '__main__':
    unittest.main()
