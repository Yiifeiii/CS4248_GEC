from collections import Counter
import copy
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from training.build_order_control import exposure_indices, build
from training.ordered import OrderedTrainer, validate_order_config


ROOT=Path(__file__).resolve().parents[1]


class OrderControlTests(unittest.TestCase):
    def test_exact_same_exposure_ids_and_eight_round_sizes(self):
        a,b,bounds=exposure_indices(5,3)
        self.assertEqual(Counter(a),Counter(b))
        self.assertEqual(Counter(a),{**{i:3 for i in range(5)},**{i:5 for i in range(5,8)}})
        self.assertEqual(bounds,[5,10,15,18,21,24,27,30])
        self.assertTrue(all(i<5 for i in a[:15]))
        self.assertTrue(all(i>=5 for i in a[15:]))
        self.assertNotEqual(a,b)
        self.assertEqual((a,b,bounds),exposure_indices(5,3))

    def test_config_pairs_change_only_order_and_artifact_paths(self):
        for family in ('bart','t5'):
            folder=ROOT/'training/configs/matched-order'
            a=json.loads((folder/f'{family}-curriculum.json').read_text())
            b=json.loads((folder/f'{family}-shuffled.json').read_text())
            for key in ('run','train','output_dir','order_mode'):
                a.pop(key);b.pop(key)
            self.assertEqual(a,b)
            validate_order_config(a)
            self.assertEqual(a['batch_size'],64)
            self.assertEqual(a['logical_rounds'],8)
            for key,value in [('epochs',8),('grad_accum',4),('max_steps',100)]:
                bad=copy.deepcopy(a);bad[key]=value
                with self.assertRaises(ValueError):validate_order_config(bad)

    def test_built_files_match_exposure_counts_even_with_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest={}
            for name,values in [('stage1',['a','b']),('stage2',['b','c'])]:
                hashes={}
                for suffix in ('src','tgt'):
                    path=root/f'{name}.{suffix}'
                    path.write_text(''.join(s+'\n' for s in values))
                    hashes[suffix]=hashlib.sha256(path.read_bytes()).hexdigest()
                manifest[name]={'pairs':len(values),'sha256':hashes}
            (root/'manifest.json').write_text(json.dumps(manifest))
            record=build(root,root/'out')
            self.assertEqual(record['total_examples'],16)
            for order in ('curriculum','shuffled'):
                text=(root/'out'/order/'train.src').read_text().splitlines()
                self.assertEqual(Counter(text),{'a':3,'b':8,'c':5})

    def test_preview_never_calls_sbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            stub=Path(directory)/'sbatch';stub.write_text('#!/bin/sh\nexit 99\n');stub.chmod(0o755)
            env=dict(os.environ,PATH=directory+os.pathsep+os.environ['PATH'])
            result=subprocess.run(['bash',str(ROOT/'cluster/submit_order_control.sh')],
                cwd=directory,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('No submission',result.stdout)

    def test_real_trainer_consumes_each_example_once_in_order_including_tail(self):
        import torch
        from transformers import BartConfig,BartForConditionalGeneration,Seq2SeqTrainingArguments
        from transformers import default_data_collator
        class RecordingTrainer(OrderedTrainer):
            def compute_loss(self,model,inputs,**kwargs):
                self.seen.extend(inputs['input_ids'][:,0].tolist())
                return super().compute_loss(model,inputs,**kwargs)
        dataset=[{'input_ids':[i,4,2],'attention_mask':[1,1,1],'labels':[0,4,2]} for i in range(5,22)]
        config=BartConfig(vocab_size=32,d_model=8,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=1,decoder_attention_heads=1,encoder_ffn_dim=16,
            decoder_ffn_dim=16,max_position_embeddings=16,pad_token_id=1,
            bos_token_id=0,eos_token_id=2,decoder_start_token_id=2)
        with tempfile.TemporaryDirectory() as directory:
            args=Seq2SeqTrainingArguments(output_dir=directory,use_cpu=True,
                num_train_epochs=1,per_device_train_batch_size=4,gradient_accumulation_steps=1,
                report_to='none',save_strategy='no',logging_strategy='no',disable_tqdm=True,
                dataloader_num_workers=0,group_by_length=False)
            trainer=RecordingTrainer(model=BartForConditionalGeneration(config),args=args,
                train_dataset=dataset,data_collator=default_data_collator)
            trainer.seen=[];trainer.train()
            self.assertEqual(trainer.seen,list(range(5,22)))
            self.assertEqual(trainer.state.global_step,5)


if __name__=='__main__':unittest.main()
