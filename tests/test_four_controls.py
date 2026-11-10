from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from training.build_four_controls import streams
from training.exposure_batching import ExposureBatchSampler,ExactExposureTrainer,validate_exposure_config


ROOT=Path(__file__).resolve().parents[1]


class FourControlsTests(unittest.TestCase):
    def test_exact_exposure_cannot_be_shortened_by_config(self):
        cfg=dict(exact_exposure_batches=True,epochs=1,grad_accum=2,errant_eval={'patience':None})
        validate_exposure_config(cfg)
        for key,value in [('epochs',3),('grad_accum',0),('max_steps',10)]:
            bad=dict(cfg,**{key:value})
            with self.assertRaises(ValueError):validate_exposure_config(bad)
        with self.assertRaises(ValueError):validate_exposure_config(dict(cfg,errant_eval={'patience':2}))
    def test_matched_and_uniform_exposures_include_overlap_correctly(self):
        a,b,c=('a','A'),('b','B'),('c','c')
        rows,n=streams([a,b,b],[b,c,c])
        self.assertEqual(n,3)
        self.assertEqual(Counter(rows['matched']),{a:3,b:11,c:10})
        self.assertEqual(Counter(rows['uniform3']),{a:3,b:3,c:3})
        self.assertEqual(Counter(rows['uniform5']),{a:5,b:5,c:5})
        for name,reps in [('uniform3',3),('uniform5',5)]:
            for i in range(reps):self.assertEqual(set(rows[name][i*n:(i+1)*n]),{a,b,c})
        self.assertEqual(rows,streams([a,b,b],[b,c,c])[0])

    def test_batching_keeps_every_example_without_padding_or_accumulation_loss(self):
        for batch in (2,4,64):
            for accumulation in (1,2,4):
                for n in range(accumulation,accumulation+4*batch*accumulation):
                    sampler=ExposureBatchSampler(n,batch,accumulation)
                    batches=list(sampler)
                    self.assertEqual([i for b in batches for i in b],list(range(n)))
                    self.assertTrue(all(0<len(b)<=batch for b in batches))
                    self.assertEqual(len(batches),len(sampler))
                    self.assertEqual(len(batches)%accumulation,0)

    def test_configs_inherit_stage_optimization_but_share_evaluation(self):
        optimization=('learning_rate','batch_size','grad_accum','warmup_steps','weight_decay','dropout','label_smoothing','seed')
        for family in ('bart','t5'):
            for slug,stage,stream in [('1a',1,'matched'),('1b',2,'matched'),('2a',1,'uniform3'),('2b',2,'uniform5')]:
                cfg=json.loads((ROOT/f'training/configs/four-controls/{family}-{slug}.json').read_text())
                old=json.loads((ROOT/f'training/configs/{family}-base.stage{stage}.json').read_text())
                for key in optimization:self.assertEqual(cfg[key],old[key],key)
                self.assertEqual(cfg['control_stream'],stream)
                self.assertEqual(cfg['epochs'],1)
                self.assertIsNone(cfg['errant_eval']['patience'])
                self.assertEqual(cfg['errant_eval']['sentences'],2000)
                self.assertEqual(cfg['errant_eval']['beams'],5)
                self.assertEqual(cfg['errant_eval']['steps']*cfg['batch_size']*cfg['grad_accum'],128000)

    def test_preview_never_submits(self):
        with tempfile.TemporaryDirectory() as directory:
            stub=Path(directory)/'sbatch';stub.write_text('#!/bin/sh\nexit 99\n');stub.chmod(0o755)
            result=subprocess.run(['bash',str(ROOT/'cluster/submit_four_controls.sh')],cwd=directory,
                env=dict(os.environ,PATH=directory+os.pathsep+os.environ['PATH']),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_real_training_keeps_tail_with_stage1_accumulation(self):
        from transformers import BartConfig,BartForConditionalGeneration,Seq2SeqTrainingArguments,default_data_collator
        class RecordingTrainer(ExactExposureTrainer):
            def compute_loss(self,model,inputs,**kwargs):
                self.seen.extend(inputs['input_ids'][:,0].tolist())
                return super().compute_loss(model,inputs,**kwargs)
        config=BartConfig(vocab_size=32,d_model=8,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=1,decoder_attention_heads=1,encoder_ffn_dim=16,
            decoder_ffn_dim=16,max_position_embeddings=16,pad_token_id=1,
            bos_token_id=0,eos_token_id=2,decoder_start_token_id=2)
        # 9 examples at batch4/accum2: the last single example must not disappear.
        dataset=[{'input_ids':[i,4,2],'attention_mask':[1,1,1],'labels':[0,4,2]} for i in range(5,14)]
        with tempfile.TemporaryDirectory() as directory:
            args=Seq2SeqTrainingArguments(output_dir=directory,use_cpu=True,num_train_epochs=1,
                per_device_train_batch_size=4,gradient_accumulation_steps=2,
                report_to='none',save_strategy='no',logging_strategy='no',disable_tqdm=True,
                dataloader_num_workers=0,group_by_length=False)
            trainer=RecordingTrainer(model=BartForConditionalGeneration(config),args=args,
                train_dataset=dataset,data_collator=default_data_collator)
            trainer.seen=[];trainer.train()
            self.assertEqual(trainer.seen,list(range(5,14)))
            self.assertEqual(trainer.state.global_step,2)


if __name__=='__main__':unittest.main()
