"""Fine-tune BART or T5 for one stage of the two-stage curriculum.

    python -m training.finetune --config training/configs/bart-base.stage1.json [--resume auto]

The config (see training/configs/) names the starting weights, the stage data and
the hyperparameters. Paths may use ${VARS}: CKPT_ROOT is where checkpoints go
(cluster: /mnt/scratch/..., see cluster/README.md). Writes to <output_dir>:
  checkpoint-*/        Trainer checkpoints for resuming (the last save_total_limit)
  best/                model with the best dev_tune ERRANT F0.5 so far (+ best.json)
  final/               model at the end of training
  errant_eval.json     every dev_tune ERRANT evaluation
  trainer_state.json   loss curve; run_info.json: config, git commit, data manifest

Smoke test (any device): add --max-steps 20 --train-limit 500 --dev-limit 20 --output-dir /tmp/x
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# torch >= 2.6 loads with weights_only=True, which rejects the numpy RNG state that
# transformers 4.46 saves in each checkpoint, so resuming fails. The checkpoints are our own.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import numpy as np  # noqa: E402
import torch
import transformers
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback)
from transformers.trainer_utils import get_last_checkpoint

from training.common import (CURRICULUM, MAX_TOKENS, PREFIX, ROOT, decode, expand, family, git_commit,
                             pick_device, read_lines, write_lines)
from training.metrics import m2_subset, score
from training.dropout import configure_dropout
from training.exposure_batching import ExactExposureTrainer, validate_exposure_config


class PairDataset(torch.utils.data.Dataset):
    """Pre-tokenized (source, target) pairs, stored as int32 arrays."""

    def __init__(self, tokenizer, prefix, sources, targets, chunk=20000):
        self.inputs, self.labels = [], []
        for i in range(0, len(sources), chunk):
            enc = tokenizer([prefix + s for s in sources[i:i + chunk]], text_target=targets[i:i + chunk],
                            truncation=True, max_length=MAX_TOKENS)
            self.inputs += [np.array(x, dtype=np.int32) for x in enc['input_ids']]
            self.labels += [np.array(x, dtype=np.int32) for x in enc['labels']]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        ids = self.inputs[i].tolist()
        return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': self.labels[i].tolist()}


class ErrantEval(TrainerCallback):
    """ERRANT F0.5 on dev_tune during training; keeps best/ and stops early on a plateau."""

    def __init__(self, cfg, tokenizer, output_dir, dev_src, dev_m2):
        self.every = cfg['steps']                     # an int, or 'epoch'
        self.beams = cfg.get('beams', 1)
        self.patience = cfg.get('patience')
        self.tokenizer = tokenizer
        self.out = Path(output_dir)
        self.dev_src, self.dev_m2 = dev_src, dev_m2
        self.sources = read_lines(dev_src)
        self.history_path = self.out / 'errant_eval.json'
        self.history = json.loads(self.history_path.read_text()) if self.history_path.exists() else []
        self.model = None

    def best(self):
        return max(self.history, key=lambda h: h['f0_5'], default=None)

    def run(self, state, control):
        if any(h['step'] == state.global_step for h in self.history):
            return
        started = time.time()
        hyps = decode(self.model, self.tokenizer, self.sources, num_beams=self.beams)
        e = score(self.dev_src, hyps, self.dev_m2, m2=False)['errant']['correction']
        record = {'step': state.global_step, 'epoch': round(state.epoch or 0, 3), 'beams': self.beams,
                  'sentences': len(self.sources), **e, 'seconds': round(time.time() - started)}
        best = self.best()
        self.history.append(record)
        self.history_path.write_text(json.dumps(self.history, indent=2) + '\n')
        print(f"[errant] step {record['step']} P {e['precision']:.4f} R {e['recall']:.4f} "
              f"F0.5 {e['f0_5']:.4f} ({record['seconds']}s)", flush=True)
        if best is None or e['f0_5'] > best['f0_5']:
            self.model.save_pretrained(self.out / 'best')
            self.tokenizer.save_pretrained(self.out / 'best')
            write_lines(self.out / 'best' / 'dev_tune.pred', hyps)
            (self.out / 'best' / 'best.json').write_text(json.dumps(record, indent=2) + '\n')
        elif self.patience:
            since = len(self.history) - 1 - self.history.index(self.best())
            if since >= self.patience:
                print(f'[errant] no improvement in {since} evaluations; stopping', flush=True)
                control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.every != 'epoch' and state.global_step % self.every == 0:
            self.run(state, control)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.every == 'epoch':
            self.run(state, control)

    def on_train_end(self, args, state, control, **kwargs):
        self.run(state, control)                  # always score the final weights


def load_pairs(prefix, errorful_only, limit):
    prefix = ROOT / expand(prefix)                # relative paths are from the repo root
    src, tgt = read_lines(f'{prefix}.src'), read_lines(f'{prefix}.tgt')
    pairs = [(s, t) for s, t in zip(src, tgt) if not errorful_only or s != t]
    return pairs[:limit] if limit else pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', choices=['auto', 'no'], default='auto',
                        help='auto: continue from the last checkpoint in output_dir if there is one')
    parser.add_argument('--output-dir', help='override the config output_dir')
    parser.add_argument('--max-steps', type=int, help='stop after this many steps (smoke tests and pilots)')
    parser.add_argument('--train-limit', type=int, help='use only the first N training pairs')
    parser.add_argument('--dev-limit', type=int, help='use only the first N dev_tune sentences')
    parser.add_argument('--errant-steps', type=int, help='override errant_eval.steps')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate_exposure_config(cfg)
    if cfg.get('exact_exposure_batches') and (args.max_steps or args.train_limit):
        raise ValueError('Fixed exposure streams cannot be shortened; use the tiny offline tests')
    out = Path(expand(args.output_dir or cfg['output_dir']))
    out.mkdir(parents=True, exist_ok=True)
    init = expand(cfg['init'])
    device = pick_device()

    config = AutoConfig.from_pretrained(init)
    model_family = family(config)
    dropout_settings = configure_dropout(config, cfg['dropout'], cfg.get('dropout_scope', 'model'))
    tokenizer = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSeq2SeqLM.from_pretrained(init, config=config)
    gen = model.generation_config
    gen.no_repeat_ngram_size, gen.num_beams, gen.max_new_tokens, gen.early_stopping = 0, 5, MAX_TOKENS + 32, True
    prefix = PREFIX[model_family]

    train = load_pairs(cfg['train'], cfg.get('errorful_only', False), args.train_limit)
    dev_src_all = read_lines(CURRICULUM / 'dev_tune.src')
    n_dev = min(x for x in (args.dev_limit, cfg['errant_eval'].get('sentences'), len(dev_src_all)) if x)
    eval_dir = out / 'eval_data'
    eval_dir.mkdir(exist_ok=True)
    dev_src = eval_dir / f'dev_tune.{n_dev}.src'
    write_lines(dev_src, dev_src_all[:n_dev])
    dev_m2 = m2_subset(CURRICULUM / 'dev_tune.m2', n_dev, eval_dir / f'dev_tune.{n_dev}.m2')
    n_loss = min(x for x in (args.dev_limit, len(dev_src_all)) if x)
    train_set = PairDataset(tokenizer, prefix, *zip(*train))
    loss_set = PairDataset(tokenizer, prefix, dev_src_all[:n_loss], read_lines(CURRICULUM / 'dev_tune.tgt')[:n_loss])
    print(f'{model_family} from {init}: {len(train_set)} training pairs, {n_loss} dev_tune pairs for loss, '
          f'{n_dev} for ERRANT, device {device.type}', flush=True)

    cuda = device.type == 'cuda'
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out), seed=cfg['seed'], data_seed=cfg['seed'],
        num_train_epochs=cfg['epochs'], max_steps=args.max_steps or cfg.get('max_steps') or -1,
        per_device_train_batch_size=cfg['batch_size'], per_device_eval_batch_size=cfg['batch_size'] * 2,
        gradient_accumulation_steps=cfg['grad_accum'],
        optim='adamw_torch', learning_rate=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
        lr_scheduler_type='linear', warmup_steps=cfg['warmup_steps'],
        label_smoothing_factor=cfg['label_smoothing'],
        bf16=cuda and torch.cuda.is_bf16_supported(), group_by_length=not cfg.get('exact_exposure_batches', False),
        eval_strategy='steps', eval_steps=cfg['loss_eval_steps'],
        save_strategy='steps', save_steps=cfg['save_steps'], save_total_limit=cfg['save_total_limit'],
        logging_steps=cfg.get('logging_steps', 50), report_to='none', remove_unused_columns=False,
        dataloader_num_workers=4 if cuda else 0, predict_with_generate=False,
    )
    errant_cfg = dict(cfg['errant_eval'])
    if args.errant_steps:
        errant_cfg['steps'] = args.errant_steps
    callback = ErrantEval(errant_cfg, tokenizer, out, dev_src, dev_m2)
    trainer_class = ExactExposureTrainer if cfg.get('exact_exposure_batches') else Seq2SeqTrainer
    trainer = trainer_class(
        model=model, args=training_args, train_dataset=train_set, eval_dataset=loss_set,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        processing_class=tokenizer, callbacks=[callback])
    callback.model = trainer.model

    manifest = ROOT / 'data/curriculum/manifest.json'
    run_info = {
        'run': cfg.get('run'), 'config': cfg, 'config_path': str(args.config), 'args': vars(args) | {'config': str(args.config)},
        'init': init, 'family': model_family, 'prefix': prefix, 'training_pairs': len(train_set),
        'dropout_settings': dropout_settings,
        'git_commit': git_commit(), 'data_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
        'device': torch.cuda.get_device_name() if cuda else device.type,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (out / 'run_info.json').write_text(json.dumps(run_info, indent=2) + '\n')

    last = get_last_checkpoint(str(out)) if args.resume == 'auto' and any(out.glob('checkpoint-*')) else None
    if last:
        print(f'resuming from {last}', flush=True)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(out / 'final'))
    trainer.state.save_to_json(str(out / 'trainer_state.json'))
    run_info.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'), steps=trainer.state.global_step,
                    best=callback.best())
    (out / 'run_info.json').write_text(json.dumps(run_info, indent=2) + '\n')
    print(f'best dev_tune ERRANT F0.5 {callback.best()["f0_5"]:.4f} at step {callback.best()["step"]}; '
          f'model in {out / "best"}', flush=True)


if __name__ == '__main__':
    main()
