"""Single-GPU, no-reshuffle trainer for a fixed experimental exposure stream."""
from torch.utils.data import SequentialSampler
from transformers import Seq2SeqTrainer


class OrderedTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self):
        if self.args.world_size != 1:
            raise ValueError('Matched order experiments require a single training process')
        return SequentialSampler(self.train_dataset)


def validate_order_config(cfg):
    if cfg.get('preserve_train_order'):
        if (cfg['epochs'] != 1 or cfg.get('max_steps') or
                (cfg['grad_accum'] != 1 and not cfg.get('exact_exposure_batches'))):
            raise ValueError('Fixed stream needs one full pass, no accumulation or max-step truncation')
        if cfg['errant_eval'].get('patience'):
            raise ValueError('Early stopping would break matched example exposure')
