"""Retain every example while preserving configured gradient accumulation."""
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import seed_worker


class ExposureBatchSampler:
    """Split the final update into nonempty microbatches; never pad/drop examples."""
    def __init__(self, size, batch_size, accumulation):
        if min(size,batch_size,accumulation)<1 or size<accumulation:
            raise ValueError('Need enough examples for nonempty accumulation microbatches')
        self.size,self.batch_size,self.accumulation=size,batch_size,accumulation
        self.drop_last=False
        self.sampler=SequentialSampler(range(size))
        effective=batch_size*accumulation
        self.full_updates,self.tail=divmod(size,effective)
        self.tail_batches=accumulation if self.tail else 0
        if self.tail and self.tail<accumulation:
            self.full_updates-=1
            self.tail+=effective
            self.tail_batches=2*accumulation

    def __len__(self):
        return self.full_updates*self.accumulation+self.tail_batches

    def __iter__(self):
        cursor=0
        for _ in range(self.full_updates*self.accumulation):
            yield list(range(cursor,cursor+self.batch_size))
            cursor+=self.batch_size
        if self.tail_batches:
            base,extra=divmod(self.tail,self.tail_batches)
            for i in range(self.tail_batches):
                count=base+(i<extra)
                yield list(range(cursor,cursor+count))
                cursor+=count
        assert cursor==self.size


class ExactExposureTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        if self.args.world_size!=1 or self.args.dataloader_drop_last:
            raise ValueError('Exact exposure requires one process and no dropped examples')
        batches=ExposureBatchSampler(len(self.train_dataset),self._train_batch_size,
                                     self.args.gradient_accumulation_steps)
        kwargs=dict(batch_sampler=batches,collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,worker_init_fn=seed_worker)
        if self.args.dataloader_num_workers:
            kwargs['persistent_workers']=self.args.dataloader_persistent_workers
            kwargs['prefetch_factor']=self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(self.train_dataset,**kwargs))
