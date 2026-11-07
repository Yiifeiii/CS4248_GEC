"""Run upstream GRECO train.py with a dataset that tokenizes each group when it is read.

Upstream's TrainDataset tokenizes the whole file up front and keeps every record in memory
as Python lists. On the 538,646 stage-1 groups that used the job's full 64 GB within
minutes. LazyTrainDataset stores only each line's byte offset and runs upstream's own
tokenize() on a group when the trainer asks for it, so the model sees the same records.

Run from external/greco, with train.py's arguments (train.sh does this):

    python ../../greco_tuning/train_lazy.py --do_train ... --data_mode hierarchical
"""

import json
import os
import runpy
import sys

from torch.utils.data import Dataset

sys.path.insert(0, os.getcwd())  # external/greco, so utils.data and models import
import utils.data as upstream_data  # noqa: E402


class LazyTrainDataset(Dataset):
    """Same interface as upstream TrainDataset, without holding the tokenized file in memory."""

    def __init__(self, train_file, tokenizer, mask_source=False, max_length=200, mode='list',
                 label_smoothing=0, additional_mask=0):
        self.path = train_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_source = mask_source
        self.label_smoothing = label_smoothing
        self.additional_mask = additional_mask
        self.offsets = []
        offset = 0
        with open(train_file, 'rb') as f:
            for line in f:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        # Upstream's bucket_size is the number of hypotheses per group
        self.bucket_size = len(self[0]['input_ids'])

    def group(self, index):
        """One JSON line as upstream's transform_list_to_dict leaves it: one list per key."""
        with open(self.path, 'rb') as f:
            f.seek(self.offsets[index])
            records = json.loads(f.readline().decode('utf-8'))
        return {key: [record[key] for record in records] for key in records[0]}

    def __getitem__(self, index):
        return upstream_data.tokenize(self.tokenizer, self.group(index), True, self.max_length,
                                      self.mask_source, self.label_smoothing, self.additional_mask)

    def __len__(self):
        return len(self.offsets)


def main():
    upstream_data.TrainDataset = LazyTrainDataset
    sys.argv[0] = 'train.py'
    runpy.run_path('train.py', run_name='__main__')


if __name__ == '__main__':
    main()
