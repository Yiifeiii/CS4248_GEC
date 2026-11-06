"""Helpers shared by the fine-tuning, evaluation and data scripts.

Models read and write text tokenized like the corpora ("It 's ... minds ."), so
no retokenization is needed. T5 has two quirks, both handled here:
- it gets the task prefix PREFIX["t5"] on every source sentence;
- its SentencePiece vocabulary cannot encode 45 characters in our data (` ^ ~ < { }
  and accented letters such as ã í ñ) and NFKC-normalizes others (… -> ..., º -> o),
  so restore_from_source copies the original tokens back from the source.
"""

from contextlib import nullcontext
from difflib import SequenceMatcher
import os
from pathlib import Path
import subprocess

import torch

# Tokenizers are used before git/errant subprocesses are forked
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / 'data' / 'curriculum'
PREFIX = {'bart': '', 't5': 'gec: '}
# Same limit for both models, measured in T5 tokens (the longer tokenization)
MAX_TOKENS = 128


def family(config):
    """'bart' or 't5' from a transformers config."""
    if config.model_type not in PREFIX:
        raise ValueError(f'unsupported model type {config.model_type}')
    return config.model_type


def expand(path):
    """Expand ${VARS} and ~ in a config path."""
    return os.path.expanduser(os.path.expandvars(str(path)))


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(line + '\n' for line in lines))


def generation_kwargs(num_beams, max_new_tokens):
    # bart-base inherits no_repeat_ngram_size=3 from summarization, which forbids
    # copying any repeated trigram of the source; GEC must copy freely.
    return dict(num_beams=num_beams, max_new_tokens=max_new_tokens, min_length=0,
                no_repeat_ngram_size=0, length_penalty=1.0, early_stopping=True, do_sample=False)


def restore_from_source(source, output, view):
    """Put back source tokens that the model could only reproduce in a lossy form.

    view(token) is the token as the model sees it (T5: decode(encode(token))), so
    'São' -> 'So' and '<' -> ''. Output tokens aligned with the views of unchanged
    source tokens are replaced by the original tokens; source tokens with an empty
    view inside such aligned stretches are reinserted.
    """
    src = source.split()
    out = output.split()
    keep = [i for i, token in enumerate(src) if view(token)]
    views = [view(src[i]) for i in keep]
    if all(view(token) == token for token in src):
        return ' '.join(out)
    result = []
    blocks = SequenceMatcher(None, views, out, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in blocks:
        if tag != 'equal':
            result.extend(out[j1:j2])
            continue
        first = 0 if i1 == 0 else keep[i1]
        last = len(src) if i2 == len(keep) else keep[i2 - 1] + 1
        result.extend(src[first:last])
    return ' '.join(result)


class Postprocessor:
    """Turn decoded model text into a corpus-tokenized correction of its source."""

    def __init__(self, tokenizer, model_family):
        self.tokenizer = tokenizer
        self.family = model_family
        self.views = {}

    def view(self, token):
        if token not in self.views:
            ids = self.tokenizer(token, add_special_tokens=False)['input_ids']
            self.views[token] = ' '.join(self.tokenizer.decode(ids, skip_special_tokens=True).split())
        return self.views[token]

    def __call__(self, source, text):
        text = ' '.join(text.split())
        if self.family == 't5':
            text = restore_from_source(source, text, self.view)
        return text


@torch.no_grad()
def decode(model, tokenizer, sources, num_beams=5, batch_size=64, max_batch_tokens=6000,
           num_return=1, prefix=None, post=None, progress=None):
    """Return num_return corrections per source (a flat list when num_return == 1).

    prefix and post default to the model family's (PREFIX, Postprocessor).
    Batches are built longest first so memory errors show up immediately.
    """
    model_family = family(model.config)
    prefix = PREFIX[model_family] if prefix is None else prefix
    post = post or Postprocessor(tokenizer, model_family)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    lengths = [len(tokenizer(prefix + s)['input_ids']) for s in sources]
    order = sorted(range(len(sources)), key=lambda i: -lengths[i])
    results = [None] * len(sources)
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    pos = 0
    while pos < len(order):
        size = max(1, min(batch_size, max_batch_tokens // (lengths[order[pos]] * num_beams)))
        idx = order[pos:pos + size]
        pos += size
        enc = tokenizer([prefix + sources[i] for i in idx], return_tensors='pt', padding=True,
                        truncation=True, max_length=512).to(device)
        with autocast:
            out = model.generate(**enc, num_return_sequences=num_return,
                                 **generation_kwargs(num_beams, max(lengths[i] for i in idx) + 32))
        texts = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for j, i in enumerate(idx):
            cands = [post(sources[i], t) for t in texts[j * num_return:(j + 1) * num_return]]
            results[i] = cands[0] if num_return == 1 else cands
        if progress:
            progress(len(sources) - (len(order) - pos), len(sources))
    if was_training:
        model.train()
    return results

