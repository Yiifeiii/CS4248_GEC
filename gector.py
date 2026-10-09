"""
GECToR: Grammatical Error Correction by Tagging (first draft)
=============================================================

Paper: Omelianchuk et al. (2020), "GECToR - Grammatical Error Correction: Tag, Not Rewrite".

The core idea
-------------
A seq2seq model (e.g. T5) rewrites the whole sentence. GECToR does not. It
predicts one EDIT TAG for every input word, then applies those tags:

    source :  $START  I     saws          the    show  of            a     wall
    tags   :  $KEEP   $KEEP $REPLACE_saw  $KEEP  $KEEP $REPLACE_on   $KEEP $KEEP
    output :          I     saw           the    show  on            a     wall

Because it is a tagger (an encoder + a linear layer), it is fast and needs no decoder.
Some errors need more than one edit, so at prediction time we tag -> apply ->
tag again, a few rounds ("iterative correction").

The script is organised in the same order as the pipeline:

    STEP 1  Edit tags: what the model can predict
    STEP 2  Turn (source, target) sentence pairs into per-word tags
    STEP 3  Build the tag vocabulary
    STEP 4  Dataset + batching (words -> subword tokens)
    STEP 5  The model: transformer encoder + two linear heads
    STEP 6  Training loop
    STEP 7  Evaluation on dev
    STEP 8  Iterative prediction (correcting new sentences)
    STEP 9  Command-line entry point

Usage
-----
    # quick sanity run on 2,000 sentences
    python gector.py train --max_train_samples 2000 --epochs 1 --out_dir models/gector_debug

    # full training
    python gector.py train --out_dir models/gector

    # correct the dev / test sets with a trained model
    python gector.py predict --model_dir models/gector --input data/dev.src --output outputs/dev.pred
    python gector.py predict --model_dir models/gector --input data/test.conll14.src --output outputs/conll14.pred

Simplifications compared to the original GECToR (good things to add later):
  * Only 4 tag types (KEEP / DELETE / REPLACE_w / APPEND_w). The original also has
    "g-transformations" such as $TRANSFORM_CASE_CAPITAL or $TRANSFORM_VERB_VB_VBZ,
    which cover many errors with far fewer tags.
  * One training stage. The original trains in 3 stages: synthetic data ->
    errorful sentences only -> fine-tune on W&I+LOCNESS including correct sentences.
  * When a word needs two edits, we keep only the first one as its training tag.
"""

import argparse
import difflib
import json
import os
import random
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


# =============================================================================
# STEP 1. Edit tags
# =============================================================================
# Every word in the sentence gets exactly one of these tags:
#   $KEEP        leave the word unchanged
#   $DELETE      remove the word
#   $REPLACE_w   replace the word with w            ("saws" -> $REPLACE_saw)
#   $APPEND_w    keep the word and insert w after it ("went" -> $APPEND_to)
#
# To be able to insert a word at the very beginning of a sentence, we put a
# dummy token $START in front of every sentence. "$START -> $APPEND_The" adds
# "The" as the first word.

START = "$START"
KEEP = "$KEEP"
DELETE = "$DELETE"
REPLACE = "$REPLACE_"
APPEND = "$APPEND_"

# The second head of the model is an error DETECTOR: is this word correct or not?
CORRECT, INCORRECT = 0, 1


# =============================================================================
# STEP 2. From (source, target) pairs to per-word tags
# =============================================================================
# Our data gives us whole corrected sentences (train.tgt), not tags. We recover
# the tags by aligning source and target words with difflib, which reports
# which word spans are equal / deleted / inserted / replaced.

def make_tags(src_words, tgt_words):
    """Return one tag per word of [$START] + src_words.

    Example:
        src = I saws the show
        tgt = I saw the show
        ->   [$KEEP, $KEEP, $REPLACE_saw, $KEEP, $KEEP]
              $START  I      saws          the    show
    """
    # candidates[i] collects every edit that word i needs (index 0 is $START)
    candidates = [[] for _ in range(len(src_words) + 1)]

    matcher = difflib.SequenceMatcher(None, src_words, tgt_words, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        # i1:i2 index src_words, so the same word is at position i+1 in candidates
        if op == "equal":
            continue

        elif op == "delete":
            # src words i1..i2 do not appear in the target -> delete each
            for i in range(i1, i2):
                candidates[i + 1].append(DELETE)

        elif op == "insert":
            # target words j1..j2 are missing before src word i1, i.e. they come
            # after candidates position i1 (which is $START when i1 == 0).
            # We only tag the first missing word; if several words are missing,
            # the next correction round inserts the rest.
            candidates[i1].append(APPEND + tgt_words[j1])

        elif op == "replace":
            # Pair up src and tgt words one by one:
            #   extra src words -> $DELETE
            #   extra tgt words -> $APPEND on the last src word of the span
            n_src, n_tgt = i2 - i1, j2 - j1
            for k in range(min(n_src, n_tgt)):
                candidates[i1 + k + 1].append(REPLACE + tgt_words[j1 + k])
            for k in range(n_tgt, n_src):
                candidates[i1 + k + 1].append(DELETE)
            if n_tgt > n_src:
                candidates[i2].append(APPEND + tgt_words[j1 + n_src])

    # A word can only carry one tag, so keep the first edit (or $KEEP).
    return [c[0] if c else KEEP for c in candidates]


def apply_tags(words, tags):
    """Apply one tag per word of [$START] + words and return the new word list.

    This is the inverse of make_tags and is what we use at prediction time.
    """
    out = []
    for i, tag in enumerate(tags):
        if i == 0:
            # $START is not a real word: it may only append
            if tag.startswith(APPEND):
                out.append(tag[len(APPEND):])
            continue
        word = words[i - 1]
        if tag == DELETE:
            continue
        elif tag.startswith(REPLACE):
            out.append(tag[len(REPLACE):])
        elif tag.startswith(APPEND):
            out.extend([word, tag[len(APPEND):]])
        else:  # $KEEP
            out.append(word)
    return out


def read_parallel(src_path, tgt_path, max_samples=None):
    with open(src_path, encoding="utf-8") as f_src, open(tgt_path, encoding="utf-8") as f_tgt:
        pairs = [(s.split(), t.split()) for s, t in zip(f_src, f_tgt)]
    if max_samples is not None:
        pairs = pairs[:max_samples]
    return pairs


# =============================================================================
# STEP 3. Tag vocabulary
# =============================================================================
# The model is a classifier, so it needs a fixed list of tags. There are many
# REPLACE_w / APPEND_w tags (one per word w), but most are rare. Like the
# original paper, we keep the most frequent ones (default 5000).
# Rarer edits are mapped to $KEEP: the model simply won't learn to make them.

def build_tag_vocab(tag_sequences, max_size):
    counts = Counter(t for tags in tag_sequences for t in tags if t != KEEP)
    return [KEEP] + [t for t, _ in counts.most_common(max_size - 1)]


# =============================================================================
# STEP 4. Dataset and batching
# =============================================================================
# Problem: we tag WORDS, but transformers read SUBWORD tokens
# ("disappointment" -> "disappoint", "ment"). The standard fix (also used by
# GECToR) is to represent each word by the vector of its FIRST subword.
#
# For each sentence the collate function therefore produces:
#   input_ids, attention_mask  subword tokens for the encoder
#   word_index[b, w]           position of the first subword of word w
#   tag_ids[b, w]              gold tag of word w      (-100 = ignore in loss)
#   detect_ids[b, w]           gold correct/incorrect  (-100 = ignore in loss)

class GECDataset(Dataset):
    def __init__(self, pairs, tag2id, max_words):
        self.items = []
        keep_id = tag2id[KEEP]
        for src, tgt in pairs:
            tags = make_tags(src, tgt)
            words = [START] + src
            # Long sentences are truncated to keep batches small
            words, tags = words[:max_words], tags[:max_words]
            tag_ids = [tag2id.get(t, keep_id) for t in tags]
            detect_ids = [CORRECT if t == KEEP else INCORRECT for t in tags]
            self.items.append((words, tag_ids, detect_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def encode_words(tokenizer, batch_words, max_subwords):
    """Tokenize lists of words and find the first subword of every word."""
    enc = tokenizer(
        batch_words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_subwords,
        padding=True,
        return_tensors="pt",
    )
    max_words = max(len(w) for w in batch_words)
    word_index = torch.zeros(len(batch_words), max_words, dtype=torch.long)
    word_mask = torch.zeros(len(batch_words), max_words, dtype=torch.bool)
    for b in range(len(batch_words)):
        seen = set()
        for pos, w in enumerate(enc.word_ids(batch_index=b)):
            if w is not None and w not in seen:
                seen.add(w)
                word_index[b, w] = pos
                word_mask[b, w] = True
    # Words cut off by subword truncation keep word_mask = False
    return enc["input_ids"], enc["attention_mask"], word_index, word_mask


def make_collate_fn(tokenizer, max_subwords):
    def collate(batch):
        batch_words = [words for words, _, _ in batch]
        input_ids, attention_mask, word_index, word_mask = encode_words(
            tokenizer, batch_words, max_subwords
        )
        tag_ids = torch.full(word_index.shape, -100, dtype=torch.long)
        detect_ids = torch.full(word_index.shape, -100, dtype=torch.long)
        for b, (_, tags, detects) in enumerate(batch):
            tag_ids[b, : len(tags)] = torch.tensor(tags)
            detect_ids[b, : len(detects)] = torch.tensor(detects)
        # Do not train on words whose subwords were truncated away
        tag_ids[~word_mask] = -100
        detect_ids[~word_mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_index": word_index,
            "tag_ids": tag_ids,
            "detect_ids": detect_ids,
        }

    return collate


# =============================================================================
# STEP 5. The model
# =============================================================================
#   subword tokens -> transformer encoder (e.g. RoBERTa) -> one vector per subword
#   pick the first-subword vector of each word           -> one vector per word
#   tag head    : Linear(hidden -> number of tags)        -> which edit?
#   detect head : Linear(hidden -> 2)                     -> is this word wrong?
#
# The detection head is an auxiliary task: it helps training and, at prediction
# time, lets us skip sentences the model thinks are already correct.

class GECToR(nn.Module):
    def __init__(self, encoder, num_tags, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(hidden, num_tags)
        self.detect_head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, word_index):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # hidden: (batch, n_subwords, H) -> word_hidden: (batch, n_words, H)
        index = word_index.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        word_hidden = self.dropout(torch.gather(hidden, 1, index))
        return self.tag_head(word_hidden), self.detect_head(word_hidden)


def save_model(model, tokenizer, tags, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    model.encoder.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "tags.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(tags) + "\n")
    with open(os.path.join(out_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "tags.txt"), encoding="utf-8") as f:
        tags = f.read().split("\n")[:-1]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Build an empty encoder from the saved config, then load the trained weights
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
    model = GECToR(encoder, len(tags))
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer, tags


def load_tokenizer(name):
    # RoBERTa's tokenizer needs add_prefix_space=True to accept pre-split words
    try:
        return AutoTokenizer.from_pretrained(name, add_prefix_space=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(name)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# STEP 6. Training
# =============================================================================
# Loss = cross-entropy(tag head) + cross-entropy(detect head), over real words.
#
# "Cold" epochs (from the paper): the tag and detect heads start random, so for
# the first epoch(s) we freeze the pretrained encoder and train only the heads
# with a larger learning rate. This avoids damaging the encoder with large,
# noisy gradients coming from the random heads.

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # 6a. Read data
    train_pairs = read_parallel(args.train_src, args.train_tgt, args.max_train_samples)
    dev_pairs = read_parallel(args.dev_src, args.dev_tgt, args.max_dev_samples)
    print(f"Train pairs: {len(train_pairs):,}   Dev pairs: {len(dev_pairs):,}")

    # 6b. Tags and vocabulary (from TRAINING data only)
    t0 = time.time()
    tags = build_tag_vocab((make_tags(s, t) for s, t in train_pairs), args.tag_vocab_size)
    tag2id = {t: i for i, t in enumerate(tags)}
    print(f"Tag vocabulary: {len(tags)} tags ({time.time() - t0:.0f}s). Most frequent: {tags[:10]}")

    # 6c. Datasets and loaders
    tokenizer = load_tokenizer(args.model_name)
    collate = make_collate_fn(tokenizer, args.max_subwords)
    train_set = GECDataset(train_pairs, tag2id, args.max_words)
    dev_set = GECDataset(dev_pairs, tag2id, args.max_words)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 6d. Model
    encoder = AutoModel.from_pretrained(args.model_name)
    model = GECToR(encoder, len(tags), dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    head_params = list(model.tag_head.parameters()) + list(model.detect_head.parameters())
    cold_optimizer = torch.optim.AdamW(head_params, lr=args.cold_lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 6e. Epoch loop
    best_f05 = -1.0
    for epoch in range(args.epochs):
        cold = epoch < args.cold_epochs
        model.encoder.requires_grad_(not cold)
        opt = cold_optimizer if cold else optimizer
        model.train()
        print(f"\nEpoch {epoch + 1}/{args.epochs}" + (" (cold: encoder frozen)" if cold else ""))

        running, t0 = 0.0, time.time()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            tag_logits, detect_logits = model(batch["input_ids"], batch["attention_mask"], batch["word_index"])
            # Flatten (batch, words, classes) -> (batch*words, classes) for cross-entropy
            loss = loss_fn(tag_logits.flatten(0, 1), batch["tag_ids"].flatten()) + loss_fn(
                detect_logits.flatten(0, 1), batch["detect_ids"].flatten()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            if step % args.log_every == 0:
                speed = step / (time.time() - t0)
                print(f"  step {step}/{len(train_loader)}  loss {running / args.log_every:.4f}  ({speed:.1f} steps/s)")
                running = 0.0

        # 6f. Evaluate on dev and keep the best model
        metrics = evaluate(model, dev_loader, tag2id[KEEP], device)
        print(f"  dev: loss {metrics['loss']:.4f}  tag P {metrics['p']:.3f}  R {metrics['r']:.3f}  F0.5 {metrics['f05']:.3f}")
        if metrics["f05"] > best_f05:
            best_f05 = metrics["f05"]
            save_model(model, tokenizer, tags, args, args.out_dir)
            print(f"  saved best model to {args.out_dir}")


# =============================================================================
# STEP 7. Evaluation (a quick proxy, computed on tags)
# =============================================================================
# The real metrics are ERRANT (BEA-19) and the M2 scorer (CoNLL-14), which work
# on corrected sentences. They are slow to run every epoch, so during training
# we compute precision/recall/F0.5 on the TAGS instead:
#   true positive  : model proposes an edit and it is the gold edit
#   false positive : model proposes an edit that is not the gold one
#   false negative : gold has an edit and the model did not predict it
# F0.5 weighs precision twice as much as recall, like the shared tasks.

@torch.no_grad()
def evaluate(model, loader, keep_id, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, n_batches, tp, fp, fn = 0.0, 0, 0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tag_logits, detect_logits = model(batch["input_ids"], batch["attention_mask"], batch["word_index"])
        total_loss += (
            loss_fn(tag_logits.flatten(0, 1), batch["tag_ids"].flatten())
            + loss_fn(detect_logits.flatten(0, 1), batch["detect_ids"].flatten())
        ).item()
        n_batches += 1

        gold = batch["tag_ids"]
        pred = tag_logits.argmax(-1)
        valid = gold != -100
        pred_edit = valid & (pred != keep_id)
        gold_edit = valid & (gold != keep_id)
        tp += (pred_edit & (pred == gold)).sum().item()
        fp += (pred_edit & (pred != gold)).sum().item()
        fn += (gold_edit & (pred != gold)).sum().item()

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    model.train()
    return {"loss": total_loss / max(n_batches, 1), "p": p, "r": r, "f05": f05}


# =============================================================================
# STEP 8. Iterative prediction
# =============================================================================
# For each round:
#   1. tag every word of every still-active sentence
#   2. add a bonus to $KEEP's probability (keep_confidence) -> fewer, safer edits
#   3. skip the sentence if the detector's highest "incorrect" probability is
#      below min_error_prob (the model thinks the sentence is fine)
#   4. apply the remaining edits; if nothing changed, the sentence is done
# Repeating lets the model fix errors that need several edits, e.g. inserting
# two missing words.

@torch.no_grad()
def correct_sentences(model, tokenizer, tags, sentences, device, iterations=5,
                      keep_confidence=0.0, min_error_prob=0.0, batch_size=64,
                      max_words=128, max_subwords=256):
    keep_id = tags.index(KEEP)
    current = [s.split() for s in sentences]
    active = list(range(len(current)))

    for _ in range(iterations):
        still_active = []
        for start in range(0, len(active), batch_size):
            idxs = active[start:start + batch_size]
            # Only the first max_words words are tagged; the rest are copied unchanged
            batch_words = [[START] + current[i][: max_words - 1] for i in idxs]
            input_ids, attention_mask, word_index, word_mask = encode_words(tokenizer, batch_words, max_subwords)
            tag_logits, detect_logits = model(input_ids.to(device), attention_mask.to(device), word_index.to(device))

            tag_probs = tag_logits.softmax(-1).cpu()
            tag_probs[..., keep_id] += keep_confidence
            error_probs = detect_logits.softmax(-1)[..., INCORRECT].cpu()

            for b, i in enumerate(idxs):
                n = len(batch_words[b])
                mask = word_mask[b, :n]
                if not mask.any() or error_probs[b, :n][mask].max() < min_error_prob:
                    continue  # sentence looks correct
                best_prob, best_tag = tag_probs[b, :n].max(-1)
                predicted = [
                    tags[t] if (mask[w] and best_prob[w] >= min_error_prob) else KEEP
                    for w, t in enumerate(best_tag.tolist())
                ]
                if all(t == KEEP for t in predicted):
                    continue
                tagged_part = current[i][: max_words - 1]
                current[i] = apply_tags(tagged_part, predicted) + current[i][max_words - 1:]
                still_active.append(i)
        active = still_active
        if not active:
            break

    return [" ".join(words) for words in current]


def predict(args):
    device = get_device()
    model, tokenizer, tags = load_model(args.model_dir, device)
    with open(args.input, encoding="utf-8") as f:
        sentences = [line.rstrip("\n") for line in f]
    corrected = correct_sentences(
        model, tokenizer, tags, sentences, device,
        iterations=args.iterations,
        keep_confidence=args.keep_confidence,
        min_error_prob=args.min_error_prob,
        batch_size=args.batch_size,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(corrected) + "\n")
    print(f"Wrote {len(corrected)} corrected sentences to {args.output}")


# =============================================================================
# STEP 9. Command line
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GECToR: train or predict")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train_src", default="data/train.src")
    t.add_argument("--train_tgt", default="data/train.tgt")
    t.add_argument("--dev_src", default="data/dev.src")
    t.add_argument("--dev_tgt", default="data/dev.tgt")
    t.add_argument("--out_dir", default="models/gector")
    t.add_argument("--model_name", default="roberta-base", help="any Hugging Face encoder")
    t.add_argument("--tag_vocab_size", type=int, default=5000)
    t.add_argument("--max_words", type=int, default=64, help="truncate training sentences to this many words")
    t.add_argument("--max_subwords", type=int, default=128)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--cold_epochs", type=int, default=1, help="epochs with the encoder frozen")
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--cold_lr", type=float, default=1e-3)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--max_train_samples", type=int, default=None, help="use only the first N pairs (debugging)")
    t.add_argument("--max_dev_samples", type=int, default=None)
    t.add_argument("--log_every", type=int, default=100)
    t.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("predict")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--input", required=True, help="tokenized sentences, one per line")
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--keep_confidence", type=float, default=0.0, help="bonus added to $KEEP probability")
    p.add_argument("--min_error_prob", type=float, default=0.0, help="minimum probability to apply an edit")
    p.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
