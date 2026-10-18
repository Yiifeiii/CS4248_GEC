"""Correct sentences with Grammarly's official pretrained GECToR (RoBERTa).

Run from the repository root with the gector/.venv environment (see gector/README.md):

    gector/.venv/bin/python gector/predict.py \
        --input data/dev.src --output outputs/gector/dev.tgt

The official checkpoint was trained with AllenNLP 0.8.4 / Python 3.7. We load it
with gotutiyan/gector, a PyTorch port that renames the official parameters and
reorders the official tag vocabulary to match its own model class.
"""

import argparse
from pathlib import Path

import torch
from gector import GECToR, load_verb_dict, predict
from transformers import AutoTokenizer

OFFICIAL = Path(__file__).resolve().parent / "models"


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_official(checkpoint, vocab_dir, transformer, special_tokens_fix, max_length):
    # The checkpoint was saved from a CUDA GPU and the port calls torch.load without
    # map_location, which fails on machines without CUDA. Load onto the CPU first.
    torch_load = torch.load
    torch.load = lambda *a, **kw: torch_load(*a, **{"map_location": "cpu", **kw})
    try:
        model = GECToR.from_official_pretrained(
            str(checkpoint),
            special_tokens_fix=special_tokens_fix,
            transformer_model=transformer,
            vocab_path=str(vocab_dir),
            max_length=max_length,
        ).eval()
    finally:
        torch.load = torch_load
    # transformers 5 drops max_length from config kwargs; the port reads it at predict time
    model.config.max_length = max_length
    # RoBERTa's tokenizer needs add_prefix_space=True to accept pre-split words
    tokenizer = AutoTokenizer.from_pretrained(transformer, add_prefix_space="roberta" in transformer)
    if special_tokens_fix:
        # The RoBERTa checkpoint was trained with $START as its own token
        tokenizer.add_special_tokens({"additional_special_tokens": ["$START"]})
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="tokenized sentences, one per line")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL / "roberta_1_gectorv2.th")
    parser.add_argument("--vocab_dir", type=Path, default=OFFICIAL / "output_vocabulary")
    parser.add_argument("--verb_file", type=Path, default=OFFICIAL / "verb-form-vocab.txt")
    parser.add_argument("--transformer", default="roberta-base")
    parser.add_argument("--special_tokens_fix", type=int, default=1, help="1 for RoBERTa, 0 for BERT/XLNet")
    parser.add_argument("--max_length", type=int, default=80, help="subwords per sentence; the rest are copied")
    # Grammarly's recommended inference tweaks for the RoBERTa model
    parser.add_argument("--keep_confidence", type=float, default=0.2)
    parser.add_argument("--min_error_prob", type=float, default=0.5)
    parser.add_argument("--n_iteration", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    device = get_device(args.device)
    model, tokenizer = load_official(
        args.checkpoint, args.vocab_dir, args.transformer, args.special_tokens_fix, args.max_length
    )
    model.to(device)
    encode, decode = load_verb_dict(str(args.verb_file))

    sources = args.input.read_text(encoding="utf-8").rstrip("\n").split("\n")
    with torch.no_grad():
        corrected = predict(
            model, tokenizer, sources, encode, decode,
            keep_confidence=args.keep_confidence,
            min_error_prob=args.min_error_prob,
            batch_size=args.batch_size,
            n_iteration=args.n_iteration,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(corrected) + "\n", encoding="utf-8")
    print(f"Wrote {len(corrected)} corrected sentences to {args.output} (device: {device})")


if __name__ == "__main__":
    main()
