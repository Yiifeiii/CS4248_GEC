"""Decode an n-best list per source sentence with a GEC seq2seq model.

Writes JSONL: a header line with the decoding parameters, then one object per
source line holding its unique candidates ranked by model log-probability.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import cached_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.common import PREFIX, Postprocessor  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name):
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    # Some BART checkpoints (gotutiyan/gec-bart-base among them) store the tied
    # token embedding only as model.decoder.embed_tokens.weight. This transformers
    # release then leaves model.shared randomly initialised without warning, and
    # the model emits junk. Restore the embedding from the file in that case.
    try:
        path = cached_file(name, "model.safetensors")
    except OSError:
        return model
    stored = "model.decoder.embed_tokens.weight"
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        if "model.shared.weight" not in keys and stored in keys:
            with torch.no_grad():
                model.get_input_embeddings().weight.copy_(f.get_tensor(stored))
    return model


def normalise(text):
    return " ".join(text.split())


def generate_kwargs(args, max_new_tokens):
    kwargs = dict(
        num_beams=args.num_beams,
        num_return_sequences=args.num_return,
        max_new_tokens=max_new_tokens,
        min_length=0,
        early_stopping=True,
        do_sample=False,
        length_penalty=args.length_penalty,
        # The checkpoint config inherits 3 from bart-base's summarisation defaults.
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    if args.mode == "diverse":
        kwargs["num_beam_groups"] = args.num_beam_groups
        kwargs["diversity_penalty"] = args.diversity_penalty
    return kwargs


@torch.no_grad()
def rescore(model, tokenizer, enc, sequences, num_return):
    """Teacher-forced log-probability of each generated sequence.

    generate() scores include the diversity penalty in diverse mode, so they
    are recomputed here to be comparable across decoding modes.
    """
    pad = tokenizer.pad_token_id
    input_ids = enc["input_ids"].repeat_interleave(num_return, dim=0)
    attention_mask = enc["attention_mask"].repeat_interleave(num_return, dim=0)
    # sequences start with decoder_start_token; the rest are the targets.
    decoder_input_ids = sequences[:, :-1]
    targets = sequences[:, 1:]
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    ).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = (targets != pad).float()
    sum_lp = (token_lp * mask).sum(-1)
    avg_lp = sum_lp / mask.sum(-1).clamp(min=1)
    return sum_lp.cpu().tolist(), avg_lp.cpu().tolist()


def build_candidates(src, texts, sum_lps, avg_lps):
    best = {}
    for text, s, a in zip(texts, sum_lps, avg_lps):
        text = normalise(text)
        if not text:
            continue
        if text not in best or a > best[text][1]:
            best[text] = (s, a)
    ranked = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)
    src_norm = normalise(src)
    return [
        {
            "rank": i + 1,
            "text": text,
            "avg_logprob": round(a, 5),
            "sum_logprob": round(s, 5),
            "is_source": text == src_norm,
        }
        for i, (text, (s, a)) in enumerate(ranked)
    ]


def fallback(src):
    return [{"rank": 1, "text": normalise(src), "avg_logprob": None,
             "sum_logprob": None, "is_source": True}]


def main(args):
    if args.mode == "diverse" and args.num_beams % args.num_beam_groups != 0:
        raise ValueError("--num_beams must be divisible by --num_beam_groups")
    if args.num_return > args.num_beams:
        raise ValueError("--num_return cannot exceed --num_beams")

    device = pick_device() if args.device == "auto" else torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model).to(device).eval()
    is_t5 = model.config.model_type == "t5"
    if device.type == "cuda":
        # T5 overflows in fp16
        model.to(torch.bfloat16 if is_t5 else torch.float16)
    prefix = args.prefix if args.prefix is not None else PREFIX.get(model.config.model_type, "")
    # T5 cannot encode some characters (ã, `, < ...); copy them back from the source
    restore = Postprocessor(tokenizer, "t5") if is_t5 else None

    if args.text is not None:
        sources = [args.text]
    else:
        with open(args.input, encoding="utf-8") as f:
            sources = [line.rstrip("\n") for line in f]
        if args.limit:
            sources = sources[:args.limit]

    lengths = [len(tokenizer(prefix + s)["input_ids"]) if s.strip() else 0 for s in sources]
    results = [None] * len(sources)
    todo = []
    skipped_long = 0
    for i, (src, n) in enumerate(zip(sources, lengths)):
        if n == 0:
            results[i] = [] if not src.strip() else fallback(src)
        elif n > args.max_source_tokens:
            results[i] = fallback(src)
            skipped_long += 1
        else:
            todo.append(i)
    # Longest first: memory problems show up at once and the time estimate
    # only gets better as the run goes on.
    todo.sort(key=lambda i: lengths[i], reverse=True)

    # Finished sentences are appended to <output>.partial so a run can be
    # watched (line count) and resumed after an interruption.
    partial = None
    if args.output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        partial_path = args.output + ".partial"
        signature = {k: v for k, v in vars(args).items()
                     if k not in ("output", "batch_size", "max_batch_tokens", "device", "k")}
        if os.path.exists(partial_path):
            with open(partial_path, encoding="utf-8") as f:
                saved = json.loads(f.readline())
                if saved != signature:
                    raise SystemExit(f"{partial_path} was made with different settings; "
                                     f"delete it to start again")
                for line in f:
                    row = json.loads(line)
                    results[row["id"]] = row["cands"]
            todo = [i for i in todo if results[i] is None]
            partial = open(partial_path, "a", encoding="utf-8")
        else:
            partial = open(partial_path, "w", encoding="utf-8")
            partial.write(json.dumps(signature) + "\n")

    total = len(sources)
    done = total - len(todo)
    if partial is not None:
        print(f"{done}/{total} sentences already done, decoding {len(todo)} on {device.type}",
              flush=True)
    tokens_left = sum(lengths[i] for i in todo)
    tokens_done = 0
    started = last_report = time.time()

    pos = 0
    while pos < len(todo):
        # The first sentence is the longest in the batch; shrink the batch for long ones.
        size = max(1, min(args.batch_size, args.max_batch_tokens // lengths[todo[pos]]))
        idx = todo[pos:pos + size]
        pos += size
        batch = [prefix + sources[i] for i in idx]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        max_new = max(lengths[i] for i in idx) + 20
        with torch.no_grad():
            sequences = model.generate(**enc, **generate_kwargs(args, max_new))
        sum_lps, avg_lps = rescore(model, tokenizer, enc, sequences, args.num_return)
        texts = tokenizer.batch_decode(
            sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        n = args.num_return
        for j, i in enumerate(idx):
            sl = slice(j * n, (j + 1) * n)
            outs = [restore(sources[i], t) for t in texts[sl]] if restore else texts[sl]
            cands = build_candidates(sources[i], outs, sum_lps[sl], avg_lps[sl])
            results[i] = cands or fallback(sources[i])
            if partial is not None:
                partial.write(json.dumps({"id": i, "cands": results[i]}, ensure_ascii=False) + "\n")
        if device.type == "mps":
            torch.mps.empty_cache()

        if partial is not None:
            partial.flush()
            done += len(idx)
            tokens_done += sum(lengths[i] for i in idx)
            now = time.time()
            if now - last_report >= args.report_every or pos >= len(todo):
                last_report = now
                eta = (now - started) / tokens_done * (tokens_left - tokens_done)
                print(f"{done}/{total} sentences ({done / total:.0%}), "
                      f"{(now - started) / 60:.1f} min elapsed, about {eta / 60:.1f} min left",
                      flush=True)

    if skipped_long:
        print(f"[warning] {skipped_long} sources exceeded --max_source_tokens "
              f"and were copied unchanged")

    if args.output is None:
        for src, cands in zip(sources, results):
            print(f"SRC   {src}")
            for c in cands[:args.k]:
                score = "" if c["avg_logprob"] is None else f"{c['avg_logprob']:.3f}"
                same = "(unchanged)" if c["is_source"] else ""
                print(f"{c['rank']:>3}  {score:>7}  {c['text']} {same}")
        return

    with open(args.output, "w", encoding="utf-8") as out:
        header = {"header": True, "num_sources": len(sources), **vars(args)}
        out.write(json.dumps(header) + "\n")
        for i, (src, cands) in enumerate(zip(sources, results)):
            out.write(json.dumps({"id": i, "src": src, "cands": cands},
                                 ensure_ascii=False) + "\n")
    partial.close()
    os.remove(partial_path)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gotutiyan/gec-bart-base",
                        help="Hugging Face model name or local checkpoint directory")
    parser.add_argument("--input", help="one tokenised source sentence per line")
    parser.add_argument("--text", help="a single tokenised sentence, instead of --input")
    parser.add_argument("--output", help="n-best JSONL path; omit to print candidates")
    parser.add_argument("--k", type=int, default=5, help="candidates to print when --output is omitted")
    parser.add_argument("--mode", choices=["beam", "diverse"], default="beam")
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--num_return", type=int, default=10)
    parser.add_argument("--num_beam_groups", type=int, default=5, help="diverse mode only")
    parser.add_argument("--diversity_penalty", type=float, default=0.5, help="diverse mode only")
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--prefix", default=None,
                        help='source prefix; default "gec: " for T5 (our fine-tuned models), none for BART')
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16, help="maximum sentences per batch")
    parser.add_argument("--max_batch_tokens", type=int, default=400,
                        help="batch size is reduced so batch_size x longest source stays under this")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--report_every", type=float, default=30,
                        help="seconds between progress lines")
    parser.add_argument("--max_source_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="only decode the first N lines")
    args = parser.parse_args()
    if (args.input is None) == (args.text is None):
        parser.error("give exactly one of --input or --text")
    main(args)
