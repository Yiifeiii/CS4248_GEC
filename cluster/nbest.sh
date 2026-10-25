#!/bin/bash
# Top-k n-best lists from a fine-tuned checkpoint for GRECO, in the same JSONL format as
# outputs/bart/*.nbest.jsonl (bart/generate_nbest.py), then the k-lines-per-source .tgt file.
# Called by cluster/job.sbatch:  nbest.sh <run> [input, default data/dev.src] [split name, default dev]
set -euo pipefail

run="$1"
input="${2:-data/dev.src}"
split="${3:-dev}"
out="outputs/$run"
mkdir -p "$out"
python bart/generate_nbest.py --model "$CKPT_ROOT/$run/best" --input "$input" \
    --output "$out/$split.nbest.jsonl" --num_beams 10 --num_return 10 --batch_size 64 --max_batch_tokens 4000
python bart/export_candidates.py --nbest "$out/$split.nbest.jsonl" --out "$out/$split.tgt" --k 5
