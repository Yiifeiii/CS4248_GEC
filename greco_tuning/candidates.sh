#!/bin/bash
# Stage-1 GRECO training data on the cluster, in one job:
#   P2: BART and T5 n-best candidates, and GECToR's correction, for all clean stage-1 sentences
#   P3: labelled training file, data/greco/stage1.train.jsonl
# Submit from the repository root on the cluster:
#   sbatch -J greco-s1 --time=12:00:00 cluster/job.sbatch greco_tuning/candidates.sh
#
# Before submitting, these must already be on the cluster (see greco_tuning/README.md):
#   - the BART and T5 s1s2 checkpoints and the DeBERTa tokenizer in the scratch Hugging Face cache
#   - gector/setup.sh has been run, so gector/.venv exists
#   - errant, en_core_web_sm, transformers==4.46.3, sentencepiece and protobuf in gec-train
#   - data/greco/stage1.clean.src and data/greco/stage1.clean.tgt copied over
set -euo pipefail

SRC=data/greco/stage1.clean.src
GOLD=data/greco/stage1.clean.tgt
OUT=outputs/greco-s1
mkdir -p "$OUT" data/greco

for pair in bart-base:bart t5-base:t5; do
  model=${pair%%:*}
  name=${pair##*:}
  python bart/generate_nbest.py --model "CS4248-GEC/gec-${model}-s1s2" --input "$SRC" \
    --output "$OUT/$name.nbest.jsonl" --num_beams 10 --num_return 10 --batch_size 64
  python bart/export_candidates.py --nbest "$OUT/$name.nbest.jsonl" --out "$OUT/$name.tgt" --k 5
done

gector/.venv/bin/python gector/predict.py --input "$SRC" --output "$OUT/gector.tgt"

python -m greco_tuning.build_labels \
  --source "$SRC" --gold "$GOLD" \
  --system bart="$OUT/bart.tgt" --system t5="$OUT/t5.tgt" --system gector="$OUT/gector.tgt" \
  --k 5 --tokenizer microsoft/deberta-v3-large --max-length 200 \
  --out data/greco/stage1.train.jsonl
