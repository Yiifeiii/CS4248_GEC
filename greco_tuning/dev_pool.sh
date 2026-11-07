#!/bin/bash
# dev_tune candidate pool: BART and T5 5-best, and GECToR 1-best, for data/curriculum/dev_tune.src.
# Ends by writing the M2 inputs for the combiner (make_m2.py), which needs ERRANT in gec-train.
# Submit from the repository root on the cluster:
#   sbatch -J greco-pool --time=3:00:00 cluster/job.sbatch greco_tuning/dev_pool.sh
# Needs the BART and T5 s1s2 checkpoints in the scratch Hugging Face cache, and gector/.venv.
set -euo pipefail

SRC=data/curriculum/dev_tune.src
POOL=greco_tuning/runs/pool
mkdir -p "$POOL"

for pair in bart-base:bart t5-base:t5; do
  model=${pair%%:*}
  name=${pair##*:}
  python bart/generate_nbest.py --model "CS4248-GEC/gec-${model}-s1s2" --input "$SRC" \
    --output "$POOL/$name.nbest.jsonl" --num_beams 10 --num_return 10 --batch_size 64
  python bart/export_candidates.py --nbest "$POOL/$name.nbest.jsonl" --out "$POOL/$name.tgt" --k 5
done

gector/.venv/bin/python gector/predict.py --input "$SRC" --output "$POOL/gector.tgt"

python -m greco_tuning.make_m2
