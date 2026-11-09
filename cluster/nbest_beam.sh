#!/bin/bash
# Plain beam search top-5 lists of BART and T5 on dev, for comparison with the diverse
# lists in outputs/{bart,t5}/full/diverse.tgt. Same models and settings as those
# (10 beams, 10 returned, length penalty 1.0, batch 64 / 400 tokens), only without beam
# groups and diversity penalty. Called by cluster/job.sbatch:
#   sbatch -J nbest-beam --time=04:00:00 cluster/job.sbatch cluster/nbest_beam.sh
# Uses the local checkpoint $CKPT_ROOT/<run>/best when it exists, else the model from
# the Hugging Face cache (CS4248-GEC/gec-<system>-base-s1s2).
set -euo pipefail

for sys in bart t5; do
    run="$sys-base.s1s2"
    model="$CKPT_ROOT/$run/best"
    [ -d "$model" ] || model="CS4248-GEC/gec-$sys-base-s1s2"
    out="outputs/$sys/full"
    mkdir -p "$out"
    echo "== $(date +%T) $sys from $model"
    if [ ! -s "$out/beam.nbest.jsonl" ]; then
        python bart/generate_nbest.py --model "$model" --input data/dev.src \
            --output "$out/beam.nbest.jsonl" --mode beam --num_beams 10 --num_return 10 \
            --length_penalty 1.0 --batch_size 64 --max_batch_tokens 400
    fi
    python bart/export_candidates.py --nbest "$out/beam.nbest.jsonl" --out "$out/beam.tgt" --k 5
done
