#!/bin/bash
# Plain beam search top-5 lists of BART and T5, with the settings of Amitha's dev lists
# (outputs/{bart,t5}/dev.nbest.jsonl): 10 beams, 10 returned, length penalty 1.0, no beam
# groups or diversity penalty. Called by cluster/job.sbatch:
#   sbatch -J nbest-beam --time=04:00:00 cluster/job.sbatch cluster/nbest_beam.sh [input] [name]
# input defaults to data/dev.src and name to full/beam; writes outputs/<system>/<name>.tgt.
# BEA-2019 test set:
#   sbatch -J nbest-bea19 --time=04:00:00 cluster/job.sbatch cluster/nbest_beam.sh data/test.bea19.src bea19
# CoNLL-2014 test set:
#   sbatch -J nbest-conll14 --time=02:00:00 cluster/job.sbatch cluster/nbest_beam.sh data/test.conll14.src conll14
# Model: $BART_MODEL / $T5_MODEL if set, else $CKPT_ROOT/<run>/best if it exists, else
# CS4248-GEC/gec-<system>-base-s1s2 from the Hugging Face cache (jobs run offline).
set -euo pipefail

input="${1:-data/dev.src}"
name="${2:-full/beam}"

for sys in bart t5; do
    run="$sys-base.s1s2"
    override="${sys^^}_MODEL"
    model="${!override:-}"
    if [ -z "$model" ]; then
        model="$CKPT_ROOT/$run/best"
        [ -d "$model" ] || model="CS4248-GEC/gec-$sys-base-s1s2"
    fi
    out="outputs/$sys/$name"
    mkdir -p "$(dirname "$out")"
    echo "== $(date +%T) $sys from $model on $input"
    if [ ! -s "$out.nbest.jsonl" ]; then
        python bart/generate_nbest.py --model "$model" --input "$input" \
            --output "$out.nbest.jsonl" --mode beam --num_beams 10 --num_return 10 \
            --length_penalty 1.0 --batch_size 64 --max_batch_tokens 400
    fi
    python bart/export_candidates.py --nbest "$out.nbest.jsonl" --out "$out.tgt" --k 5
done
