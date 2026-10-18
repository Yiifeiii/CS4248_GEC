#!/usr/bin/env bash
# Run official GRECO (external/greco/run_combination.py) on the dev30 hypotheses for
# k = 1, 3, 5, with voting (vote_coef 0.4, the paper's value) and without (control).
# Each ranked hypothesis is one input, as in the official code: 3, 7 or 11 inputs.
# Run from the repo root with the GRECO environment active.
set -euo pipefail
dir=$PWD/outputs/greco/bart-t5-gector-dev30-vote
cd external/greco
for vote in 0.4 0; do
    for k in 1 3 5; do
        inputs=("$dir"/m2/gector-1.m2)
        for r in $(seq 1 "$k"); do inputs+=("$dir/m2/bart-$r.m2" "$dir/m2/t5-$r.m2"); done
        out="$dir/vote$vote/k$k"
        mkdir -p "$out"
        echo "== vote_coef=$vote k=$k (${#inputs[@]} inputs)"
        python run_combination.py --model greco --lm_model microsoft/deberta-v3-large \
            --checkpoint "$OLDPWD/models/greco/checkpoint.bin" \
            --beam_size 16 --batch_size 8 --vote_coef "$vote" \
            --output_path "$out/greco.pred" --data "${inputs[@]}"
    done
done
