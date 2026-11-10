#!/bin/bash
# The single BART and T5 models on the CoNLL-2014 and BEA-2019 test sets, without and with
# the curriculum: stage 1 only (*.s1: BEA-19 train) and stage 1 + stage 2 (*.s1s2: then W&I
# train). Decoded with training.evaluate (beam 5), as for the dev scores in
# experiments/training/. Called by cluster/job.sbatch:
#   sbatch -J test-singles --time=02:00:00 cluster/job.sbatch cluster/test_singles.sh
# Writes experiments/test/<run>/{conll14_test,bea19_test}.{pred,scores.json}; score them with
#   python -m greco_setup.test_report
# where git-lfs has pulled data/test.conll14.m2 (it is a pointer file on the cluster).
# Model: $CKPT_ROOT/<run>/best if it exists, else CS4248-GEC/gec-<model>-<stages> from the
# Hugging Face cache (jobs run offline; see cluster/README.md to download it first).
# Runs given as arguments replace the default four, e.g. test_singles.sh t5-base.s1
set -euo pipefail

runs=("$@")
[ ${#runs[@]} -gt 0 ] || runs=(bart-base.s1 bart-base.s1s2 t5-base.s1 t5-base.s1s2)

for run in "${runs[@]}"; do
    model="$CKPT_ROOT/$run/best"
    # bart-base.s1s2 -> CS4248-GEC/gec-bart-base-s1s2
    [ -d "$model" ] || model="CS4248-GEC/gec-${run/./-}"
    out="experiments/test/$run"
    if [ -s "$out/conll14_test.pred" ] && [ -s "$out/bea19_test.pred" ]; then
        echo "== $(date +%T) $run done already, skipped"
        continue
    fi
    echo "== $(date +%T) $run from $model"
    python -m training.evaluate --model "$model" --split conll14_test bea19_test --out "$out"
done
echo "== $(date +%T) predictions in experiments/test/"
