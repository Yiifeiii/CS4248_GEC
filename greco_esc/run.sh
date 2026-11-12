#!/bin/bash
# The whole GRECO + ESC pipeline as one cluster job. Submit from the repository root:
#   sbatch -J greco-esc --time=10:00:00 cluster/job.sbatch greco_esc/run.sh
#
#   1. Candidates (gec-train): BART and T5 s1s2 5-best, GECToR 1-best, on dev_tune and dev_test
#   2. M2 files (gec-train): one ERRANT M2 per ranked output; ESC's folders for every k
#   3. ESC (greco): train on dev_tune for each k; edit scores and ESC's own output on dev_test.
#      Then the same the other way round (train on dev_test, score dev_tune), for the full dev set
#   4. GRECO (greco): official run_combination.py (through combine.py, which adds the "system" and
#      "rank" vote modes) on dev_test, for every run in settings.py: with and without ESC's edit
#      scores, with no vote and with the tuned vote_coef of each vote mode
#   5. Scores (gec-train): ERRANT and M2 against dev_test.m2, and on the full dev set
#      (dev_tune and dev_test outputs merged in data/dev.src order; greco_esc/evaluate.py)
#
# Every step skips output that is already complete, so a job cut off by its time limit can be
# submitted again. Delete a file to redo its step. Needs the one-time setup in the README.
set -euo pipefail

REPO=$PWD
ROOT=greco_esc/runs
CKPT=$REPO/models/greco/checkpoint.bin
step() { echo "== $(date +%T) $*"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
# run_combination.py ends its output without a final newline
plines() { [ -f "$1" ] && python3 -c "import sys; t = open(sys.argv[1], encoding='utf-8').read(); print(len(t.split(chr(10))) - t.endswith(chr(10)))" "$1" || echo 0; }
use() { set +u; source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"; conda activate "$1"; set -u; }

[ -f external/esc/run.py ] || { echo "external/esc missing; run greco_esc/setup.sh" >&2; exit 1; }
[ -s "$CKPT" ] || { echo "$CKPT missing; run greco_setup/setup.sh" >&2; exit 1; }
grep -q "attention_mask=texts\['attention_mask'\]" external/greco/models.py || {
  echo "GRECO models.py lacks the attention-mask fix; rerun greco_setup/setup.sh" >&2; exit 1; }

# 1-2. Candidates and M2 files
use gec-train
for split in dev_tune dev_test; do
  src=data/curriculum/$split.src
  n=$(lines "$src")
  pool=$ROOT/$split/pool
  mkdir -p "$pool"
  for name in bart t5; do
    if [ "$(lines "$pool/$name.tgt")" -ne $((5 * n)) ]; then
      step "$split: $name 5-best"
      model=$(python3 -c "from greco_esc.settings import NBEST; print(NBEST['$name'])")
      python bart/generate_nbest.py --model "$model" --input "$src" \
        --output "$pool/$name.nbest.jsonl" --num_beams 10 --num_return 10 --batch_size 64
      python bart/export_candidates.py --nbest "$pool/$name.nbest.jsonl" --out "$pool/$name.tgt" --k 5
    fi
  done
  if [ "$(lines "$pool/gector.tgt")" -ne "$n" ]; then
    step "$split: GECToR"
    gector/.venv/bin/python gector/predict.py --input "$src" --output "$pool/gector.tgt"
  fi
  if [ "$(ls "$ROOT/$split/m2" 2>/dev/null | wc -l)" -ne 11 ]; then
    step "$split: M2 files"
    python -m greco_esc.prepare m2 "$split"
  fi
done
python -m greco_esc.prepare esc
# The trained ESC models are committed (greco_esc/esc_weights); put them in place so ESC is
# not retrained. GRECO_ESC_RETRAIN=1 trains from scratch instead.
python -m greco_esc.prepare restore_esc

# 3-4 run on each test split: dev_test (ESC trained on dev_tune) and then dev_tune (ESC trained
# on dev_test), so that the full dev set can be scored without ESC seeing a sentence's gold edits
read -r BEAM BATCH RATIO < <(python3 -c "from greco_esc import settings as s; print(s.BEAM_SIZE, s.BATCH_SIZE, s.SCORE_RATIO)")
use greco
while read -r SPLIT TRAIN <&4; do
  N=$(lines "data/curriculum/$SPLIT.src")

  # 3. ESC
  for k in $(python3 -c "from greco_esc.settings import KS; print(*KS)"); do
    home=$(python3 -m greco_esc.settings esc_home "$k" "$SPLIT")
    out=$ROOT/$SPLIT/k$k
    mkdir -p "$out/esc"
    if [ ! -s "$home/model/model.pt" ]; then
      step "$SPLIT k$k: training ESC on $TRAIN"
      python external/esc/run.py --train --data_dir "$home/$TRAIN/text" --m2_dir "$home/$TRAIN/m2" \
        --model_path "$home/model" --vocab_path "$home/model/vocab.idx"
    fi
    if [ ! -s "$out/esc.scores.jsonl" ]; then
      step "$SPLIT k$k: ESC edit scores"
      python external/esc/run.py --score --data_dir "$home/$SPLIT/text" --m2_dir "$home/$SPLIT/m2" \
        --model_path "$home/model/model.pt" --vocab_path "$home/model/vocab.idx" \
        --output_path "$out/esc.scores.raw.jsonl"
      python -m greco_esc.prepare scores "$k" "$SPLIT"
    fi
    if [ "$(plines "$out/esc/pred.txt")" -ne "$N" ]; then
      step "$SPLIT k$k: ESC alone"
      python external/esc/run.py --test --data_dir "$home/$SPLIT/text" --m2_dir "$home/$SPLIT/m2" \
        --model_path "$home/model/model.pt" --vocab_path "$home/model/vocab.idx" \
        --output_path "$out/esc/pred.txt"
    fi
  done

  # 4. GRECO
  # Read the runs on fd 3, so the programs inside the loop cannot consume the list from stdin
  while read -r name k vote esc mode <&3; do
    dir=$REPO/$ROOT/$SPLIT/k$k/$name
    [ "$(plines "$dir/pred.txt")" -eq "$N" ] && continue
    step "$SPLIT k$k: $name"
    mkdir -p "$dir"
    inputs=()
    for i in $(python3 -m greco_esc.settings inputs "$k"); do inputs+=("$REPO/$ROOT/$SPLIT/m2/$i.m2"); done
    extra=()
    [ "$esc" = 1 ] && extra=(--edit_scores "$REPO/$ROOT/$SPLIT/k$k/esc.scores.jsonl" --score_ratio "$RATIO")
    # combine.py is upstream run_combination.py with the vote counted per --vote_mode
    (cd external/greco && python "$REPO/greco_esc/combine.py" --vote_mode "$mode" \
      --model greco --lm_model microsoft/deberta-v3-large \
      --checkpoint "$CKPT" --beam_size "$BEAM" --batch_size "$BATCH" --vote_coef "$vote" \
      ${extra[@]+"${extra[@]}"} --output_path "$dir/pred.txt" --data "${inputs[@]}" > "$dir/log.txt" 2>&1) || {
      tail -n 20 "$dir/log.txt" >&2; exit 1; }
    missing=$(grep -c "edit score for .* is not found" "$dir/log.txt" || true)
    [ "$missing" -eq 0 ] || { echo "$SPLIT $name k$k: $missing edits without an ESC score" >&2; exit 1; }
  done 3< <(python3 -m greco_esc.settings runs)
done 4< <(python3 -m greco_esc.settings splits)

# 5. Scores: dev_test (the main results) and the full dev set (dev_tune + dev_test)
use gec-train
step "scoring dev_test"
python -m greco_esc.evaluate
step "scoring the full dev set"
python -m greco_esc.evaluate dev
