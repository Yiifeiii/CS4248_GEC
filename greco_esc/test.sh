#!/bin/bash
# The final run on the test sets, with every setting fixed on dev_test beforehand (settings.py,
# TEST_RUNS and FINAL_K). Submit from the repository root, after run.sh has finished:
#   sbatch -J greco-test --time=4:00:00 cluster/job.sbatch greco_esc/test.sh
#   sbatch -J greco-test --time=4:00:00 cluster/job.sbatch greco_esc/test.sh conll14   # one set only
#
#   1. Candidates (gec-train): BART and T5 s1s2 5-best, GECToR 1-best, on each test set; M2 files
#   2. ESC (greco): trained once on the full dev set (dev_tune + dev_test; run.sh's dev candidates),
#      then edit scores and ESC's own output on each test set
#   3. GRECO (greco): each run in settings.TEST_RUNS, through combine.py as in run.sh
#   4. Scores (gec-train): CoNLL-2014 with the M2 scorer and ERRANT (needs data/test.conll14.m2
#      pulled from Git LFS); BEA-2019 has no public gold, so its outputs are for Codabench
#
# Nothing here is tuned on a test set. Outputs: greco_esc/runs/<test set>/<tag>/<run>/pred.txt.
# GRECO_CKPT picks another GRECO checkpoint (e.g. the fine-tuned one from greco_tuning), and
# GRECO_TAG names its outputs (default "released"):
#   GRECO_CKPT=$CKPT_ROOT/greco.s1.lr2e-5.seed42/checkpoint-<step>/pytorch_model.bin GRECO_TAG=s1 \
#     sbatch -J greco-test --time=4:00:00 cluster/job.sbatch greco_esc/test.sh
# Every step skips output that is already complete, so the job can be submitted again.
set -euo pipefail

REPO=$PWD
ROOT=greco_esc/runs
CKPT=$(realpath "${GRECO_CKPT:-$REPO/models/greco/checkpoint.bin}")
TAG=${GRECO_TAG:-released}
SETS=("$@")
[ "${#SETS[@]}" -gt 0 ] || SETS=(conll14 bea19)
step() { echo "== $(date +%T) $*"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
# run_combination.py ends its output without a final newline
plines() { [ -f "$1" ] && python3 -c "import sys; t = open(sys.argv[1], encoding='utf-8').read(); print(len(t.split(chr(10))) - t.endswith(chr(10)))" "$1" || echo 0; }
use() { set +u; source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"; conda activate "$1"; set -u; }

[ -f external/esc/run.py ] || { echo "external/esc missing; run greco_esc/setup.sh" >&2; exit 1; }
[ -s "$CKPT" ] || { echo "$CKPT missing" >&2; exit 1; }
grep -q "attention_mask=texts\['attention_mask'\]" external/greco/models.py || {
  echo "GRECO models.py lacks the attention-mask fix; rerun greco_setup/setup.sh" >&2; exit 1; }
# ESC trains on run.sh's dev candidates; both halves must be complete
for half in dev_tune dev_test; do
  [ "$(ls "$ROOT/$half/m2" 2>/dev/null | wc -l)" -eq 11 ] || {
    echo "$ROOT/$half/m2 is incomplete; run greco_esc/run.sh first" >&2; exit 1; }
done
echo "GRECO checkpoint: $CKPT (tag $TAG); test sets: ${SETS[*]}"

# 1. Candidates and M2 files
use gec-train
for set in "${SETS[@]}"; do
  src=$(python3 -m greco_esc.settings source "$set")
  n=$(lines "$src")
  pool=$ROOT/$set/pool
  mkdir -p "$pool"
  for name in bart t5; do
    if [ "$(lines "$pool/$name.tgt")" -ne $((5 * n)) ]; then
      step "$set: $name 5-best"
      model=$(python3 -c "from greco_esc.settings import NBEST; print(NBEST['$name'])")
      python bart/generate_nbest.py --model "$model" --input "$src" \
        --output "$pool/$name.nbest.jsonl" --num_beams 10 --num_return 10 --batch_size 64
      python bart/export_candidates.py --nbest "$pool/$name.nbest.jsonl" --out "$pool/$name.tgt" --k 5
    fi
  done
  if [ "$(lines "$pool/gector.tgt")" -ne "$n" ]; then
    step "$set: GECToR"
    gector/.venv/bin/python gector/predict.py --input "$src" --output "$pool/gector.tgt"
  fi
  if [ "$(ls "$ROOT/$set/m2" 2>/dev/null | wc -l)" -ne 11 ]; then
    step "$set: M2 files"
    python -m greco_esc.prepare m2 "$set"
  fi
done
python -m greco_esc.prepare final_esc "${SETS[@]}"

# 2. ESC, trained on the full dev set
use greco
K=$(python3 -c "from greco_esc.settings import FINAL_K; print(FINAL_K)")
home=$(python3 -m greco_esc.settings esc_home "$K" "${SETS[0]}")
if [ ! -s "$home/model/model.pt" ]; then
  step "k$K: training ESC on the full dev set"
  python external/esc/run.py --train --data_dir "$home/dev/text" --m2_dir "$home/dev/m2" \
    --model_path "$home/model" --vocab_path "$home/model/vocab.idx"
fi
for set in "${SETS[@]}"; do
  N=$(lines "$(python3 -m greco_esc.settings source "$set")")
  out=$ROOT/$set/k$K
  mkdir -p "$out/esc"
  if [ ! -s "$out/esc.scores.jsonl" ]; then
    step "$set k$K: ESC edit scores"
    python external/esc/run.py --score --data_dir "$home/$set/text" --m2_dir "$home/$set/m2" \
      --model_path "$home/model/model.pt" --vocab_path "$home/model/vocab.idx" \
      --output_path "$out/esc.scores.raw.jsonl"
    python -m greco_esc.prepare scores "$K" "$set"
  fi
  if [ "$(plines "$out/esc/pred.txt")" -ne "$N" ]; then
    step "$set k$K: ESC alone"
    python external/esc/run.py --test --data_dir "$home/$set/text" --m2_dir "$home/$set/m2" \
      --model_path "$home/model/model.pt" --vocab_path "$home/model/vocab.idx" \
      --output_path "$out/esc/pred.txt"
  fi

  # 3. GRECO; read the runs on fd 3, so the programs inside the loop cannot consume the list
  read -r BEAM BATCH RATIO < <(python3 -c "from greco_esc import settings as s; print(s.BEAM_SIZE, s.BATCH_SIZE, s.SCORE_RATIO)")
  while read -r name k vote esc mode <&3; do
    dir=$REPO/$ROOT/$set/$TAG/$name
    [ "$(plines "$dir/pred.txt")" -eq "$N" ] && continue
    step "$set: $name ($TAG)"
    mkdir -p "$dir"
    inputs=()
    for i in $(python3 -m greco_esc.settings inputs "$k"); do inputs+=("$REPO/$ROOT/$set/m2/$i.m2"); done
    extra=()
    if [ "$esc" = 1 ]; then
      [ "$k" = "$K" ] || { echo "$name: ESC runs must use k = $K, the full-dev ESC model's k" >&2; exit 1; }
      extra=(--edit_scores "$REPO/$out/esc.scores.jsonl" --score_ratio "$RATIO")
    fi
    (cd external/greco && python "$REPO/greco_esc/combine.py" --vote_mode "$mode" \
      --model greco --lm_model microsoft/deberta-v3-large \
      --checkpoint "$CKPT" --beam_size "$BEAM" --batch_size "$BATCH" --vote_coef "$vote" \
      ${extra[@]+"${extra[@]}"} --output_path "$dir/pred.txt" --data "${inputs[@]}" > "$dir/log.txt" 2>&1) || {
      tail -n 20 "$dir/log.txt" >&2; exit 1; }
    missing=$(grep -c "edit score for .* is not found" "$dir/log.txt" || true)
    [ "$missing" -eq 0 ] || { echo "$set $name: $missing edits without an ESC score" >&2; exit 1; }
  done 3< <(python3 -m greco_esc.settings test_runs)
done

# 4. Scores
use gec-train
for set in "${SETS[@]}"; do
  step "scoring $set"
  python -m greco_esc.evaluate test "$set" "$TAG"
done
