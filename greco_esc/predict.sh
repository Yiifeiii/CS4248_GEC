#!/bin/bash
# Run one GRECO (+ ESC) combination on one split or file, with the committed ESC models.
# Writes <out>/pred.txt (one corrected sentence per line) and, where the split has public gold,
# <out>/scores.json with ERRANT and M2 precision, recall and F0.5.
#
#   bash greco_esc/predict.sh --split conll14_test --preset best-dev --out outputs/review/greco-esc
#   bash greco_esc/predict.sh --split bea19_test --k 2 --vote_mode system --vote_coef 0.1 --out ...
#   bash greco_esc/predict.sh --split dev_test --k 3 --vote_mode rank --vote_coef 0.4 --no-esc --out ...
#   bash greco_esc/predict.sh --split my_sentences.txt --preset best-dev --out ...   # any tokenized file
#
# --split      dev_tune, dev_test, conll14_test, bea19_test (no public gold: submit pred.txt to
#              Codabench), or a path to a file of tokenized source sentences
# --preset     best-dev (main result), best-bea19 or best-conll14; or set the next four yourself
# --k          hypotheses taken from each of BART and T5: 1, 2, 3 or 5 (default 3)
# --vote_mode  hypothesis, system or rank (default rank); --vote_coef its alpha (default 0.4; 0 = no vote)
# --no-esc     GRECO alone; by default ESC's edit scores are used (score_ratio 0.7)
# --limit N    only the first N sentences, for a quick check
#
# Needs the one-time setup in greco_esc/README.md (gec-train and greco environments, GECToR,
# external/greco with its checkpoint, external/esc). Runs on one GPU; on the SoC cluster:
#   sbatch -J greco-predict --time=3:00:00 cluster/job.sbatch greco_esc/predict.sh --split ... --out ...
# The BART, T5 and GECToR candidates (the slow part) are kept in <out>/pool and reused when the
# command is run again on the same input; use a new --out for a different input.
set -euo pipefail

SPLIT= OUT= K=3 MODE=rank VOTE=0.4 ESC=1 LIMIT= PRESET=
while [ $# -gt 0 ]; do
  case "$1" in
    --split) SPLIT=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --preset) PRESET=$2; shift 2 ;;
    --k) K=$2; shift 2 ;;
    --vote_mode) MODE=$2; shift 2 ;;
    --vote_coef) VOTE=$2; shift 2 ;;
    --no-esc) ESC=0; shift ;;
    --limit) LIMIT=$2; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$SPLIT" ] && [ -n "$OUT" ] || { echo "--split and --out are required (see --help)" >&2; exit 2; }
if [ -n "$PRESET" ]; then
  vals=$(python3 -m greco_esc.predict preset "$PRESET" 2>/dev/null) \
    || { echo "unknown preset $PRESET: best-dev, best-bea19 or best-conll14" >&2; exit 2; }
  read -r K MODE VOTE ESC <<< "$vals"
fi
case "$K" in 1|2|3|5) ;; *) echo "--k must be 1, 2, 3 or 5 (the k ESC was trained for)" >&2; exit 2 ;; esac
case "$MODE" in hypothesis|system|rank) ;; *) echo "--vote_mode must be hypothesis, system or rank" >&2; exit 2 ;; esac

REPO=$PWD
CKPT=${GRECO_CKPT:-$REPO/models/greco/checkpoint.bin}
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
step() { echo "== $(date +%T) $*"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
# run_combination.py ends its output without a final newline
plines() { [ -f "$1" ] && python3 -c "import sys; t = open(sys.argv[1], encoding='utf-8').read(); print(len(t.split(chr(10))) - t.endswith(chr(10)))" "$1" || echo 0; }
use() { set +u; source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"; conda activate "$1"; set -u; }

[ -f external/esc/run.py ] || { echo "external/esc missing; run greco_esc/setup.sh" >&2; exit 1; }
[ -s "$CKPT" ] || { echo "$CKPT missing; run greco_setup/setup.sh" >&2; exit 1; }
read -r BEAM BATCH RATIO < <(python3 -c "from greco_esc import settings as s; print(s.BEAM_SIZE, s.BATCH_SIZE, s.SCORE_RATIO)")
echo "GRECO$([ "$ESC" = 1 ] && echo ' + ESC'), k = $K, $MODE vote $VOTE, on $SPLIT -> $OUT"

# 1. Inputs, candidates and M2 files (gec-train)
use gec-train
N=$(python -m greco_esc.predict inputs "$SPLIT" "$OUT" "$LIMIT" | tail -n 1)
mkdir -p "$OUT/pool"
for name in bart t5; do
  if [ "$(lines "$OUT/pool/$name.tgt")" -ne $((5 * N)) ]; then
    step "$name 5-best"
    model=$(python3 -c "from greco_esc.settings import NBEST; print(NBEST['$name'])")
    python bart/generate_nbest.py --model "$model" --input "$OUT/source.txt" \
      --output "$OUT/pool/$name.nbest.jsonl" --num_beams 10 --num_return 10 --batch_size 64
    python bart/export_candidates.py --nbest "$OUT/pool/$name.nbest.jsonl" --out "$OUT/pool/$name.tgt" --k 5
  fi
done
if [ "$(lines "$OUT/pool/gector.tgt")" -ne "$N" ]; then
  step "GECToR"
  gector/.venv/bin/python gector/predict.py --input "$OUT/source.txt" --output "$OUT/pool/gector.tgt"
fi
step "M2 files for k = $K"
python -m greco_esc.predict m2 "$OUT" "$K"

# 2. ESC edit scores with the committed model (greco)
use greco
extra=()
if [ "$ESC" = 1 ]; then
  model=$(python3 -m greco_esc.predict esc_model "$SPLIT" "$K")
  step "ESC edit scores ($model)"
  python external/esc/run.py --score --data_dir "$OUT/esc/text" --m2_dir "$OUT/esc/m2" \
    --model_path "$model/model.pt" --vocab_path "$model/vocab.idx" --output_path "$OUT/esc.scores.raw.jsonl"
  python -m greco_esc.predict clip "$OUT/esc.scores.raw.jsonl" "$OUT/esc.scores.jsonl" "$N"
  extra=(--edit_scores "$OUT/esc.scores.jsonl" --score_ratio "$RATIO")
fi

# 3. GRECO (combine.py: upstream run_combination.py with the vote modes)
step "GRECO"
inputs=()
for i in $(python3 -m greco_esc.settings inputs "$K"); do inputs+=("$OUT/m2/$i.m2"); done
(cd external/greco && python "$REPO/greco_esc/combine.py" --vote_mode "$MODE" \
  --model greco --lm_model microsoft/deberta-v3-large --checkpoint "$CKPT" \
  --beam_size "$BEAM" --batch_size "$BATCH" --vote_coef "$VOTE" ${extra[@]+"${extra[@]}"} \
  --output_path "$OUT/pred.txt" --data "${inputs[@]}" > "$OUT/greco.log" 2>&1) || {
  tail -n 20 "$OUT/greco.log" >&2; exit 1; }
missing=$(grep -c "edit score for .* is not found" "$OUT/greco.log" || true)
[ "$missing" -eq 0 ] || { echo "$missing edits without an ESC score; see $OUT/greco.log" >&2; exit 1; }
[ "$(plines "$OUT/pred.txt")" -eq "$N" ] || { echo "$OUT/pred.txt: not $N lines" >&2; exit 1; }

# 4. Scores where the split has public gold (gec-train)
if [ -f "$OUT/gold.m2" ]; then
  use gec-train
  step "scoring"
  python -m greco_esc.predict score "$OUT"
else
  echo "no public gold for $SPLIT: $OUT/pred.txt is the output (BEA-2019: python3 -m greco_esc.codabench, or zip it)"
fi
echo "wrote $OUT/pred.txt"
