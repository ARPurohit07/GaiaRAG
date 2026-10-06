#!/usr/bin/env bash
# Run the pipeline's own eval, then score the same answers with RAGAS.
#
# Chained because RAGAS reads a completed eval-*.json: it scores the exact
# answers this project's own metrics scored, so the two sets of numbers
# describe identical text rather than two separate generations.
#
# The eval skips its relevancy judge (one less call per case); RAGAS
# measures response relevancy itself, so nothing is lost.
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
LOG_DIR=${1:-.}

echo "=== [1/2] pipeline eval (50 cases) ==="
date +"start %H:%M:%S"
python -u -m gaia_rag.eval.run_eval --no-relevancy > "$LOG_DIR/eval50.log" 2>&1
EVAL_RC=$?
tail -32 "$LOG_DIR/eval50.log"
date +"end   %H:%M:%S"

# A non-zero exit means some cases failed, which is a result, not a
# breakage: RAGAS should still score whatever was produced. Only a
# missing output file is fatal.
if ! ls data/processed/eval/eval-*.json >/dev/null 2>&1; then
  echo "FATAL: eval produced no output; not running RAGAS"
  exit 1
fi
echo "(eval exit code $EVAL_RC)"

echo
echo "=== [2/2] RAGAS on the same answers ==="
date +"start %H:%M:%S"
.venv-ragas/Scripts/python.exe -u -m gaia_rag.eval.ragas_eval --all \
  > "$LOG_DIR/ragas50.log" 2>&1
RAGAS_RC=$?
tail -40 "$LOG_DIR/ragas50.log"
date +"end   %H:%M:%S"
echo "(ragas exit code $RAGAS_RC)"
