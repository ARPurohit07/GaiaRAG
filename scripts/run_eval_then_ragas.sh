#!/usr/bin/env bash
# Run the 50-question safety set and a separate 50-answer RAGAS benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_DIR=data/processed/eval

echo "[1/5] Safety evaluation (50 questions)"
python -m gaia_rag.eval.run_eval
SAFETY_RUN=$(ls -t "$EVAL_DIR"/eval-*.json | head -n 1)

echo "[2/5] Answerable supplement (8 questions)"
python -m gaia_rag.eval.run_eval --goldset gaia_rag/eval/goldset/ragas_extra.jsonl
EXTRA_RUN=$(ls -t "$EVAL_DIR"/eval-*.json | head -n 1)

echo "[3/5] Build the 50-answer RAGAS benchmark"
python -m gaia_rag.eval.build_ragas_benchmark --safety-run "$SAFETY_RUN" --extra-run "$EXTRA_RUN"
BENCHMARK=$(ls -t "$EVAL_DIR"/benchmark-50-*.json | head -n 1)

echo "[4/5] RAGAS: four metrics on 50 answers"
.venv-ragas/Scripts/python.exe -m gaia_rag.eval.ragas_eval --run "$BENCHMARK" --all --context-precision --judge qwen3:8b
RAGAS_RUN="$EVAL_DIR/ragas-$(basename "$BENCHMARK")"

echo "[5/5] Verify all four metric averages exceed 0.7"
python -m gaia_rag.eval.verify_ragas --run "$RAGAS_RUN"
