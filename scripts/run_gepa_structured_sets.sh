#!/usr/bin/env bash
# Run GEPA on structured splits (time-weighted Seattle, n=400) with GPT-5 mini/nano, 50 steps.
# Usage: ./scripts/run_gepa_structured_sets.sh
# Assumes: OPENROUTER_API_KEY set, metadata updated, and Concorde built.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_DIR="$ROOT/runs/gepa_structured"
MODEL_STUDENT="openai/gpt-5-nano"
MODEL_REFLECTOR="openai/gpt-5-mini"
STEPS=50
REPEATS=1
TIMEOUT=""

splits=(
  "structured_seattle_time"
)

mkdir -p "$SAVE_DIR"

for split in "${splits[@]}"; do
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  label="${split}_n400_time"
  run_dir="$SAVE_DIR/${timestamp}_${label}"

  echo "=== Running GEPA on $split -> $run_dir ==="
  python "$ROOT/scripts/run_gepa.py" \
    --student-model "$MODEL_STUDENT" \
    --reflector-model "$MODEL_REFLECTOR" \
    --split "$split" \
    --steps "$STEPS" \
    --repeats "$REPEATS" \
    ${TIMEOUT:+--timeout "$TIMEOUT"} \
    --label-prefix "$label" \
    --save-dir "$SAVE_DIR"
done
