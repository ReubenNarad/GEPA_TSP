#!/usr/bin/env bash
# Run GEPA on the Seattle travel-time split (n=400) with higher repeats:
#   - baseline repeats: 10
#   - candidate repeats: 5
# Usage: ./scripts/run_gepa_seattle_time_highrepeats.sh
# Assumes: OPENROUTER_API_KEY set, metadata updated, Concorde built.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_DIR="$ROOT/runs/gepa_structured"
MODEL_STUDENT="openai/gpt-5-nano"
MODEL_REFLECTOR="openai/gpt-5"
STEPS=50
REPEATS=5          # per-candidate repeats
BASELINE_REPEATS=10
TIMEOUT=""

split="structured_seattle_time"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
label="${split}_n400_time_hreps"
run_dir="$SAVE_DIR/${timestamp}_${label}"

echo "=== Running GEPA on $split -> $run_dir (baseline repeats=$BASELINE_REPEATS, candidate repeats=$REPEATS) ==="
python "$ROOT/scripts/run_gepa.py" \
  --student-model "$MODEL_STUDENT" \
  --reflector-model "$MODEL_REFLECTOR" \
  --split "$split" \
  --steps "$STEPS" \
  --repeats "$REPEATS" \
  --baseline-repeats "$BASELINE_REPEATS" \
  ${TIMEOUT:+--timeout "$TIMEOUT"} \
  --label-prefix "$label" \
  --save-dir "$SAVE_DIR"

# Auto-plot if the run produced a summary.
SUMMARY_PATH="$run_dir/summary.json"
if [ -f "$SUMMARY_PATH" ]; then
  OUT_DIR="$ROOT/out"
  mkdir -p "$OUT_DIR"
  OUT_PNG="$OUT_DIR/${label}.png"
  python "$ROOT/scripts/plot_gepa_metrics.py" \
    --run "$run_dir" \
    --output "$OUT_PNG" \
    --title "GEPA metrics for $label" || true
  echo "Plot (if generated): $OUT_PNG"
else
  echo "No summary.json found at $SUMMARY_PATH; skipping plot."
fi
