#!/usr/bin/env bash
# Run a 10-step GEPA optimization on the tsplib_random split and plot results.
# Usage: ./scripts/run_gepa_tsplib.sh [extra CLI args for run_gepa.py]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_DIR="$ROOT/runs/gepa"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LABEL="tsplib_run"
RUN_DIR="$SAVE_DIR/${TIMESTAMP}_${LABEL}"
PLOT_PATH="$RUN_DIR/gepa_plot.png"

mkdir -p "$SAVE_DIR"

# Run GEPA
python "$ROOT/scripts/run_gepa.py" \
  --student-model openai/gpt-5-nano \
  --reflector-model openai/gpt-5-mini \
  --split tsplib_random \
  --steps 10 \
  --label-prefix "$LABEL" \
  --save-dir "$SAVE_DIR" \
  "$@"

# Plot the rollouts
python "$ROOT/scripts/plot_gepa_metrics.py" --run "$RUN_DIR" --output "$PLOT_PATH"

echo "Plot saved to $PLOT_PATH"
