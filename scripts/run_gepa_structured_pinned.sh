#!/usr/bin/env bash
# Kick off a GEPA run on the structured_seattle_time split with CPU pinning.
# Adjust AFFINITY, STEPS, and REPEATS via env vars as needed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Tuning knobs (override via env, e.g., AFFINITY=0-3 STEPS=20 REPEATS=3 ./scripts/run_gepa_structured_pinned.sh)
AFFINITY="${AFFINITY:-0-3}"
STEPS="${STEPS:-20}"
REPEATS="${REPEATS:-3}"
BASELINE_REPEATS="${BASELINE_REPEATS:-10}"
LABEL_PREFIX="${LABEL_PREFIX:-structured_seattle_time_n400_time_hreps}"
SAVE_DIR="${SAVE_DIR:-$ROOT/runs/gepa_structured}"

echo "Running GEPA with CPU affinity '${AFFINITY}', steps=${STEPS}, repeats=${REPEATS}, baseline_repeats=${BASELINE_REPEATS}"

python "$ROOT/scripts/run_gepa.py" \
  --split structured_seattle_time \
  --repeats "${REPEATS}" \
  --baseline-repeats "${BASELINE_REPEATS}" \
  --steps "${STEPS}" \
  --cpu-affinity "${AFFINITY}" \
  --label-prefix "${LABEL_PREFIX}" \
  --save-dir "${SAVE_DIR}"
