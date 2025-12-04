#!/usr/bin/env bash
set -euo pipefail

# Plot the latest GEPA Seattle large run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$(ls -dt "${ROOT_DIR}"/runs/gepa_seattle_large/* 2>/dev/null | head -n 1 || true)}"
TITLE="${TITLE:-GEPA metrics for structured_seattle_time (latest)}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "No runs found under runs/gepa_seattle_large" >&2
  exit 0
fi

OUT_PNG="${OUT_PNG:-${RUN_DIR}/gepa_plot.png}"

python "${ROOT_DIR}/scripts/plot_gepa_metrics.py" \
  --run-dir "${RUN_DIR}" \
  --title "${TITLE}" \
  --output "${OUT_PNG}" || true

echo "Plot saved to ${OUT_PNG}"
