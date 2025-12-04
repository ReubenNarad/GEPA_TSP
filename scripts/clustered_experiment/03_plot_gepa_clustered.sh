#!/usr/bin/env bash
set -euo pipefail

# Plot the latest GEPA clustered run with smoothed metrics.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LATEST_RUN=$(ls -dt "${ROOT_DIR}"/runs/gepa_clustered/* 2>/dev/null | head -n 1 || true)

if [[ -z "${LATEST_RUN}" ]]; then
  echo "No runs found under runs/gepa_clustered." >&2
  exit 1
fi

OUT_PNG="${ROOT_DIR}/out/gepa_clustered_latest.png"
TITLE="${TITLE:-GEPA metrics for clustered_val (latest)}"

echo "Plotting ${LATEST_RUN} -> ${OUT_PNG}"
python "${ROOT_DIR}/scripts/plot_gepa_metrics_smoothed.py" \
  --run "${LATEST_RUN}" \
  --output "${OUT_PNG}" \
  --title "${TITLE}"

echo "Done."
