#!/usr/bin/env bash
set -euo pipefail

# Evaluate baseline and the latest GEPA final candidate from
# runs/gepa_seattle_large/20251202T212244Z_seattle_n400_large on the 50-instance
# structured_seattle_time_test split (n=400).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
META="${ROOT_DIR}/out/metadata_seattle_time_experiment.json"
CPU_AFFINITY="${CPU_AFFINITY:-}"

CANDIDATE="${ROOT_DIR}/runs/gepa_seattle_large/20251202T212244Z_seattle_n400_large/eval/20251202T214326Z_seattle_n400_large_final/candidate_linkern_block.c"

if [[ ! -f "${META}" ]]; then
  echo "Metadata not found: ${META}. Run 01_generate_seattle_time_splits.sh first." >&2
  exit 1
fi
if [[ ! -f "${CANDIDATE}" ]]; then
  echo "Candidate block missing: ${CANDIDATE}" >&2
  exit 1
fi

echo "Running baseline on structured_seattle_time_test..."
python "${ROOT_DIR}/scripts/run_concorde_eval.py" \
  --binary "${ROOT_DIR}/concorde/install/bin/concorde" \
  --metadata "${META}" \
  --split structured_seattle_time_test \
  --repeats 1 \
  --label seattle_time_test_baseline_latest \
  --run-dir "${ROOT_DIR}/runs/eval" \
  ${CPU_AFFINITY:+--cpu-affinity "${CPU_AFFINITY}"}

echo "Running GEPA final candidate on structured_seattle_time_test..."
python "${ROOT_DIR}/scripts/evaluate_heuristic_candidate.py" \
  --candidate-file "${CANDIDATE}" \
  --label seattle_time_test_gepa_final \
  --split structured_seattle_time_test \
  --repeats 1 \
  --metadata "${META}" \
  ${CPU_AFFINITY:+--cpu-affinity "${CPU_AFFINITY}"} \
  --run-root "${ROOT_DIR}/runs/eval"

echo "Done."
