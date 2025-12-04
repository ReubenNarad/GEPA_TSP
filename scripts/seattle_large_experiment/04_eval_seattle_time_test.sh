#!/usr/bin/env bash
set -euo pipefail

# Evaluate baseline and the selected GEPA candidate on the 50-instance Seattle travel-time test split (n=400).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
META="${ROOT_DIR}/out/metadata_seattle_time_experiment.json"
CPU_AFFINITY="${CPU_AFFINITY:-}"

CANDIDATE="${ROOT_DIR}/runs/gepa_structured/20251126T031935Z_structured_seattle_time_n400_time_hreps/eval/20251126T033940Z_structured_seattle_time_n400_time_hreps_d1740dc7/candidate_linkern_block.c"

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
  --label seattle_time_test_baseline \
  --run-dir "${ROOT_DIR}/runs/eval" \
  ${CPU_AFFINITY:+--cpu-affinity "${CPU_AFFINITY}"}

echo "Running GEPA candidate (iter 31) on structured_seattle_time_test..."
python "${ROOT_DIR}/scripts/evaluate_heuristic_candidate.py" \
  --candidate-file "${CANDIDATE}" \
  --label seattle_time_test_gepa_iter31 \
  --split structured_seattle_time_test \
  --repeats 1 \
  --metadata "${META}" \
  ${CPU_AFFINITY:+--cpu-affinity "${CPU_AFFINITY}"} \
  --run-root "${ROOT_DIR}/runs/eval"

echo "Done."
