#!/usr/bin/env bash
set -euo pipefail

# Evaluate baseline and the selected GEPA candidate on the 50-instance uniform test split (n=400).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
META="${ROOT_DIR}/out/metadata_uniform_experiment.json"
CPU_AFFINITY="${CPU_AFFINITY:-}"

CANDIDATE="${ROOT_DIR}/runs/gepa_uniform/20251129T014400Z_uniform_time_n400_total/eval/20251129T025753Z_uniform_time_n400_total_bdf0236a/candidate_linkern_block.c"

if [[ ! -f "${META}" ]]; then
  echo "Metadata not found: ${META}. Run 01_generate_uniform_splits.sh first." >&2
  exit 1
fi
if [[ ! -f "${CANDIDATE}" ]]; then
  echo "Candidate block missing: ${CANDIDATE}" >&2
  exit 1
fi

echo "Running baseline on uniform_test..."
python "${ROOT_DIR}/scripts/run_concorde_eval.py" \
  --binary "${ROOT_DIR}/concorde/install/bin/concorde" \
  --metadata "${META}" \
  --split uniform_test \
  --repeats 1 \
  --label uniform_test_baseline \
  --run-dir "${ROOT_DIR}/runs/eval" \
  ${CPU_AFFINITY:+--cpu-affinity "${CPU_AFFINITY}"}

echo "Running GEPA candidate (iter 40) on uniform_test..."
python "${ROOT_DIR}/scripts/evaluate_heuristic_candidate.py" \
  --candidate-file "${CANDIDATE}" \
  --label uniform_test_gepa_iter40 \
  --split uniform_test \
  --repeats 1 \
  --metadata "${META}" \
  ${CPU_AFFINITY:+--cpu-affinity "${CPU_AFFINITY}"} \
  --run-root "${ROOT_DIR}/runs/eval"

echo "Done."
