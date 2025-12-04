#!/usr/bin/env bash
set -euo pipefail

# Evaluate baseline and the selected GEPA candidate on the 50-instance clustered test split (n=400).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
META="${ROOT_DIR}/out/metadata_clustered_experiment.json"
CPU_AFFINITY="${CPU_AFFINITY:-}"

CANDIDATE="${ROOT_DIR}/runs/gepa_clustered/20251129T230105Z_clustered_n400_descriptive_prompt/eval/20251129T232855Z_clustered_n400_descriptive_prompt_69d101f1/candidate_linkern_block.c"

if [[ ! -f "${META}" ]]; then
  echo "Metadata not found: ${META}. Run 01_generate_clustered_splits.sh first." >&2
  exit 1
fi
if [[ ! -f "${CANDIDATE}" ]]; then
  echo "Candidate block missing: ${CANDIDATE}" >&2
  exit 1
fi

echo "Running baseline on clustered_test..."
python "${ROOT_DIR}/scripts/run_concorde_eval.py" \
  --binary "${ROOT_DIR}/concorde/install/bin/concorde" \
  --metadata "${META}" \
  --split clustered_test \
  --repeats 1 \
  --label clustered_test_baseline \
  --run-dir "${ROOT_DIR}/runs/eval" \
  ${CPU_AFFINITY:+--cpu-affinity "${CPU_AFFINITY}"}

echo "Running GEPA candidate (iter 30) on clustered_test..."
python "${ROOT_DIR}/scripts/evaluate_heuristic_candidate.py" \
  --candidate-file "${CANDIDATE}" \
  --label clustered_test_gepa_iter30 \
  --split clustered_test \
  --repeats 1 \
  --metadata "${META}" \
  ${CPU_AFFINITY:+--cpu-affinity "${CPU_AFFINITY}"} \
  --run-root "${ROOT_DIR}/runs/eval"

echo "Done."
