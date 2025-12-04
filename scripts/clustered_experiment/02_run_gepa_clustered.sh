#!/usr/bin/env bash
set -euo pipefail

# Run baseline + GEPA on the clustered validation split (explicit matrix with 0.5x intra-cluster edges).
# - Baseline repeats: 5
# - GEPA steps: 16
# - Reflection batch (candidates per step): 2 by default
# - Candidate repeats: 3
# - CPU affinity optional (defaults to unset to let the OS schedule)
# - Dataset: clustered_val n=400

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
META="${ROOT_DIR}/out/metadata_clustered_experiment.json"
RUN_LABEL_PREFIX="${RUN_LABEL_PREFIX:-clustered_n400_descriptive_prompt}"
CPU_AFFINITY="${CPU_AFFINITY:-}"
REFLECTION_BATCH="${REFLECTION_BATCH:-2}"
REFLECTOR_CONTEXT="${REFLECTOR_CONTEXT:-These TSP instances have 400 nodes arranged in 4 clusters inside a unit square. Edge weights are an explicit full matrix; intra-cluster edges are discounted by 0.5x, so staying within a cluster is cheaper than jumping across clusters. The evaluator never shows you coordinates, only the explicit distances. Optimize Lin-Kernighan for this clustered, discounted setting.}"

if [[ ! -f "${META}" ]]; then
  echo "Metadata not found: ${META}. Run 01_generate_clustered_splits.sh first." >&2
  exit 1
fi

echo "Running GEPA on clustered_val (reflection batch=${REFLECTION_BATCH}, cpu_affinity='${CPU_AFFINITY:-<unset>}')"
cmd=(python "${ROOT_DIR}/scripts/run_gepa.py"
  --split clustered_val
  --repeats 3
  --baseline-repeats 5
  --steps 16
  --metadata "${META}"
  --reflection-batch "${REFLECTION_BATCH}"
  --student-model openai/gpt-5-nano
  --reflector-model openai/gpt-5-mini
  --label-prefix "${RUN_LABEL_PREFIX}"
  --save-dir "${ROOT_DIR}/runs/gepa_clustered"
)
if [[ -n "${REFLECTOR_CONTEXT}" ]]; then
  cmd+=(--reflector-context "${REFLECTOR_CONTEXT}")
fi
if [[ -n "${CPU_AFFINITY}" ]]; then
  cmd+=(--cpu-affinity "${CPU_AFFINITY}")
fi

"${cmd[@]}"

echo "Done."
