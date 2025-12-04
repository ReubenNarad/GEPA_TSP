#!/usr/bin/env bash
set -euo pipefail

# Run baseline + GEPA on the random explicit validation split (symmetric U(0,1) distances).
# - Baseline repeats: 5
# - GEPA steps: 16
# - Reflection batch: 2 by default
# - Candidate repeats: 3
# - Dataset: random_explicit_val n=400

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
META="${ROOT_DIR}/out/metadata_random_explicit_experiment.json"
RUN_LABEL_PREFIX="${RUN_LABEL_PREFIX:-random_explicit_n400}"
CPU_AFFINITY="${CPU_AFFINITY:-}"
REFLECTION_BATCH="${REFLECTION_BATCH:-2}"
REFLECTOR_CONTEXT="${REFLECTOR_CONTEXT:-These TSP instances use symmetric explicit distances drawn uniformly in [0,1]. There is no geometric structure and coordinates are meaningless; you cannot assume Euclidean geometry. Optimize Lin-Kernighan for random explicit weights, n=400.}"

if [[ ! -f "${META}" ]]; then
  echo "Metadata not found: ${META}. Run 01_generate_random_explicit_splits.sh first." >&2
  exit 1
fi

echo "Running GEPA on random_explicit_val (reflection batch=${REFLECTION_BATCH}, cpu_affinity='${CPU_AFFINITY:-<unset>}')"
cmd=(python "${ROOT_DIR}/scripts/run_gepa.py"
  --split random_explicit_val
  --repeats 3
  --baseline-repeats 5
  --steps 16
  --metadata "${META}"
  --reflection-batch "${REFLECTION_BATCH}"
  --student-model openai/gpt-5-nano
  --reflector-model openai/gpt-5-mini
  --label-prefix "${RUN_LABEL_PREFIX}"
  --save-dir "${ROOT_DIR}/runs/gepa_random_explicit"
)
if [[ -n "${REFLECTOR_CONTEXT}" ]]; then
  cmd+=(--reflector-context "${REFLECTOR_CONTEXT}")
fi
if [[ -n "${CPU_AFFINITY}" ]]; then
  cmd+=(--cpu-affinity "${CPU_AFFINITY}")
fi

"${cmd[@]}"

echo "Done."
