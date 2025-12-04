#!/usr/bin/env bash
set -euo pipefail

# Run baseline + GEPA on the uniform validation split.
# - Baseline repeats: 5
# - GEPA steps: 16
# - Reflection batch (candidates per step): 1 (serialize to avoid contention)
# - Candidate repeats: 3
# - CPU affinity optional (defaults to unset to let the OS schedule)
# - Dataset: uniform_val n=400

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
META="${ROOT_DIR}/out/metadata_uniform_experiment.json"
RUN_LABEL_PREFIX="${RUN_LABEL_PREFIX:-uniform_time_n400_total}"
CPU_AFFINITY="${CPU_AFFINITY:-}"
REFLECTION_BATCH="${REFLECTION_BATCH:-2}"

if [[ ! -f "${META}" ]]; then
  echo "Metadata not found: ${META}. Run 01_generate_uniform_splits.sh first." >&2
  exit 1
fi

echo "Running GEPA on uniform_val (reflection batch=${REFLECTION_BATCH}, cpu_affinity='${CPU_AFFINITY:-<unset>}')"
cmd=(python "${ROOT_DIR}/scripts/run_gepa.py"
  --split uniform_val
  --repeats 3
  --baseline-repeats 5
  --steps 18
  --metadata "${META}"
  --reflection-batch "${REFLECTION_BATCH}"
  --student-model openai/gpt-5-nano
  --reflector-model openai/gpt-5-mini
  --label-prefix "${RUN_LABEL_PREFIX}"
  --save-dir "${ROOT_DIR}/runs/gepa_uniform"
)
if [[ -n "${CPU_AFFINITY}" ]]; then
  cmd+=(--cpu-affinity "${CPU_AFFINITY}")
fi

"${cmd[@]}"

echo "Done."
