#!/usr/bin/env bash
set -euo pipefail

# Heavy GEPA run on the structured Seattle travel-time split (n≈400).
# Uses reward signaling (best-so-far + baseline injected into feedback) and
# logs reflector datasets/proposed instructions.
#
# Config:
#   - split: structured_seattle_time
#   - repeats: 3 (candidate), baseline_repeats: 5
#   - steps: 20 metric calls (adjust STEPS env var to change)
#   - reflection batch: 3
#   - timeout: 60s per instance (adjust TIMEOUT)
#   - models: student gpt-5-nano, reflector gpt-5-mini

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Default to the generated multi-instance Seattle time metadata if present.
DEFAULT_META="${ROOT_DIR}/out/metadata_seattle_time_experiment.json"
META="${META:-${DEFAULT_META}}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/runs/gepa_seattle_large}"
RUN_LABEL_PREFIX="${RUN_LABEL_PREFIX:-seattle_n400_large}"
CPU_AFFINITY="${CPU_AFFINITY:-}"
REFLECTION_BATCH="${REFLECTION_BATCH:-3}"
STEPS="${STEPS:-20}"
TIMEOUT="${TIMEOUT:-60}"
STUDENT_MODEL="${STUDENT_MODEL:-openai/gpt-5}"
REFLECTOR_MODEL="${REFLECTOR_MODEL:-openai/gpt-5}"
REFLECTOR_CONTEXT_FILE="${ROOT_DIR}/scripts/seattle_large_experiment/reflector_context.txt"
REFLECTOR_PROMPT_OVERRIDE="${ROOT_DIR}/scripts/seattle_large_experiment/reflector_prompt_override.txt"

if [[ ! -f "${META}" ]]; then
  echo "Metadata not found: ${META}" >&2
  exit 1
fi

REFLECTOR_CONTEXT=""
if [[ -f "${REFLECTOR_CONTEXT_FILE}" ]]; then
  REFLECTOR_CONTEXT="$(cat "${REFLECTOR_CONTEXT_FILE}")"
fi

echo "Running GEPA on structured_seattle_time (reflection batch=${REFLECTION_BATCH}, steps=${STEPS}, cpu_affinity='${CPU_AFFINITY:-<unset>}')"
cmd=(python "${ROOT_DIR}/scripts/run_gepa.py"
  --split structured_seattle_time_val
  --repeats 3
  --baseline-repeats 5
  --steps "${STEPS}"
  --reflection-batch "${REFLECTION_BATCH}"
  --train-examples 1
  --timeout "${TIMEOUT}"
  --student-model "${STUDENT_MODEL}"
  --reflector-model "${REFLECTOR_MODEL}"
  --label-prefix "${RUN_LABEL_PREFIX}"
  --metadata "${META}"
  --save-dir "${SAVE_DIR}"
)
if [[ -n "${REFLECTOR_CONTEXT}" ]]; then
  cmd+=(--reflector-context "${REFLECTOR_CONTEXT}")
fi
if [[ -n "${CPU_AFFINITY}" ]]; then
  cmd+=(--cpu-affinity "${CPU_AFFINITY}")
fi
# Override the instruction proposal prompt for this experiment, if present.
if [[ -f "${REFLECTOR_PROMPT_OVERRIDE}" ]]; then
  export GEPA_INSTRUCTION_PROMPT_OVERRIDE_FILE="${REFLECTOR_PROMPT_OVERRIDE}"
fi

"${cmd[@]}"

echo "Done."
