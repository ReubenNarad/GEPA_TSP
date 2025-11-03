#!/usr/bin/env bash
# Convenience wrapper demonstrating how to launch a GEPA run.
# Adjust the model identifiers or dataset parameters as needed.

set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set; add it to .env or export it before running this script." >&2
  exit 1
fi

python3 scripts/run_gepa.py \
  --student-model openai/gpt-5-nano \
  --reflector-model openai/gpt-5-mini \
  --split toy20 \
  --repeats 1 \
  --steps 3 \
  --reflection-batch 2 \
  --label-prefix gepa_demo \
  "${@}"
