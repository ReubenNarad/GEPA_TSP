#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Required: OPENROUTER_API_KEY exported in your env
python -u scripts/run_gepa.py \
  --student-model openai/gpt-5-nano \
  --reflector-model openai/gpt-5-mini \
  --student-max-tokens 4000 \
  --reflector-max-tokens 3000 \
  --split toy20 \
  --repeats 1 \
  --baseline-repeats 1 \
  --timeout 30 \
  --steps 6 \
  --train-examples 1 \
  --reflection-batch 2 \
  --label-prefix smoke_cache \
  --save-dir runs/gepa_smoke_cache
