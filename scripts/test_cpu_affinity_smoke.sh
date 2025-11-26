#!/usr/bin/env bash
# Quick smoke test for the CPU affinity plumbing.
# Runs the toy20 split twice: once with taskset pinning, once without.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$ROOT/concorde/install/bin/concorde"
METADATA="$ROOT/data/eval/metadata.json"

if [[ ! -x "$BINARY" ]]; then
  echo "error: Concorde binary not found or not executable at $BINARY" >&2
  exit 1
fi

AFFINITY="${AFFINITY:-0}"  # export AFFINITY=0-3 to change pinning
LABEL_PREFIX="cpu_affinity_smoke"

echo "== Run with CPU affinity (${AFFINITY}) =="
python "$ROOT/scripts/run_concorde_eval.py" \
  --binary "$BINARY" \
  --metadata "$METADATA" \
  --split toy20 \
  --repeats 1 \
  --label "${LABEL_PREFIX}_pinned" \
  --cpu-affinity "$AFFINITY"

echo
echo "== Run without CPU affinity =="
python "$ROOT/scripts/run_concorde_eval.py" \
  --binary "$BINARY" \
  --metadata "$METADATA" \
  --split toy20 \
  --repeats 1 \
  --label "${LABEL_PREFIX}_unpinned"

echo
echo "Done. Check runs/eval/*${LABEL_PREFIX}* config.json for binary_sha256 and cpu_affinity."
