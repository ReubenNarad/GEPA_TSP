#!/usr/bin/env bash
set -euo pipefail

# Generate random explicit-distance TSP instances for validation (train) and test.
# Distances are symmetric U(0,1) in an EXPLICIT FULL_MATRIX; coordinates are dummy for display only.
# Splits:
#   - random_explicit_val  : 20 instances, seeds 7000–7019
#   - random_explicit_test : 50 instances, seeds 8000–8049
# All instances are n=400.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/data/eval/experiments/random_explicit_experiment"
META_OUT="${ROOT_DIR}/out/metadata_random_explicit_experiment.json"

mkdir -p "${EXP_DIR}" "${ROOT_DIR}/out"

echo "Generating random explicit validation instances..."
for seed in $(seq 7000 7019); do
  tsp_path="${EXP_DIR}/random_explicit_val_seed${seed}.tsp"
  python "${ROOT_DIR}/scripts/generators/gen_synthetic_structured.py" \
    --kind random_explicit \
    --n 400 \
    --seed "${seed}" \
    --tsp-path "${tsp_path}"
done

echo "Generating random explicit test instances..."
for seed in $(seq 8000 8049); do
  tsp_path="${EXP_DIR}/random_explicit_test_seed${seed}.tsp"
  python "${ROOT_DIR}/scripts/generators/gen_synthetic_structured.py" \
    --kind random_explicit \
    --n 400 \
    --seed "${seed}" \
    --tsp-path "${tsp_path}"
done

echo "Writing metadata to ${META_OUT}"
python - <<PY
import json, pathlib
root = pathlib.Path("${ROOT_DIR}")
instances = []
for seed in range(7000, 7020):
    tsp = root / "data/eval/experiments/random_explicit_experiment" / f"random_explicit_val_seed{seed}.tsp"
    instances.append({
        "id": f"random_explicit_val_seed{seed}",
        "file": str(tsp),
        "n": 400,
        "split": "random_explicit_val",
        "generator": "random_explicit",
        "seed": seed,
        "edge_weight_type": "explicit_random_uniform",
        "description": "Random symmetric explicit distances U(0,1) (n=400) for validation"
    })
for seed in range(8000, 8050):
    tsp = root / "data/eval/experiments/random_explicit_experiment" / f"random_explicit_test_seed{seed}.tsp"
    instances.append({
        "id": f"random_explicit_test_seed{seed}",
        "file": str(tsp),
        "n": 400,
        "split": "random_explicit_test",
        "generator": "random_explicit",
        "seed": seed,
        "edge_weight_type": "explicit_random_uniform",
        "description": "Random symmetric explicit distances U(0,1) (n=400) held-out test set"
    })
meta = {"instances": instances}
out = root / "out" / "metadata_random_explicit_experiment.json"
out.write_text(json.dumps(meta, indent=2))
print(f"Wrote {out}")
PY

echo "Done."
