#!/usr/bin/env bash
set -euo pipefail

# Generate uniform Euclidean TSP instances for validation (train) and test,
# and write a self-contained metadata file with absolute paths.
#
# Splits:
#   - uniform_val  : 20 instances, seeds 3000–3019
#   - uniform_test : 50 instances, seeds 4000–4049
# All instances are n=400.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/data/eval/experiments/uniform_experiment"
META_OUT="${ROOT_DIR}/out/metadata_uniform_experiment.json"

mkdir -p "${EXP_DIR}" "${ROOT_DIR}/out"

echo "Generating uniform validation instances..."
for seed in $(seq 3000 3019); do
  tsp_path="${EXP_DIR}/uniform_val_seed${seed}.tsp"
  python "${ROOT_DIR}/scripts/generators/gen_synthetic_structured.py" \
    --kind uniform \
    --n 400 \
    --seed "${seed}" \
    --bbox 0 1 0 1 \
    --tsp-path "${tsp_path}"
done

echo "Generating uniform test instances..."
for seed in $(seq 4000 4049); do
  tsp_path="${EXP_DIR}/uniform_test_seed${seed}.tsp"
  python "${ROOT_DIR}/scripts/generators/gen_synthetic_structured.py" \
    --kind uniform \
    --n 400 \
    --seed "${seed}" \
    --bbox 0 1 0 1 \
    --tsp-path "${tsp_path}"
done

echo "Writing metadata to ${META_OUT}"
python - <<PY
import json, pathlib
root = pathlib.Path("${ROOT_DIR}")
instances = []
for seed in range(3000, 3020):
    tsp = root / "data/eval/experiments/uniform_experiment" / f"uniform_val_seed{seed}.tsp"
    instances.append({
        "id": f"uniform_val_seed{seed}",
        "file": str(tsp),
        "n": 400,
        "split": "uniform_val",
        "generator": "uniform_euclidean",
        "bbox": [0, 1, 0, 1],
        "seed": seed,
        "description": "Uniform Euclidean points (n=400) for validation"
    })
for seed in range(4000, 4050):
    tsp = root / "data/eval/experiments/uniform_experiment" / f"uniform_test_seed{seed}.tsp"
    instances.append({
        "id": f"uniform_test_seed{seed}",
        "file": str(tsp),
        "n": 400,
        "split": "uniform_test",
        "generator": "uniform_euclidean",
        "bbox": [0, 1, 0, 1],
        "seed": seed,
        "description": "Uniform Euclidean points (n=400) held-out test set"
    })
meta = {"instances": instances}
out = root / "out" / "metadata_uniform_experiment.json"
out.write_text(json.dumps(meta, indent=2))
print(f"Wrote {out}")
PY

echo "Done."
