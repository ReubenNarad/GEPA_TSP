#!/usr/bin/env bash
set -euo pipefail

# Generate clustered TSP instances with an explicit distance matrix that applies
# a 50% discount to intra-cluster edges. Produces:
#   - clustered_val  : 20 instances, seeds 5000–5019
#   - clustered_test : 50 instances, seeds 6000–6049
# All instances are n=400 with k=4 clusters.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/data/eval/experiments/clustered_experiment"
META_OUT="${ROOT_DIR}/out/metadata_clustered_experiment.json"

mkdir -p "${EXP_DIR}" "${ROOT_DIR}/out"

echo "Generating clustered validation instances (explicit matrix, 50% intra-cluster discount)..."
for seed in $(seq 5000 5019); do
  tsp_path="${EXP_DIR}/clustered_val_seed${seed}.tsp"
  python "${ROOT_DIR}/scripts/generators/gen_synthetic_structured.py" \
    --kind clustered \
    --n 400 \
    --clusters 4 \
    --seed "${seed}" \
    --bbox 0 1 0 1 \
    --explicit-cluster-discount 0.5 \
    --tsp-path "${tsp_path}"
done

echo "Generating clustered test instances (explicit matrix, 50% intra-cluster discount)..."
for seed in $(seq 6000 6049); do
  tsp_path="${EXP_DIR}/clustered_test_seed${seed}.tsp"
  python "${ROOT_DIR}/scripts/generators/gen_synthetic_structured.py" \
    --kind clustered \
    --n 400 \
    --clusters 4 \
    --seed "${seed}" \
    --bbox 0 1 0 1 \
    --explicit-cluster-discount 0.5 \
    --tsp-path "${tsp_path}"
done

echo "Writing metadata to ${META_OUT}"
python - <<PY
import json, pathlib
root = pathlib.Path("${ROOT_DIR}")
instances = []
for seed in range(5000, 5020):
    tsp = root / "data/eval/experiments/clustered_experiment" / f"clustered_val_seed{seed}.tsp"
    instances.append({
        "id": f"clustered_val_seed{seed}",
        "file": str(tsp),
        "n": 400,
        "split": "clustered_val",
        "generator": "clustered_structured_explicit_discount",
        "bbox": [0, 1, 0, 1],
        "k": 4,
        "seed": seed,
        "within_cluster_discount": 0.5,
        "edge_weight_type": "EXPLICIT_FULL_MATRIX",
        "description": "Clustered (k=4) with explicit distances and 0.5x intra-cluster edges (n=400) for validation"
    })
for seed in range(6000, 6050):
    tsp = root / "data/eval/experiments/clustered_experiment" / f"clustered_test_seed{seed}.tsp"
    instances.append({
        "id": f"clustered_test_seed{seed}",
        "file": str(tsp),
        "n": 400,
        "split": "clustered_test",
        "generator": "clustered_structured_explicit_discount",
        "bbox": [0, 1, 0, 1],
        "k": 4,
        "seed": seed,
        "within_cluster_discount": 0.5,
        "edge_weight_type": "EXPLICIT_FULL_MATRIX",
        "description": "Clustered (k=4) with explicit distances and 0.5x intra-cluster edges (n=400) held-out test set"
    })
meta = {"instances": instances}
out = root / "out" / "metadata_clustered_experiment.json"
out.write_text(json.dumps(meta, indent=2))
print(f"Wrote {out}")
PY

echo "Done."
