#!/usr/bin/env bash
set -euo pipefail

# Generate Seattle travel-time TSP instances (n=400) from the OSM road network.
# Produces validation and test splits plus a metadata file listing all instances.
# Split names:
#   - structured_seattle_time_val  : 20 instances (seeds 2025112500–2025112519)
#   - structured_seattle_time_test : 10 instances (seeds 2025112600–2025112609)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PBF_PATH="${PBF_PATH:-${ROOT_DIR}/data/osm/Seattle.osm.pbf}"
EXP_DIR="${ROOT_DIR}/data/eval/experiments/seattle_time_experiment"
META_OUT="${ROOT_DIR}/out/metadata_seattle_time_experiment.json"

BBOX_LAT_MIN=47.58
BBOX_LAT_MAX=47.64
BBOX_LON_MIN=-122.36
BBOX_LON_MAX=-122.30

mkdir -p "${EXP_DIR}" "${ROOT_DIR}/out"

if [[ ! -f "${PBF_PATH}" ]]; then
  echo "Missing Seattle OSM extract: ${PBF_PATH}" >&2
  exit 1
fi

generate_split() {
  local split_name=$1
  local start_seed=$2
  local end_seed=$3
  for seed in $(seq "${start_seed}" "${end_seed}"); do
    local tsp_path="${EXP_DIR}/${split_name}_seed${seed}.tsp"
    echo "Generating ${tsp_path}"
    python "${ROOT_DIR}/scripts/generators/gen_seattle_from_osm.py" \
      --pbf "${PBF_PATH}" \
      --bbox "${BBOX_LAT_MIN}" "${BBOX_LAT_MAX}" "${BBOX_LON_MIN}" "${BBOX_LON_MAX}" \
      --n 400 \
      --seed "${seed}" \
      --weight time \
      --tsp-path "${tsp_path}"
  done
}

echo "Generating validation split (structured_seattle_time_val)..."
generate_split "structured_seattle_time_val" 2025112500 2025112519

echo "Generating test split (structured_seattle_time_test)..."
generate_split "structured_seattle_time_test" 2025112600 2025112609

echo "Writing metadata to ${META_OUT}"
python - <<PY
import json, pathlib
root = pathlib.Path("${ROOT_DIR}")
exp = root / "data/eval/experiments/seattle_time_experiment"
instances = []
for seed in range(2025112500, 2025112520):
    tsp = exp / f"structured_seattle_time_val_seed{seed}.tsp"
    instances.append({
        "id": f"structured_seattle_time_val_seed{seed}",
        "file": str(tsp),
        "n": 400,
        "split": "structured_seattle_time_val",
        "generator": "seattle_osm_traveltime",
        "bbox": [${BBOX_LAT_MIN}, ${BBOX_LAT_MAX}, ${BBOX_LON_MIN}, ${BBOX_LON_MAX}],
        "seed": seed,
        "weight": "time",
        "description": "Seattle road-network travel-time TSP (n=400) validation set"
    })
for seed in range(2025112600, 2025112610):
    tsp = exp / f"structured_seattle_time_test_seed{seed}.tsp"
    instances.append({
        "id": f"structured_seattle_time_test_seed{seed}",
        "file": str(tsp),
        "n": 400,
        "split": "structured_seattle_time_test",
        "generator": "seattle_osm_traveltime",
        "bbox": [${BBOX_LAT_MIN}, ${BBOX_LAT_MAX}, ${BBOX_LON_MIN}, ${BBOX_LON_MAX}],
        "seed": seed,
        "weight": "time",
        "description": "Seattle road-network travel-time TSP (n=400) held-out test set"
    })
out = pathlib.Path("${META_OUT}")
out.write_text(json.dumps({"instances": instances}, indent=2))
print(f"Wrote {out} with {len(instances)} instances")
PY

echo "Done."
