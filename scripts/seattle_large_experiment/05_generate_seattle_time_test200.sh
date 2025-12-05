#!/usr/bin/env bash
set -euo pipefail

# Generate the full 50-instance Seattle travel-time test split
# using the same parameters as 01_generate_seattle_time_splits.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PBF_PATH="${PBF_PATH:-${ROOT_DIR}/data/osm/Seattle.osm.pbf}"
EXP_DIR="${ROOT_DIR}/data/eval/experiments/seattle_time_experiment"
META_OUT="${META_OUT:-${ROOT_DIR}/out/metadata_seattle_time_experiment.json}"

BBOX_LAT_MIN=47.58
BBOX_LAT_MAX=47.64
BBOX_LON_MIN=-122.36
BBOX_LON_MAX=-122.30
N=400
SEED_START=2025112600
SEED_END=2025112649

if [[ ! -f "${PBF_PATH}" ]]; then
  echo "Missing Seattle OSM extract: ${PBF_PATH}" >&2
  exit 1
fi

mkdir -p "${EXP_DIR}" "${ROOT_DIR}/out"

for seed in $(seq "${SEED_START}" "${SEED_END}"); do
  tsp_path="${EXP_DIR}/structured_seattle_time_test_seed${seed}.tsp"
  if [[ -f "${tsp_path}" ]]; then
    echo "Skipping existing ${tsp_path}"
    continue
  fi
  echo "Generating ${tsp_path}"
  python "${ROOT_DIR}/scripts/generators/gen_seattle_from_osm.py" \
    --pbf "${PBF_PATH}" \
    --bbox "${BBOX_LAT_MIN}" "${BBOX_LAT_MAX}" "${BBOX_LON_MIN}" "${BBOX_LON_MAX}" \
    --n "${N}" \
    --seed "${seed}" \
    --weight time \
    --tsp-path "${tsp_path}"
done

echo "Refreshing metadata at ${META_OUT}"
python - <<PY
import json, pathlib
root = pathlib.Path("${ROOT_DIR}")
exp = root / "data/eval/experiments/seattle_time_experiment"
instances = []
for seed in range(2025112500, 2025112699):
    tsp = exp / f"structured_seattle_time_val_seed{seed}.tsp"
    instances.append({
        "id": f"structured_seattle_time_val_seed{seed}",
        "file": str(tsp),
        "n": ${N},
        "split": "structured_seattle_time_val",
        "generator": "seattle_osm_traveltime",
        "bbox": [${BBOX_LAT_MIN}, ${BBOX_LAT_MAX}, ${BBOX_LON_MIN}, ${BBOX_LON_MAX}],
        "seed": seed,
        "weight": "time",
        "description": "Seattle road-network travel-time TSP (n=${N}) validation set"
    })
for seed in range(${SEED_START}, ${SEED_END} + 1):
    tsp = exp / f"structured_seattle_time_test_seed{seed}.tsp"
    instances.append({
        "id": f"structured_seattle_time_test_seed{seed}",
        "file": str(tsp),
        "n": ${N},
        "split": "structured_seattle_time_test",
        "generator": "seattle_osm_traveltime",
        "bbox": [${BBOX_LAT_MIN}, ${BBOX_LAT_MAX}, ${BBOX_LON_MIN}, ${BBOX_LON_MAX}],
        "seed": seed,
        "weight": "time",
        "description": "Seattle road-network travel-time TSP (n=${N}) held-out test set"
    })
out = pathlib.Path("${META_OUT}")
out.write_text(json.dumps({"instances": instances}, indent=2))
print(f"Wrote {out} with {len(instances)} instances")
PY

echo "Done."
