#!/usr/bin/env python3
"""
Add TSPLIB instances to data/eval/metadata.json under a specified split.

Usage:
  # Drop *.tsp files into data/eval/tsplib/ first, then run:
  python scripts/add_tsplib_eval.py --split tsplib_hard

By default the script scans data/eval/tsplib/ for .tsp files, extracts the
NAME and DIMENSION fields, and appends entries to metadata.json with paths
relative to data/eval/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


def parse_header(tsp_path: Path) -> Dict[str, Any]:
    """Extract NAME and DIMENSION from a TSPLIB file header."""
    name = tsp_path.stem
    dimension = None

    header_pattern = re.compile(r"^\s*([A-Z_]+)\s*:\s*(.+)\s*$")
    with tsp_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = header_pattern.match(line.upper())
            if not match:
                continue
            key, value = match.group(1), match.group(2).strip()
            if key == "NAME":
                name = value
            elif key == "DIMENSION":
                try:
                    dimension = int(value)
                except ValueError:
                    dimension = None
            elif key in {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION"}:
                # Reached data portion; stop parsing header.
                break

    if dimension is None:
        raise ValueError(f"Could not determine DIMENSION for {tsp_path}")

    return {"name": name, "dimension": dimension}


def load_metadata(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"instances": []}


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    metadata["instances"].sort(key=lambda entry: entry["id"])
    path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Register TSPLIB instances in metadata.json.")
    parser.add_argument(
        "--split",
        default="tsplib_hard",
        help="Split label to assign to the added instances (default: tsplib_hard).",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("data/eval/tsplib"),
        help="Directory to scan for TSPLIB .tsp files (default: data/eval/tsplib).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/eval/metadata.json"),
        help="Path to the evaluation metadata JSON file.",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.exists():
        raise SystemExit(f"{directory} does not exist. Create it and place TSPLIB .tsp files inside.")

    metadata_path = args.metadata.resolve()
    metadata = load_metadata(metadata_path)
    existing_ids = {entry["id"] for entry in metadata.get("instances", [])}

    added = 0
    eval_root = Path("data/eval").resolve()
    for tsp_path in sorted(directory.glob("*.tsp")):
        header = parse_header(tsp_path)
        name = header["name"]
        dimension = header["dimension"]

        try:
            rel_path = tsp_path.relative_to(eval_root)
        except ValueError:
            rel_path = tsp_path.name

        instance_id = f"tsplib_{name.lower()}"
        if instance_id in existing_ids:
            print(f"Skipping {tsp_path.name}: id '{instance_id}' already in metadata.")
            continue

        metadata.setdefault("instances", []).append(
            {
                "id": instance_id,
                "file": str(rel_path),
                "n": dimension,
                "seed": None,
                "generator": "tsplib",
                "distribution": "tsplib",
                "split": args.split,
                "description": f"TSPLIB instance {name} with {dimension} nodes.",
            }
        )
        existing_ids.add(instance_id)
        added += 1
        print(f"Registered {tsp_path.name} as {instance_id} (n={dimension}).")

    if added == 0:
        print("No new TSPLIB instances were added.")
    else:
        write_metadata(metadata_path, metadata)
        print(f"Updated metadata with {added} TSPLIB instance(s).")


if __name__ == "__main__":
    main()
