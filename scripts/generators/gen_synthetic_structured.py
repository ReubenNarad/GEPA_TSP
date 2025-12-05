#!/usr/bin/env python3
"""Generate synthetic structured TSPLIB instances (uniform, clustered, grid)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from instance_sampler import StructuredSampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic structured TSPLIB instances.")
    p.add_argument("--kind", choices=["uniform", "clustered", "grid", "random_explicit"], required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--bbox", type=float, nargs=4, default=[0.0, 1.0, 0.0, 1.0], help="xmin xmax ymin ymax")
    p.add_argument("--clusters", type=int, default=4, help="K for clustered.")
    p.add_argument("--cov-min", type=float, default=0.001, help="Min eigenvalue for clustered cov.")
    p.add_argument("--cov-max", type=float, default=0.02, help="Max eigenvalue for clustered cov.")
    p.add_argument("--jitter", type=float, default=0.01, help="Grid jitter.")
    p.add_argument("--drop-prob", type=float, default=0.0, help="Grid row/col drop probability.")
    p.add_argument(
        "--explicit-cluster-discount",
        type=float,
        default=None,
        help="If set (clustered only), emit EXPLICIT distances and multiply intra-cluster edges by this factor.",
    )
    p.add_argument("--tsp-path", type=Path, required=True, help="Output TSPLIB path.")
    return p.parse_args()


def write_tsplib(coords: np.ndarray, path: Path) -> None:
    scale = 100000.0
    ints = np.rint(coords * scale).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"NAME: {path.stem}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {len(ints)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(ints, start=1):
            f.write(f"{idx} {x} {y}\n")
        f.write("EOF\n")


def write_tsplib_explicit(dist_matrix: np.ndarray, coords: np.ndarray, path: Path) -> None:
    """Write an EXPLICIT FULL_MATRIX instance with display coords."""
    scale = 100000.0
    sym = np.minimum(dist_matrix, dist_matrix.T)
    np.fill_diagonal(sym, 0.0)
    mat = np.rint(sym * scale).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"NAME: {path.stem}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {mat.shape[0]}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("DISPLAY_DATA_TYPE: TWOD_DISPLAY\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in mat:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
        f.write("DISPLAY_DATA_SECTION\n")
        scale_coords = np.rint(coords * scale).astype(int)
        for idx, (x, y) in enumerate(scale_coords, start=1):
            f.write(f"{idx} {x} {y}\n")
        f.write("EOF\n")


def main() -> None:
    args = parse_args()
    sampler = StructuredSampler(seed=args.seed)
    bbox = tuple(args.bbox)  # type: ignore[arg-type]
    if args.kind == "uniform":
        res = sampler.sample_uniform(n=args.n, bbox=bbox)
    elif args.kind == "clustered":
        res = sampler.sample_clustered(
            n=args.n,
            k=args.clusters,
            bbox=bbox,
            cov_min=args.cov_min,
            cov_max=args.cov_max,
        )
    elif args.kind == "random_explicit":
        # Generate dummy coordinates for display only; distances are random.
        rng = np.random.default_rng(args.seed)
        coords = rng.uniform(0.0, 1.0, size=(args.n, 2))
        # Build symmetric random matrix in [0,1]
        mat = rng.uniform(0.0, 1.0, size=(args.n, args.n))
        mat = np.triu(mat, 1)
        mat = mat + mat.T
        np.fill_diagonal(mat, 0.0)
        # Use explicit writer; store coords for display
        write_tsplib_explicit(mat, coords, args.tsp_path)
        meta = {
            "distribution": "random_explicit",
            "n": args.n,
            "seed": args.seed,
            "notes": "Symmetric explicit distance matrix with random U(0,1) entries; display coords are dummy.",
        }
        metadata: Dict[str, object] = {
            "id": args.tsp_path.stem,
            "file": args.tsp_path.name,
            "n": args.n,
            "split": f"structured_{args.kind}",
            "generator": "random_explicit",
            "seed": args.seed,
            "edge_weight_type": "explicit_random_uniform",
            "notes": meta["notes"],
        }
        print(metadata)
        return
    else:
        res = sampler.sample_grid(
            n=args.n,
            bbox=bbox,
            jitter=args.jitter,
            drop_prob=args.drop_prob,
        )

    metadata: Dict[str, object] = {
        "id": args.tsp_path.stem,
        "file": args.tsp_path.name,
        "n": args.n,
        "split": f"structured_{args.kind}",
        "generator": f"{args.kind}_structured",
        "bbox": bbox,
        "seed": args.seed,
    }
    if args.explicit_cluster_discount is not None:
        if args.kind != "clustered":
            raise ValueError("--explicit-cluster-discount is only supported for clustered instances.")
        labels = res.metadata.get("labels")
        if labels is None:
            raise ValueError("Cluster labels missing from sampler metadata.")
        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] != res.coords.shape[0]:
            raise ValueError("Labels length does not match number of points.")
        diff = res.coords[:, None, :] - res.coords[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        same_cluster = labels_arr[:, None] == labels_arr[None, :]
        dists = np.where(same_cluster, dists * args.explicit_cluster_discount, dists)
        write_tsplib_explicit(dists, res.coords, args.tsp_path)
        metadata["edge_weight_type"] = "explicit_discounted_euclidean"
        metadata["within_cluster_discount"] = args.explicit_cluster_discount
    else:
        write_tsplib(res.coords, args.tsp_path)
    print(metadata)


if __name__ == "__main__":
    main()
