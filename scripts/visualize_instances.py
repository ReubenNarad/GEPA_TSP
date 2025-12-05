#!/usr/bin/env python3
"""Visualize synthetic TSP point sets; optionally overlay Concorde tour.

Usage:
  python scripts/visualize_instances.py --kind clustered --n 300 --seed 1234 --output out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from instance_sampler import StructuredSampler  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize synthetic TSP distributions.")
    parser.add_argument("--kind", choices=["uniform", "clustered", "grid"], required=True)
    parser.add_argument("--n", type=int, default=300, help="Number of points.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True, help="PNG path to save.")
    parser.add_argument("--bbox", type=float, nargs=4, default=[0.0, 1.0, 0.0, 1.0], help="xmin xmax ymin ymax")
    parser.add_argument("--clusters", type=int, default=4, help="K for clustered.")
    parser.add_argument("--cov-min", type=float, default=0.001, help="Min eigenvalue for clustered cov.")
    parser.add_argument("--cov-max", type=float, default=0.02, help="Max eigenvalue for clustered cov.")
    parser.add_argument("--jitter", type=float, default=0.01, help="Grid jitter.")
    parser.add_argument("--drop-prob", type=float, default=0.0, help="Grid row/col drop probability.")
    parser.add_argument("--concorde-bin", type=Path, default=None, help="Path to concorde binary to compute tour overlay.")
    return parser.parse_args()


def write_tsplib(coords: np.ndarray, path: Path) -> None:
    """Write coords to a TSPLIB EUC_2D file (scaled to integers)."""
    scale = 100000.0
    ints = np.rint(coords * scale).astype(int)
    with path.open("w") as f:
        f.write("NAME: viz\n")
        f.write("TYPE: TSP\n")
        f.write("DIMENSION: {}\n".format(len(ints)))
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(ints, start=1):
            f.write(f"{idx} {x} {y}\n")
        f.write("EOF\n")


def run_concorde(concorde_bin: Path, tsp_path: Path, workdir: Path) -> list[int]:
    tour_path = workdir / "tour.sol"
    bin_path = concorde_bin.resolve()
    cmd = [str(bin_path), "-o", str(tour_path), str(tsp_path)]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Concorde failed: {proc.stderr}")
    if not tour_path.exists():
        raise RuntimeError("Tour file missing after Concorde run.")
    with tour_path.open() as f:
        lines = [line.strip() for line in f if line.strip()]
    tokens = []
    for line in lines[1:]:
        tokens.extend(line.split())
    nodes = [int(x) for x in tokens]
    return nodes


def main() -> None:
    args = parse_args()
    sampler = StructuredSampler(seed=args.seed)
    bbox = tuple(args.bbox)  # type: ignore[arg-type]

    if args.kind == "uniform":
        res = sampler.sample_uniform(n=args.n, bbox=bbox)
        title = f"Uniform (n={args.n})"
        colors = "tab:blue"
    elif args.kind == "clustered":
        res = sampler.sample_clustered(
            n=args.n,
            k=args.clusters,
            bbox=bbox,
            cov_min=args.cov_min,
            cov_max=args.cov_max,
        )
        labels = np.array(res.metadata.get("labels", []))
        # Cycle a small palette
        palette = np.array(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])
        colors = palette[labels % len(palette)] if len(labels) == res.coords.shape[0] else "tab:blue"
        title = f"Clustered (k={args.clusters}, n={args.n})"
    else:  # grid
        res = sampler.sample_grid(
            n=args.n,
            bbox=bbox,
            jitter=args.jitter,
            drop_prob=args.drop_prob,
        )
        colors = "tab:blue"
        title = f"Grid (n={args.n}, jitter={args.jitter}, drop={args.drop_prob})"

    coords = res.coords
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=10, c=colors, alpha=0.8, label="cities")

    if args.concorde_bin:
        with tempfile.TemporaryDirectory() as tmpdir:
            tsp_path = Path(tmpdir) / "instance.tsp"
            write_tsplib(coords, tsp_path)
            tour = run_concorde(args.concorde_bin, tsp_path, Path(tmpdir))
        order = np.array(tour, dtype=int)
        tour_coords = coords[order]
        # close the loop
        tour_coords = np.vstack([tour_coords, tour_coords[0]])
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], color="tab:red", linewidth=1.2, alpha=0.9, label="Concorde tour")
        title = title + " + OPT tour"

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    if args.concorde_bin:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
