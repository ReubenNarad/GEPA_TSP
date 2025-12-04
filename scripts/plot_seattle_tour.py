#!/usr/bin/env python3
"""
Plot a Seattle travel-time TSP instance with the optimal tour snapped to roads.
Uses OSM data to render the road background and reconstructs shortest paths
between tour cities on the driving network.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import networkit as nk  # type: ignore
import numpy as np
import pyrosm
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Seattle TSP tour snapped to roads.")
    p.add_argument("--tsp", type=Path, required=True, help="Path to the Seattle TSPLIB instance.")
    p.add_argument("--pbf", type=Path, default=Path("data/osm/Seattle.osm.pbf"), help="OSM PBF path.")
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        default=[47.58, 47.64, -122.36, -122.30],
        help="Bounding box (lat_min lat_max lon_min lon_max).",
    )
    p.add_argument("--pad", type=float, default=0.01, help="Padding added to each side of bbox.")
    p.add_argument("--output", type=Path, required=True, help="PNG output path.")
    p.add_argument(
        "--concorde",
        type=Path,
        default=Path("concorde/install/bin/concorde"),
        help="Path to Concorde binary.",
    )
    return p.parse_args()


def compute_length(row) -> float:
    if "length" in row and row["length"] is not None:
        return float(row["length"])
    geom = row.get("geometry")
    if isinstance(geom, LineString):
        return float(geom.length)
    return 0.0


def parse_maxspeed(ms) -> float:
    # returns m/s
    if ms is None:
        return 35 / 3.6
    if isinstance(ms, list):
        ms = ms[0]
    try:
        if isinstance(ms, str):
            ms = ms.split()[0]
        v = float(ms)
        return v / 3.6
    except Exception:
        return 35 / 3.6


def main() -> None:
    args = parse_args()
    lat_min, lat_max, lon_min, lon_max = args.bbox
    lat_min -= args.pad
    lat_max += args.pad
    lon_min -= args.pad
    lon_max += args.pad

    # Solve tour with Concorde
    workdir = Path(tempfile.mkdtemp(prefix="plot_seattle_"))
    try:
        local_tsp = workdir / args.tsp.name
        shutil.copy2(args.tsp, local_tsp)
        res = subprocess.run([str(args.concorde), str(local_tsp)], cwd=workdir, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"Concorde failed: {res.stderr}")
        tour_file = local_tsp.with_suffix(".tour")
        if not tour_file.exists():
            tour_file = local_tsp.with_suffix(".sol")
        entries = tour_file.read_text().split()
        n = int(entries[0])
        ids = [int(x) for x in entries[1 : 1 + n]]
        tour = ids + [ids[0]]

        # Parse display coords (lon, lat)
        lines = args.tsp.read_text().splitlines()
        idx = lines.index("DISPLAY_DATA_SECTION") + 1
        coords = []
        for line in lines[idx:]:
            if line.strip().upper() == "EOF":
                break
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
        coords = np.array(coords)

        # Load OSM network in padded bbox
        bb = [lon_min, lat_min, lon_max, lat_max]  # pyrosm expects lon/lat order
        osm = pyrosm.OSM(str(args.pbf), bounding_box=bb)
        nodes_df, edges_df = osm.get_network(network_type="driving", nodes=True, extra_attributes=["maxspeed"])

        id_to_idx = {}
        idx_to_coords = {}
        for idx2, row in enumerate(nodes_df.itertuples()):
            nid = int(row.id)
            id_to_idx[nid] = idx2
            idx_to_coords[idx2] = (float(row.lon), float(row.lat))
        g = nk.graph.Graph(n=len(id_to_idx), weighted=True, directed=False)

        for _, row in edges_df.iterrows():
            u, v = row["u"], row["v"]
            if u not in id_to_idx or v not in id_to_idx:
                continue
            ui, vi = id_to_idx[u], id_to_idx[v]
            length = compute_length(row)
            speed = parse_maxspeed(row.get("maxspeed"))
            w = length / speed if speed > 0 else length / (35 / 3.6)
            g.addEdge(ui, vi, w=w)

        tree = cKDTree(np.array(list(idx_to_coords.values())))
        city_to_graph = []
        for lon, lat in coords:
            _, idx = tree.query([lon, lat])
            city_to_graph.append(idx)

        paths = []
        for a, b in zip(tour[:-1], tour[1:]):
            src = city_to_graph[a - 1]
            tgt = city_to_graph[b - 1]
            dijkstra = nk.distance.Dijkstra(g, src, storePaths=True)
            dijkstra.run()
            seq = dijkstra.getPath(tgt)
            if not seq:
                continue
            path_coords = np.array([idx_to_coords[i] for i in seq])
            paths.append(path_coords)

        fig, ax = plt.subplots(figsize=(8, 8))
        # Background roads (darker)
        for geom in edges_df["geometry"]:
            if isinstance(geom, LineString):
                x, y = geom.xy
                ax.plot(x, y, color="lightgray", linewidth=0.5, alpha=0.9, zorder=1)
            elif isinstance(geom, MultiLineString):
                for part in geom.geoms:
                    x, y = part.xy
                    ax.plot(x, y, color="lightgray", linewidth=0.5, alpha=0.9, zorder=1)
        for seg in paths:
            ax.plot(seg[:, 0], seg[:, 1], "-", color="tab:red", linewidth=1.2, alpha=0.9, zorder=3)
        ax.scatter(coords[:, 0], coords[:, 1], s=8, color="tab:blue", alpha=0.9, zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved {args.output}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
