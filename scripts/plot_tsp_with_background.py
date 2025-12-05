#!/usr/bin/env python3
"""Plot a TSPLIB instance with optional Concorde tour and OSM road background."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import networkit as nk  # type: ignore
import numpy as np
import pyrosm
from shapely.geometry import box


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot TSPLIB instance with tour and OSM backdrop.")
    p.add_argument("--tsp", type=Path, required=True, help="TSPLIB file (EUC_2D or EXPLICIT with coords).")
    p.add_argument("--sol", type=Path, default=None, help="Optional Concorde .sol file; if missing and --concorde-bin provided, it will be computed.")
    p.add_argument("--concorde-bin", type=Path, default=None, help="Path to concorde binary to compute tour.")
    p.add_argument("--pbf", type=Path, default=None, help="OSM PBF for backdrop (optional).")
    p.add_argument("--png", type=Path, required=True, help="Output PNG path.")
    p.add_argument("--nodes-json", type=Path, default=None, help="Sidecar JSON with node_ids (from generator).")
    p.add_argument("--line-alpha", type=float, default=0.8, help="Tour line alpha.")
    p.add_argument("--line-width", type=float, default=1.1, help="Tour line width.")
    p.add_argument("--city-size", type=float, default=8.0, help="Scatter size for cities.")
    p.add_argument("--plot-lon-lims", type=float, nargs=2, default=None, help="Optional lon min max for plot.")
    p.add_argument("--plot-lat-lims", type=float, nargs=2, default=None, help="Optional lat min max for plot.")
    return p.parse_args()


def load_tsp_coords(path: Path) -> np.ndarray:
    coords = []
    in_coords = False
    in_display = False
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NODE_COORD_SECTION"):
                in_coords, in_display = True, False
                continue
            if line.startswith("DISPLAY_DATA_SECTION"):
                in_display, in_coords = True, False
                continue
            if line == "EOF":
                break
            if in_coords or in_display:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    if not coords:
        raise ValueError(f"No coords parsed from {path}")
    return np.array(coords)


def ensure_tour(tsp: Path, sol: Path | None, concorde_bin: Path | None) -> Path:
    if sol and sol.exists():
        return sol
    if not concorde_bin:
        raise ValueError("No tour file provided and concorde_bin not set.")
    out = tsp.with_suffix(".sol")
    cmd = [str(concorde_bin.resolve()), "-o", str(out), str(tsp)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def load_sol(path: Path) -> np.ndarray:
    tokens = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0:
                continue  # length line
            tokens.extend(line.split())
    order = np.array([int(t) for t in tokens], dtype=int) - 1
    return order


def plot_osm_background(pbf: Path, bbox: Tuple[float, float, float, float]):
    if pbf is None:
        return
    lon_min, lon_max, lat_min, lat_max = bbox
    # pyrosm expects [lon_min, lat_min, lon_max, lat_max]
    osm = pyrosm.OSM(str(pbf), bounding_box=[lon_min, lat_min, lon_max, lat_max])
    _, edges = osm.get_network(network_type="driving", nodes=True)
    roi = box(lon_min, lat_min, lon_max, lat_max)
    geoms = edges.geometry
    xs = []
    ys = []
    for g in geoms:
        if g is None:
            continue
        inter = g.intersection(roi)
        if inter.is_empty:
            continue
        coords = getattr(inter, "coords", None)
        if coords:
            arr = np.asarray(coords)
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])
        elif inter.geom_type == "MultiLineString":
            for seg in inter:
                arr = np.asarray(seg.coords)
                xs.append(arr[:, 0])
                ys.append(arr[:, 1])
    for x, y in zip(xs, ys):
        plt.plot(x, y, color="0.8", linewidth=0.4, alpha=0.6, zorder=1)


def rebuild_graph(pbf: Path, bbox: Tuple[float, float, float, float]):
    lon_min, lon_max, lat_min, lat_max = bbox
    osm = pyrosm.OSM(str(pbf), bounding_box=[lon_min, lat_min, lon_max, lat_max])
    nodes_df, edges_df = osm.get_network(network_type="driving", nodes=True, extra_attributes=["maxspeed"])
    id_to_idx: dict[int, int] = {}
    idx_to_coords: dict[int, Tuple[float, float]] = {}
    for _, row in nodes_df.iterrows():
        nid = int(row["id"])
        idx = len(id_to_idx)
        id_to_idx[nid] = idx
        idx_to_coords[idx] = (float(row["lon"]), float(row["lat"]))
    g = nk.graph.Graph(n=len(id_to_idx), weighted=True, directed=False)
    for _, row in edges_df.iterrows():
        u = int(row["u"]); v = int(row["v"])
        if u not in id_to_idx or v not in id_to_idx:
            continue
        ui = id_to_idx[u]; vi = id_to_idx[v]
        length = row.get("length")
        if length is None or np.isnan(length):
            geom = row.get("geometry")
            length = geom.length if geom is not None else 1.0
        maxspeed = row.get("maxspeed")
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0]
        try:
            speed = float(str(maxspeed).split()[0]) / 3.6
        except Exception:
            speed = 35 / 3.6
        time_sec = float(length) / speed if speed > 0 else float(length) / (35 / 3.6)
        g.addEdge(ui, vi, w=time_sec)
    # Keep only largest component to improve reachability
    UG = nk.graphtools.toUndirected(g)
    comps = nk.components.ConnectedComponents(UG)
    comps.run()
    comp_list = comps.getComponents()
    largest = max(comp_list, key=len)
    keep = set(largest)
    remap = {old: new for new, old in enumerate(sorted(keep))}
    g2 = nk.graph.Graph(n=len(keep), weighted=True, directed=False)
    for u, v, w in g.iterEdgesWeights():
        if u in keep and v in keep:
            g2.addEdge(remap[u], remap[v], w=w)
    id_to_idx2 = {}
    idx_to_coords2 = {}
    for old_idx, (lon, lat) in idx_to_coords.items():
        if old_idx in keep:
            new_idx = remap[old_idx]
            idx_to_coords2[new_idx] = (lon, lat)
    # Rebuild id mapping filtered to keep nodes
    for nid, idx in id_to_idx.items():
        if idx in keep:
            id_to_idx2[nid] = remap[idx]
    return g2, id_to_idx2, idx_to_coords2


def shortest_path_polyline(g: nk.graph.Graph, id_to_idx: dict[int, int], idx_to_coords: dict[int, Tuple[float, float]], a_id: int, b_id: int):
    if a_id not in id_to_idx or b_id not in id_to_idx:
        return None
    a = id_to_idx[a_id]; b = id_to_idx[b_id]
    d = nk.distance.Dijkstra(g, a, storePaths=True, target=b)
    d.run()
    path = d.getPath(b)
    if not path:
        return None
    coords = np.array([idx_to_coords[i] for i in path])
    return coords


def main() -> None:
    args = parse_args()
    coords = load_tsp_coords(args.tsp)
    sol_path = ensure_tour(args.tsp, args.sol, args.concorde_bin)
    order = load_sol(sol_path)
    # Default bbox from coords; if nodes_json includes original bbox, prefer that.
    if args.nodes_json and args.nodes_json.exists():
        payload = json.loads(args.nodes_json.read_text())
        bb = payload.get("bbox")
        if bb and len(bb) == 4:
            lat_min, lat_max, lon_min, lon_max = bb
        else:
            lon_min, lon_max = float(coords[:, 0].min()), float(coords[:, 0].max())
            lat_min, lat_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    else:
        lon_min, lon_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        lat_min, lat_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    pad = 0.005
    lon_min -= pad
    lon_max += pad
    lat_min -= pad
    lat_max += pad

    plt.figure(figsize=(7, 7))
    if args.pbf:
        plot_osm_background(args.pbf, (lon_min, lon_max, lat_min, lat_max))

    used_path_overlay = False
    if args.nodes_json and args.pbf and args.nodes_json.exists():
        payload = json.loads(args.nodes_json.read_text())
        node_ids = payload.get("node_ids", [])
        if len(node_ids) == coords.shape[0]:
            g, id_to_idx, idx_to_coords = rebuild_graph(args.pbf, (lon_min, lon_max, lat_min, lat_max))
            for i in range(len(order)):
                a_id = node_ids[order[i]]
                b_id = node_ids[order[(i + 1) % len(order)]]
                poly = shortest_path_polyline(g, id_to_idx, idx_to_coords, a_id, b_id)
                if poly is not None:
                    plt.plot(poly[:, 0], poly[:, 1], color="tab:red", linewidth=args.line_width, alpha=args.line_alpha, zorder=3)
                else:
                    pa = coords[order[i]]
                    pb = coords[order[(i + 1) % len(order)]]
                    plt.plot([pa[0], pb[0]], [pa[1], pb[1]], color="tab:red", linewidth=args.line_width, alpha=args.line_alpha, zorder=3, linestyle="--")
            used_path_overlay = True

    if not used_path_overlay:
        tour = coords[order]
        tour = np.vstack([tour, tour[0]])
        plt.plot(tour[:, 0], tour[:, 1], color="tab:red", linewidth=args.line_width, alpha=args.line_alpha, label="Concorde tour", zorder=2)

    plt.scatter(coords[:, 0], coords[:, 1], s=args.city_size, c="tab:blue", alpha=0.9, label="cities", zorder=4)
    # Highlight start node
    start_xy = coords[order[0]]
    plt.scatter([start_xy[0]], [start_xy[1]], s=args.city_size * 1.8, facecolors="none", edgecolors="black", linewidths=1.4, zorder=5)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.title(args.tsp.stem)
    plt.axis("equal")
    if args.plot_lon_lims:
        plt.xlim(args.plot_lon_lims)
    if args.plot_lat_lims:
        plt.ylim(args.plot_lat_lims)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.png, dpi=250)
    plt.close()
    print(f"Saved {args.png}")


if __name__ == "__main__":
    main()
