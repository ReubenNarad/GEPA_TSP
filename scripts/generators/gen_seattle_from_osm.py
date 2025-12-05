#!/usr/bin/env python3
"""
Generate a TSPLIB instance from a local Seattle OSM extract using road-network travel times.

Inputs:
  - Local PBF: data/osm/Seattle.osm.pbf (downloaded separately)

Outputs:
  - TSPLIB .tsp with FULL_MATRIX edge weights (seconds, rounded to int)
  - Optional PNG scatter of sampled nodes
  - Metadata JSON printed to stdout for easy insertion into data/eval/metadata.json

Example:
  python scripts/generators/gen_seattle_from_osm.py \
      --pbf data/osm/Seattle.osm.pbf \
      --bbox 47.58 47.64 -122.36 -122.30 \
      --n 300 \
      --seed 20251124 \
      --tsp-path data/eval/structured_seattle_300_seed20251124.tsp \
      --png out/structured_seattle_300_seed20251124.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkit as nk  # type: ignore
import numpy as np
import pyrosm
from shapely.geometry import LineString


BBox = Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Seattle road-network TSP instance.")
    p.add_argument("--pbf", type=Path, required=True, help="Path to Seattle .osm.pbf extract.")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"), required=True)
    p.add_argument("--n", type=int, default=300, help="Number of cities to sample.")
    p.add_argument("--seed", type=int, default=20251124)
    p.add_argument("--tsp-path", type=Path, required=True, help="Output TSPLIB path.")
    p.add_argument("--png", type=Path, default=None, help="Optional PNG scatter output.")
    p.add_argument(
        "--weight",
        choices=["time", "distance"],
        default="time",
        help="Edge weight: travel time (seconds, default) or road distance (meters).",
    )
    p.add_argument(
        "--sample-bbox",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        default=None,
        help="Optional sub-bbox to restrict sampled nodes (lat_min lat_max lon_min lon_max).",
    )
    p.add_argument(
        "--sample-mode",
        choices=["uniform", "bfs"],
        default="uniform",
        help="How to pick nodes inside the largest component (default: uniform random; bfs = connected ball).",
    )
    return p.parse_args()


def load_graph(pbf: Path, bbox: BBox):
    lat_min, lat_max, lon_min, lon_max = bbox
    bb = [lon_min, lat_min, lon_max, lat_max]  # pyrosm expects lon/lat order
    osm = pyrosm.OSM(str(pbf), bounding_box=bb)
    nodes, edges = osm.get_network(network_type="driving", nodes=True, extra_attributes=["maxspeed"])
    return nodes, edges


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
        return 35 / 3.6  # default 35 km/h
    if isinstance(ms, list):
        # take first
        ms = ms[0]
    try:
        if isinstance(ms, str):
            ms = ms.split()[0]
        v = float(ms)
        return v / 3.6  # km/h -> m/s
    except Exception:
        return 35 / 3.6


def build_graph(nodes_df, edges_df, weight: str):
    id_to_idx: Dict[int, int] = {}
    idx_to_id: Dict[int, int] = {}
    lats = []
    lons = []
    for idx, row in nodes_df.iterrows():
        nid = int(row["id"])
        cur = len(id_to_idx)
        id_to_idx[nid] = cur
        idx_to_id[cur] = nid
        lats.append(float(row["lat"]))
        lons.append(float(row["lon"]))

    g = nk.graph.Graph(n=len(id_to_idx), weighted=True, directed=False)
    lengths = []
    speeds = []
    for _, row in edges_df.iterrows():
        u = row["u"]
        v = row["v"]
        if u not in id_to_idx or v not in id_to_idx:
            continue
        ui = id_to_idx[u]
        vi = id_to_idx[v]
        length = compute_length(row)
        if weight == "time":
            speed = parse_maxspeed(row.get("maxspeed"))
            w = length / speed if speed > 0 else length / (35 / 3.6)
            speeds.append(speed)
        else:
            w = length
        g.addEdge(ui, vi, w=w)
        lengths.append(length)
    stats = {
        "nodes": g.numberOfNodes(),
        "edges": g.numberOfEdges(),
        "avg_length_m": float(np.mean(lengths)) if lengths else None,
        "avg_speed_mps": float(np.mean(speeds)) if speeds and weight == "time" else None,
        "weight": weight,
    }
    coords = np.column_stack([np.array(lons), np.array(lats)])  # lon, lat
    return g, coords, stats, idx_to_id


def largest_component_subgraph(g: nk.graph.Graph, coords: np.ndarray, idx_to_id: Dict[int, int]):
    UG = nk.graphtools.toUndirected(g)
    comps = nk.components.ConnectedComponents(UG)
    comps.run()
    comps_list = comps.getComponents()
    sizes = [len(c) for c in comps_list]
    largest = max(range(len(sizes)), key=lambda c: sizes[c])
    keep_nodes = comps_list[largest]
    mapping = {old: new for new, old in enumerate(keep_nodes)}
    sub = nk.graph.Graph(n=len(keep_nodes), weighted=True, directed=False)
    for u, v, w in g.iterEdgesWeights():
        if u in mapping and v in mapping:
            sub.addEdge(mapping[u], mapping[v], w=w)
    sub_coords = coords[keep_nodes]
    sub_ids = [idx_to_id[i] for i in keep_nodes]
    keep_set = set(keep_nodes)
    return sub, sub_coords, sub_ids, keep_set


def sample_nodes(coords: np.ndarray, n: int, seed: int, allowed_idx: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    pool = allowed_idx if allowed_idx is not None else np.arange(coords.shape[0], dtype=int)
    total = pool.shape[0]
    if n > total:
        raise ValueError(f"Requested {n} nodes, but pool has only {total}")
    idx = rng.choice(pool, size=n, replace=False)
    return idx


def bfs_connected_sample(g: nk.graph.Graph, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = rng.integers(0, g.numberOfNodes())
    visited = []
    seen = set()
    queue = [int(start)]
    while queue and len(visited) < n:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        visited.append(cur)
        for v in g.iterNeighbors(cur):
            if v not in seen:
                queue.append(int(v))
    if len(visited) < n:
        # if graph is too small (shouldn't happen), pad with random seen nodes
        extra = rng.choice(list(seen), size=n - len(visited), replace=False)
        visited.extend(extra.tolist())
    return np.array(visited[:n], dtype=int)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def all_pairs_dijkstra(g: nk.graph.Graph, sources: np.ndarray, coords: np.ndarray, default_kmh: float = 35.0) -> np.ndarray:
    n = len(sources)
    dist_matrix = np.zeros((n, n), dtype=float)
    inf_repl = 1e12
    default_speed = default_kmh / 3.6
    for i, node in enumerate(sources):
        d = nk.distance.Dijkstra(g, int(node))
        d.run()
        for j, target in enumerate(sources):
            dist = d.distance(int(target))
            if np.isinf(dist):
                # fallback to straight-line travel time
                lat1, lon1 = coords[int(node)][1], coords[int(node)][0]
                lat2, lon2 = coords[int(target)][1], coords[int(target)][0]
                meters = haversine_m(lat1, lon1, lat2, lon2)
                dist_matrix[i, j] = meters / default_speed
            else:
                dist_matrix[i, j] = dist
    return dist_matrix


def write_tsplib(full_matrix: np.ndarray, coords: np.ndarray, path: Path) -> None:
    # Symmetrize (undirected TSP); use min of i->j and j->i.
    sym = np.minimum(full_matrix, full_matrix.T)
    inf_repl = 1e12
    sym = np.nan_to_num(sym, nan=inf_repl, posinf=inf_repl, neginf=inf_repl)
    sym[sym < 0] = inf_repl
    sym[sym > inf_repl] = inf_repl
    mat = np.rint(sym).astype(np.int64)
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
        # write lon/lat for reference
        for idx, (lon, lat) in enumerate(coords, start=1):
            f.write(f"{idx} {lon:.6f} {lat:.6f}\n")
        f.write("EOF\n")


def plot_points(coords: np.ndarray, path: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=6, c="tab:blue", alpha=0.9)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.title(path.stem)
    plt.axis("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    bbox = tuple(args.bbox)  # type: ignore
    nodes_df, edges_df = load_graph(args.pbf, bbox)
    g_full, coords_all, stats, idx_to_id = build_graph(nodes_df, edges_df, args.weight)
    g, coords_all, node_ids_all, keep_set = largest_component_subgraph(g_full, coords_all, idx_to_id)

    allowed_idx = None
    sample_bb = None
    if args.sample_bbox:
        lat_min_s, lat_max_s, lon_min_s, lon_max_s = args.sample_bbox
        sample_bb = args.sample_bbox
        mask = (
            (coords_all[:, 0] >= lon_min_s)
            & (coords_all[:, 0] <= lon_max_s)
            & (coords_all[:, 1] >= lat_min_s)
            & (coords_all[:, 1] <= lat_max_s)
        )
        allowed_idx = np.nonzero(mask)[0]
    if allowed_idx is not None:
        sample_idx = sample_nodes(coords_all, n=args.n, seed=args.seed, allowed_idx=allowed_idx)
    else:
        if args.sample_mode == "bfs":
            sample_idx = bfs_connected_sample(g, n=args.n, seed=args.seed)
        else:
            rng = np.random.default_rng(args.seed)
            if g.numberOfNodes() < args.n:
                raise ValueError(f"Requested {args.n} nodes, but graph has only {g.numberOfNodes()}")
            sample_idx = rng.choice(g.numberOfNodes(), size=args.n, replace=False)
    node_ids = [int(node_ids_all[i]) for i in sample_idx]
    coords = coords_all[sample_idx]
    dist = all_pairs_dijkstra(g, sample_idx, coords_all)

    write_tsplib(dist, coords, args.tsp_path)
    # sidecar for plotting/repro
    sidecar = args.tsp_path.with_suffix(".nodes.json")
    sidecar.write_text(json.dumps({"node_ids": node_ids, "bbox": bbox, "seed": args.seed}, indent=2))
    if args.png:
        plot_points(coords, args.png)

    metadata = {
        "id": args.tsp_path.stem,
        "file": args.tsp_path.name,
        "n": args.n,
        "split": "structured_seattle",
        "generator": "seattle_osm_traveltime",
        "bbox": bbox,
        "seed": args.seed,
        "weight": args.weight,
        "sample_bbox": sample_bb if sample_bb else None,
        "notes": "travel-time shortest paths (seconds) on drive network" if args.weight == "time" else "road distance shortest paths (meters)",
    }
    print(json.dumps({"metadata_entry": metadata, "graph_stats": stats}, indent=2))


if __name__ == "__main__":
    main()
