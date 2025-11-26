#!/usr/bin/env python3
"""
Visualize GEPA rollouts (wall-clock time vs. iteration) from a run directory, with optional
rolling averages and confidence intervals when multiple repeats were run.

Usage:
    python scripts/plot_gepa_metrics.py \
        --run runs/gepa/20251103T185401Z_gepa_demo \
        --output gepa_plot.png

The script scans run_root/eval/ for evaluation subdirectories, extracts the
average wall time from summary.json, and plots one point per rollout. The
baseline run is treated as iteration -1.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt


EVAL_DIR_PATTERN = re.compile(r".*_(\w+)$")  # matches label suffix


@dataclass
class Rollout:
    iteration: int
    label: str
    wall_time: Optional[float]
    wall_ci: Optional[float]
    bb_nodes: Optional[float]
    bb_ci: Optional[float]
    path: Path


def parse_iteration(label: str) -> int:
    if label.endswith("_baseline"):
        return -1
    tokens = label.split("_")
    for token in reversed(tokens):
        if token.startswith("iter") or token.startswith("ee"):
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                return int(digits)
    return 0


def _ci_90(values: List[float]) -> Optional[float]:
    if len(values) <= 1:
        return None
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)  # population stdev across repeats/instances
    return 1.645 * stdev / (len(values) ** 0.5)


def _load_metrics(subdir: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return mean wall, wall CI, mean bb, bb CI from results.jsonl if present, else summary.json."""
    results_path = subdir / "results.jsonl"
    walls: List[float] = []
    bbs: List[float] = []
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            w = rec.get("wall_time")
            if isinstance(w, (int, float)):
                walls.append(float(w))
            bb = (rec.get("metrics") or {}).get("bb_nodes")
            if isinstance(bb, (int, float)):
                bbs.append(float(bb))
    summary_path = subdir / "summary.json"
    wall_mean = None
    bb_mean = None
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
            wall_mean = data.get("average_wall_time_sec")
            bb_mean = data.get("average_bb_nodes")
        except json.JSONDecodeError:
            pass

    def pick_mean(values: List[float], fallback: Optional[float]) -> Optional[float]:
        return statistics.mean(values) if values else fallback

    wall = pick_mean(walls, wall_mean)
    bb = pick_mean(bbs, bb_mean)
    wall_ci = _ci_90(walls)
    bb_ci = _ci_90(bbs)
    return wall, wall_ci, bb, bb_ci


def load_rollouts(run_root: Path) -> List[Rollout]:
    eval_dir = run_root / "eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"No eval directory found at {eval_dir}")

    rollouts: List[Rollout] = []
    iteration_counter = 0

    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        match = EVAL_DIR_PATTERN.match(subdir.name)
        label_suffix = match.group(1) if match else subdir.name

        if label_suffix == "baseline":
            iteration = -1
        else:
            iteration_counter += 1
            iteration = iteration_counter

        wall, wall_ci, bb, bb_ci = _load_metrics(subdir)
        rollouts.append(
            Rollout(
                iteration=iteration,
                label=label_suffix,
                wall_time=wall,
                wall_ci=wall_ci,
                bb_nodes=bb,
                bb_ci=bb_ci,
                path=subdir,
            )
        )

    return rollouts


def _rolling(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(len(values)):
        window_vals = [v for v in values[max(0, i - window + 1) : i + 1] if v is not None]
        out.append(sum(window_vals) / len(window_vals) if window_vals else None)
    return out


def plot_rollouts(run_root: Path, rollouts: List[Rollout], title: str, output: Optional[Path], window: int = 5) -> None:
    xs = [r.iteration for r in rollouts if r.wall_time is not None]
    wall = [float(r.wall_time) for r in rollouts if r.wall_time is not None]
    wall_ci = [r.wall_ci for r in rollouts if r.wall_time is not None]
    bb = [float(r.bb_nodes) if r.bb_nodes is not None else np.nan for r in rollouts if r.wall_time is not None]
    bb_ci = [r.bb_ci for r in rollouts if r.wall_time is not None]

    wall_sm = _rolling(wall, window)
    bb_sm = _rolling(bb, window)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Wall time
    axes[0].errorbar(
        xs,
        wall,
        yerr=[c if c is not None else 0 for c in wall_ci],
        fmt="o",
        color="tab:blue",
        alpha=0.4,
        label="Wall time (raw)",
        capsize=3,
    )
    axes[0].plot(xs, wall_sm, color="tab:blue", label=f"Wall time (window={window})")

    # Baseline (iteration -1)
    if rollouts:
        base = next((r for r in rollouts if r.iteration == -1), None)
        if base and base.wall_time is not None:
            axes[0].axhline(base.wall_time, color="tab:red", linestyle="--", label="Baseline")

    axes[0].set_xlabel("GEPA iteration (-1 = baseline)")
    axes[0].set_ylabel("Average wall time (sec)")
    axes[0].set_title("Wall time")
    axes[0].legend()

    # BB nodes
    axes[1].errorbar(
        xs,
        bb,
        yerr=[c if c is not None else 0 for c in bb_ci],
        fmt="o",
        color="tab:green",
        alpha=0.4,
        label="BB nodes (raw)",
        capsize=3,
    )
    axes[1].plot(xs, bb_sm, color="tab:green", label=f"BB nodes (window={window})")
    if rollouts:
        base = next((r for r in rollouts if r.iteration == -1), None)
        if base and base.bb_nodes is not None:
            axes[1].axhline(base.bb_nodes, color="tab:red", linestyle="--", label="Baseline")

    axes[1].set_xlabel("GEPA iteration (-1 = baseline)")
    axes[1].set_ylabel("Average BB nodes")
    axes[1].set_title("BB nodes")
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GEPA rollout wall times.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory (runs/gepa/.../).")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the plot image.")
    parser.add_argument("--title", default=None, help="Optional plot title (defaults to run directory name).")
    args = parser.parse_args()

    run_root = args.run.resolve()
    summary_path = run_root / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found; ensure this is a GEPA run directory.")

    rollouts = load_rollouts(run_root)
    if not rollouts:
        raise SystemExit(f"No evaluation subdirectories found under {run_root / 'eval'}")

    title = args.title or f"GEPA rollouts for {run_root.name}"
    plot_rollouts(run_root, rollouts, title, args.output)


if __name__ == "__main__":
    main()
