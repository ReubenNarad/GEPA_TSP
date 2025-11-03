#!/usr/bin/env python3
"""
Visualize GEPA rollouts (wall-clock time vs. iteration) from a run directory.

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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


EVAL_DIR_PATTERN = re.compile(r".*_(\w+)$")  # matches label suffix


@dataclass
class Rollout:
    iteration: int
    label: str
    wall_time: Optional[float]
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


def load_rollouts(run_root: Path) -> List[Rollout]:
    eval_dir = run_root / "eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"No eval directory found at {eval_dir}")

    rollouts: List[Rollout] = []
    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        match = EVAL_DIR_PATTERN.match(subdir.name)
        label_suffix = match.group(1) if match else subdir.name

        iteration = parse_iteration(label_suffix)
        summary_path = subdir / "summary.json"
        wall_time = None
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text())
                wall_time = data.get("average_wall_time_sec")
            except (json.JSONDecodeError, OSError):
                wall_time = None

        rollouts.append(Rollout(iteration=iteration, label=label_suffix, wall_time=wall_time, path=subdir))

    return rollouts


def plot_rollouts(run_root: Path, rollouts: List[Rollout], title: str, output: Optional[Path]) -> None:
    xs = [r.iteration for r in rollouts if r.wall_time is not None]
    ys = [r.wall_time for r in rollouts if r.wall_time is not None]
    labels = [r.label for r in rollouts if r.wall_time is not None]

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, color="tab:blue")

    for x, y, lbl in zip(xs, ys, labels):
        plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    plt.title(title)
    plt.xlabel("GEPA iteration (-1 = baseline)")
    plt.ylabel("Average wall time (sec)")
    plt.grid(True, linestyle="--", alpha=0.4)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, bbox_inches="tight")
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

