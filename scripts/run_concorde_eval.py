#!/usr/bin/env python3
"""
Utility to benchmark a Concorde binary against a set of TSP instances.

Usage example:
  python scripts/run_concorde_eval.py \
      --binary concorde/install/bin/concorde \
      --metadata data/eval/metadata.json \
      --split toy20 \
      --label baseline
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional


_TOTAL_TIME_RE = re.compile(r"Total Running Time:\s*([0-9.]+)")
_BBNODE_RE = re.compile(r"Number of bbnodes:\s*([0-9]+)")
_OPT_RE = re.compile(r"Optimal Solution:\s*([-+]?[0-9.]+)")
_FINAL_BOUNDS_RE = re.compile(
    r"Final lower bound\s+([-+]?[0-9.]+),\s+upper bound\s+([-+]?[0-9.]+)"
)
_LP_VALUE_RE = re.compile(
    r"LP Value\s+([0-9]+):\s+([-+]?[0-9.]+)\s+\(([0-9.]+)\s+seconds\)"
)
_PROBLEM_NAME_RE = re.compile(r"Problem Name:\s*([A-Za-z0-9_\-\.]+)")


def _load_instances(metadata_path: Path) -> List[Dict[str, Any]]:
    with metadata_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict) or "instances" not in data:
        raise ValueError(f"Malformed metadata file: {metadata_path}")
    instances = data["instances"]
    if not isinstance(instances, list):
        raise ValueError(f"'instances' must be a list in {metadata_path}")
    return instances


def _filter_instances(
    instances: Iterable[Dict[str, Any]],
    ids: Optional[Iterable[str]],
    splits: Optional[Iterable[str]],
    max_instances: Optional[int],
) -> List[Dict[str, Any]]:
    selected = []
    ids_set = set(ids) if ids else None
    splits_set = set(splits) if splits else None
    for inst in instances:
        inst_id = inst.get("id")
        inst_split = inst.get("split")
        if ids_set is not None and inst_id not in ids_set:
            continue
        if splits_set is not None and inst_split not in splits_set:
            continue
        selected.append(inst)
    if ids_set:
        missing = ids_set - {inst.get("id") for inst in selected}
        if missing:
            raise ValueError(f"Requested ids not found in metadata: {sorted(missing)}")
    if max_instances is not None:
        selected = selected[:max_instances]
    return selected


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _run_instance(
    binary: Path,
    instance_path: Path,
    timeout: Optional[float],
) -> Dict[str, Any]:
    cmd = [str(binary), str(instance_path)]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        return {
            "cmd": cmd,
            "wall_time": duration,
            "timeout": timeout,
            "timed_out": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "returncode": None,
            "metrics": {},
        }
    duration = time.perf_counter() - start

    stdout = proc.stdout or ""
    metrics: Dict[str, Any] = {}

    if m := _TOTAL_TIME_RE.search(stdout):
        metrics["total_running_time_sec"] = float(m.group(1))
    if m := _BBNODE_RE.search(stdout):
        metrics["bb_nodes"] = int(m.group(1))
    if m := _OPT_RE.search(stdout):
        metrics["optimal_solution"] = float(m.group(1))
    if m := _FINAL_BOUNDS_RE.search(stdout):
        metrics["final_lower_bound"] = float(m.group(1))
        metrics["final_upper_bound"] = float(m.group(2))
    lp_values = []
    for m in _LP_VALUE_RE.finditer(stdout):
        lp_values.append(
            {
                "index": int(m.group(1)),
                "value": float(m.group(2)),
                "time_sec": float(m.group(3)),
            }
        )
    if lp_values:
        metrics["lp_values"] = lp_values

    result = {
        "cmd": cmd,
        "wall_time": duration,
        "timeout": timeout,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": proc.stderr or "",
        "returncode": proc.returncode,
        "metrics": metrics,
    }

    cleanup_names = {instance_path.stem}
    if m := _PROBLEM_NAME_RE.search(stdout):
        cleanup_names.add(m.group(1).strip())
    unique_dirs = {Path.cwd(), instance_path.parent}
    for name in cleanup_names:
        for directory in unique_dirs:
            for suffix in (".sol", ".res"):
                candidate = directory / f"{name}{suffix}"
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass

    return result


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    failures = sum(1 for r in results if r["returncode"] not in (0, None))
    timeouts = sum(1 for r in results if r.get("timed_out"))

    def avg(key: str) -> Optional[float]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def avg_metric(metric_key: str) -> Optional[float]:
        vals = []
        for r in results:
            metrics = r.get("metrics") or {}
            value = metrics.get(metric_key)
            if value is not None:
                vals.append(float(value))
        if not vals:
            return None
        return sum(vals) / len(vals)

    summary: Dict[str, Any] = {
        "total_instances": total,
        "failures": failures,
        "timeouts": timeouts,
        "average_wall_time_sec": avg("wall_time"),
        "average_total_running_time_sec": avg_metric("total_running_time_sec"),
        "average_bb_nodes": avg_metric("bb_nodes"),
    }
    return summary


def _save_run_artifacts(
    run_dir: Path,
    config: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    _ensure_dir(run_dir / "instances")

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    results_path = run_dir / "results.jsonl"
    with results_path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    for result in results:
        inst_id = result["instance_id"]
        repeat = result["repeat"]
        prefix = f"{inst_id}_r{repeat}"
        (run_dir / "instances" / f"{prefix}.stdout").write_text(result["stdout"])
        (run_dir / "instances" / f"{prefix}.stderr").write_text(result["stderr"])

    summary = _summarize(results)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Concorde on an evaluation set and collect metrics."
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("concorde/install/bin/concorde"),
        help="Path to the Concorde executable (default: concorde/install/bin/concorde).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/eval/metadata.json"),
        help="Path to the dataset metadata JSON.",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="Filter instances by split label (can be used multiple times).",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Explicit instance ids to run (overrides --split).",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit the number of instances to run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repetitions per instance (default: 1).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for each Concorde run.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label to embed in the run directory name.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory for run artifacts (default: runs/).",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Optional JSON string to attach environment metadata.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    binary = args.binary.resolve()
    metadata_path = args.metadata.resolve()

    if not binary.exists():
        print(f"error: Concorde binary not found: {binary}", file=sys.stderr)
        return 1
    if not os.access(binary, os.X_OK):
        print(f"error: Concorde binary is not executable: {binary}", file=sys.stderr)
        return 1
    if not metadata_path.exists():
        print(f"error: metadata file not found: {metadata_path}", file=sys.stderr)
        return 1

    instances = _load_instances(metadata_path)
    selected = _filter_instances(
        instances,
        ids=args.ids,
        splits=args.splits,
        max_instances=args.max_instances,
    )
    if not selected:
        print("No instances selected. Check filters.", file=sys.stderr)
        return 1

    metadata_dir = metadata_path.parent
    timestamp = _timestamp()
    label = args.label.replace(" ", "_") if args.label else "eval"
    run_dir = (args.run_dir / "eval" / f"{timestamp}_{label}").resolve()
    _ensure_dir(run_dir)

    environment = None
    if args.environment:
        try:
            environment = json.loads(args.environment)
        except json.JSONDecodeError as err:
            print(f"error parsing --environment JSON: {err}", file=sys.stderr)
            return 1

    config: Dict[str, Any] = {
        "timestamp": timestamp,
        "binary": str(binary),
        "metadata": str(metadata_path),
        "selected_ids": [inst["id"] for inst in selected],
        "repeats": args.repeats,
        "timeout": args.timeout,
    }
    if environment is not None:
        config["environment"] = environment

    results: List[Dict[str, Any]] = []

    print(f"Running {len(selected)} instance(s) x {args.repeats} repeat(s)")
    print(f"Binary: {binary}")
    print(f"Run directory: {run_dir}")

    for inst in selected:
        inst_id = inst.get("id")
        file_rel = inst.get("file")
        if file_rel is None:
            print(f"Skipping instance with no file entry: {inst}", file=sys.stderr)
            continue
        instance_path = (metadata_dir / file_rel).resolve()
        if not instance_path.exists():
            print(
                f"warning: instance file missing ({inst_id} -> {instance_path}), skipping",
                file=sys.stderr,
            )
            continue

        for repeat in range(1, args.repeats + 1):
            print(f"[{inst_id}] repeat {repeat}/{args.repeats} ... ", end="", flush=True)
            run_info = _run_instance(binary, instance_path, args.timeout)
            run_info["instance_id"] = inst_id
            run_info["repeat"] = repeat
            run_info["instance_file"] = str(instance_path)
            run_info["metadata"] = inst

            ok = (run_info["returncode"] == 0) and not run_info.get("timed_out", False)
            status = "ok" if ok else "FAIL"
            wall_time = run_info.get("wall_time")
            if wall_time is not None:
                status += f" ({wall_time:.3f}s)"
            print(status)

            results.append(run_info)

    _save_run_artifacts(run_dir, config, results)

    summary = _summarize(results)
    print("\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nArtifacts saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
