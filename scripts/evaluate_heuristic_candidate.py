#!/usr/bin/env python3
"""
CLI helper for evaluating Lin-Kernighan heuristic candidates in a sandbox.

Examples:
  python scripts/evaluate_heuristic_candidate.py \
      --candidate-file path/to/block.c \
      --label trial01

  cat my_block.c | python scripts/evaluate_heuristic_candidate.py --label foo

  python scripts/evaluate_heuristic_candidate.py --use-default --label baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from heuristic_bridge import evaluate_candidate  # type: ignore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a Lin-Kernighan heuristic candidate.")
    parser.add_argument(
        "--candidate-file",
        type=Path,
        help="Path to a file containing the replacement block (between BEGIN/END markers).",
    )
    parser.add_argument(
        "--use-default",
        action="store_true",
        help="Evaluate the baseline heuristic (ignore any candidate input).",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Label used to tag the evaluation artifacts (e.g., candidate identifier).",
    )
    parser.add_argument(
        "--split",
        default="toy20",
        help="Dataset split to benchmark (default: toy20).",
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
        help="Optional per-instance timeout in seconds.",
    )
    parser.add_argument(
        "--cpu-affinity",
        type=str,
        default=None,
        help="Optional CPU affinity passed to taskset (e.g., '0' or '0-3').",
    )
    parser.add_argument(
        "--sandbox-root",
        type=Path,
        default=None,
        help="Optional directory in which temporary sandboxes are created.",
    )
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="Keep the sandbox directory after evaluation for debugging.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional directory under which evaluation outputs are stored.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Optional JSON metadata attached to run_concorde_eval (stored in config.json).",
    )
    return parser.parse_args(argv)


def load_candidate(args) -> str | None:
    if args.use_default:
        return None
    if args.candidate_file:
        return args.candidate_file.read_text()
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    raise SystemExit("No candidate provided. Use --candidate-file, pipe code via stdin, or pass --use-default.")


def main(argv=None) -> int:
    args = parse_args(argv)
    code = load_candidate(args)

    extra_env = None
    if args.environment:
        try:
            extra_env = json.loads(args.environment)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --environment: {exc}") from exc

    result = evaluate_candidate(
        code=code,
        label=args.label,
        split=args.split,
        repeats=args.repeats,
        timeout=args.timeout,
        cpu_affinity=args.cpu_affinity,
        sandbox_root=args.sandbox_root,
        keep_sandbox=args.keep_sandbox,
        environment=extra_env,
        run_root=args.run_root,
    )

    print(json.dumps(result, indent=2))

    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
