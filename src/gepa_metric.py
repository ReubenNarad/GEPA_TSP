"""
GEPA metric wrapper around the sandboxed Concorde evaluator.

The primary entry point is `evaluate_and_score`, which conforms to DSPy’s
expected metric signature: it returns both a scalar score and a textual
feedback summary that the reflection LLM can process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from heuristic_bridge import evaluate_candidate, HeuristicEvaluationError


def _build_feedback(result: Dict[str, Any]) -> str:
    parts = []
    status = result.get("status")
    parts.append(f"status: {status}")

    summary = result.get("summary") or {}
    if summary:
        parts.append("summary: " + json.dumps(summary, sort_keys=True))

    if result.get("build_log"):
        parts.append("build_log (tail): " + result["build_log"].splitlines()[-1])

    if result.get("evaluation_stderr"):
        parts.append("stderr (tail): " + result["evaluation_stderr"].splitlines()[-1])

    run_dir = result.get("run_dir")
    if run_dir:
        parts.append(f"artifacts: {run_dir}")

    if status != "ok":
        parts.append("full_result: " + json.dumps(result, indent=2))

    return "\n".join(parts)


def evaluate_and_score(
    code: Optional[str],
    label: str,
    split: str = "toy20",
    repeats: int = 1,
    timeout: Optional[float] = None,
    environment: Optional[Dict[str, Any]] = None,
    run_root: Optional[Path] = None,
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Evaluate a candidate and produce a scalar score plus feedback text.

    Returns
    -------
    score : float
        Higher is better (negative wall time by default).
    feedback : str
        Textual diagnostics for GEPA reflection models.
    result : dict
        Full evaluation result dictionary (for logging or caching).
    """

    result = evaluate_candidate(
        code=code,
        label=label,
        split=split,
        repeats=repeats,
        timeout=timeout,
        environment=environment,
        run_root=run_root,
    )

    if result.get("status") != "ok":
        feedback = _build_feedback(result)
        return -1e9, feedback, result

    summary = result.get("summary") or {}
    avg_wall = summary.get("average_wall_time_sec")
    failures = summary.get("failures", 0) or 0
    timeouts = summary.get("timeouts", 0) or 0
    if failures or timeouts:
        feedback = _build_feedback(result) + f"\npenalized: failures={failures}, timeouts={timeouts}"
        return -1e9, feedback, result
    if avg_wall is None:
        feedback = _build_feedback(result)
        return -1e9, feedback, result

    score = -float(avg_wall)
    feedback = _build_feedback(result)
    return score, feedback, result
