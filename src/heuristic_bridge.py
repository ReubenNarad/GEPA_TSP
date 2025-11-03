"""
Sandboxed evaluation utilities for GEPA heuristic candidates.

Key entry point:
    evaluate_candidate(code: str, label: str, ...)

This routine:
  * creates an isolated copy of the Concorde tree,
  * injects the provided code into the sentinel block of linkern.c,
  * rebuilds Concorde within the sandbox,
  * runs scripts/run_concorde_eval.py on the requested dataset split,
  * stores artifacts in runs/eval/... and returns a structured summary.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


BEGIN_MARKER = "/* BEGIN LLM HEURISTIC BLOCK"
END_MARKER = "/* END LLM HEURISTIC BLOCK */"


class HeuristicEvaluationError(RuntimeError):
    """Wraps errors encountered during sandbox evaluation."""


def repo_root() -> Path:
    """Return the project root (directory containing this file's parent)."""
    return Path(__file__).resolve().parents[1]


def default_block() -> str:
    """Return the baseline heuristic block contents."""
    path = repo_root() / "concorde" / "concorde" / "LINKERN" / "linkern_llm_default.c"
    return path.read_text()


def inject_block(linkern_path: Path, new_block: str) -> None:
    """Replace the sentinel block in linkern.c with new_block."""
    content = linkern_path.read_text()
    try:
        begin_idx = content.index(BEGIN_MARKER)
        start = content.index("\n", begin_idx) + 1
        end_idx = content.index(END_MARKER, start)
    except ValueError as exc:
        raise HeuristicEvaluationError(f"Sentinel markers not found in {linkern_path}") from exc

    # Preserve indentation: assume block is already indented with four spaces.
    replacement = new_block.rstrip() + "\n"
    updated = content[:start] + replacement + content[end_idx:]
    linkern_path.write_text(updated)


def copy_concorde_tree(destination: Path) -> None:
    """Copy the concorde directory into destination/concorde."""
    root = repo_root()
    src = root / "concorde"
    dest = destination / "concorde"
    if dest.exists():
        raise HeuristicEvaluationError(f"Destination already exists: {dest}")

    shutil.copytree(src, dest, symlinks=True)


def evaluate_candidate(
    code: Optional[str],
    label: str,
    split: str = "toy20",
    repeats: int = 1,
    timeout: Optional[float] = None,
    sandbox_root: Optional[Path] = None,
    keep_sandbox: bool = False,
    environment: Optional[Dict[str, Any]] = None,
    run_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate a candidate heuristic block.

    Parameters
    ----------
    code:
        String containing the code to inject between sentinel markers.
        If None, the default block is used.
    label:
        Identifier used for the evaluation run directory label.
    split:
        Dataset split to benchmark (matches entries in metadata.json).
    repeats:
        Number of repetitions per instance.
    timeout:
        Optional per-instance timeout (seconds) passed to Concorde.
    sandbox_root:
        Optional directory to host sandbox copies. Defaults to system temp.
    keep_sandbox:
        If True, leave the sandbox directory on disk for inspection.
    environment:
        Extra metadata stored in the evaluation run's config.json.
    run_root:
        Optional directory under which evaluation artifacts should be stored.
        When provided, artifacts will land in ``run_root / "eval" / ...``.

    Returns
    -------
    dict
        Contains build status, evaluation summary, run directory, and artifacts.
    """

    root = repo_root()
    code_block = code if code is not None else default_block()

    if not label:
        raise ValueError("label must be a non-empty string")

    sandbox_parent = Path(sandbox_root) if sandbox_root else None
    sandbox_dir = Path(
        tempfile.mkdtemp(prefix="gepa_sandbox_", dir=None if sandbox_parent is None else sandbox_parent)
    )
    build_log: Optional[str] = None
    build_rc: Optional[int] = None
    eval_stdout: Optional[str] = None
    eval_stderr: Optional[str] = None
    eval_rc: Optional[int] = None
    run_dir_path: Optional[Path] = None

    try:
        copy_concorde_tree(sandbox_dir)

        linkern_path = sandbox_dir / "concorde" / "concorde" / "LINKERN" / "linkern.c"
        inject_block(linkern_path, code_block)

        # Rebuild Concorde inside sandbox.
        make_cmd = ["make", f"-j{os.cpu_count() or 4}"]
        build_proc = subprocess.run(
            make_cmd,
            cwd=sandbox_dir / "concorde" / "concorde",
            capture_output=True,
            text=True,
            check=False,
        )
        build_log = build_proc.stdout + ("\n" + build_proc.stderr if build_proc.stderr else "")
        build_rc = build_proc.returncode
        if build_rc != 0:
            raise HeuristicEvaluationError("Concorde rebuild failed")

        # Prepare evaluation command.
        eval_root = Path(run_root) if run_root is not None else root / "runs"
        eval_root.mkdir(parents=True, exist_ok=True)

        eval_cmd = [
            sys.executable,
            str(root / "scripts" / "run_concorde_eval.py"),
            "--binary",
            str(sandbox_dir / "concorde" / "install" / "bin" / "concorde"),
            "--metadata",
            str(root / "data" / "eval" / "metadata.json"),
            "--split",
            split,
            "--repeats",
            str(repeats),
            "--label",
            label,
            "--run-dir",
            str(eval_root),
        ]
        if timeout is not None:
            eval_cmd.extend(["--timeout", str(timeout)])
        if environment is not None:
            eval_cmd.extend(["--environment", json.dumps(environment)])

        eval_proc = subprocess.run(
            eval_cmd,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        eval_stdout = eval_proc.stdout
        eval_stderr = eval_proc.stderr
        eval_rc = eval_proc.returncode
        if eval_rc != 0:
            raise HeuristicEvaluationError("Evaluation script failed")

        match = re.search(r"Artifacts saved to:\s*(\S+)", eval_stdout)
        if match:
            run_dir_path = Path(match.group(1)).resolve()
        else:
            raise HeuristicEvaluationError("Unable to locate run directory from evaluator output.")

        summary_path = run_dir_path / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        # Persist candidate code and build log alongside artifacts.
        (run_dir_path / "candidate_linkern_block.c").write_text(code_block.rstrip() + "\n")
        (run_dir_path / "build.log").write_text(build_log or "")

        return {
            "status": "ok",
            "run_dir": str(run_dir_path),
            "summary": summary,
            "build_log": build_log,
            "evaluation_stdout": eval_stdout,
            "evaluation_stderr": eval_stderr,
            "sandbox_dir": str(sandbox_dir) if keep_sandbox else None,
            "label": label,
            "split": split,
            "repeats": repeats,
            "timeout": timeout,
        }

    except HeuristicEvaluationError as exc:
        return {
            "status": "error",
            "error": str(exc),
            "build_log": build_log,
            "build_returncode": build_rc,
            "evaluation_returncode": eval_rc,
            "evaluation_stdout": eval_stdout,
            "evaluation_stderr": eval_stderr,
            "sandbox_dir": str(sandbox_dir),
        }
    finally:
        if not keep_sandbox and sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)
