#!/usr/bin/env python3
"""Run GEPA to evolve Concorde's Lin-Kernighan heuristic."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Repository layout ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Load secrets from .env if python-dotenv is available (so OPENROUTER_API_KEY is picked up).
env_path = ROOT / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=env_path)
    except ImportError:
        print("warning: python-dotenv not installed; ignoring .env file", file=sys.stderr)

# Quiet down DSPy/GEPA info logs (prompts, metrics spam) while keeping progress bars.
for _logger_name in ("dspy", "gepa", "dspy.teleprompt.gepa", "dspy.evaluate"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

# Keep DSPy caches inside the repo so we can run without HOME write access.
CACHE_DIR = ROOT / ".dspy_cache"
os.environ.setdefault("DSPY_CACHEDIR", str(CACHE_DIR))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

import dspy
from dspy import Example
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from dspy.utils.callback import BaseCallback

from gepa_metric import evaluate_and_score  # type: ignore
from heuristic_bridge import default_block  # type: ignore

PROMPT_FILE = ROOT / "prompt_scratchpad.md"


def load_prompt(section: str) -> str:
    text = PROMPT_FILE.read_text()
    pattern = rf"## {re.escape(section)}.*?```(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find prompt section '{section}' in {PROMPT_FILE}")
    return match.group(1).strip()


def config_to_json(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = vars(args).copy()
    for key, value in list(cfg.items()):
        if isinstance(value, Path):
            cfg[key] = str(value)
    return cfg


def collect_lm_usage(lms: list[dspy.LM]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, float]] = {}

    def record(model_name: str, field: str, value: float) -> None:
        entry = summary.setdefault(
            model_name,
            {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0, "calls": 0},
        )
        entry[field] += float(value)

    for lm in lms:
        for entry in getattr(lm, "history", []) or []:
            model_name = entry.get("response_model") or entry.get("model") or lm.model
            usage = entry.get("usage") or {}
            input_tokens = (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or usage.get("promptTokens")
                or 0
            )
            output_tokens = (
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or usage.get("completionTokens")
                or 0
            )
            total_tokens = (
                usage.get("total_tokens")
                or usage.get("totalTokens")
                or (input_tokens + output_tokens)
            )

            record(model_name, "input_tokens", input_tokens)
            record(model_name, "output_tokens", output_tokens)
            record(model_name, "total_tokens", total_tokens)
            summary[model_name]["calls"] += 1

    aggregated: Dict[str, Dict[str, int]] = {}
    for model_name, payload in summary.items():
        aggregated[model_name] = {
            "input_tokens": int(round(payload["input_tokens"])),
            "output_tokens": int(round(payload["output_tokens"])),
            "total_tokens": int(round(payload["total_tokens"])),
            "calls": int(payload["calls"]),
        }
    return aggregated


class LKSignature(dspy.Signature):
    """Rewrite the Lin-Kernighan heuristic block."""

    current_block = dspy.InputField(
        prefix="Current block:",
        desc="The existing C code between BEGIN/END markers.",
    )
    heuristic_block = dspy.OutputField(
        prefix="Replacement block:",
        desc="Rewritten C code to paste between the markers.",
        format=str,
    )


class LinKernighanProgram(dspy.Module):
    """DSPy module that emits a replacement heuristic block via dspy.Predict."""

    def __init__(self, student_instructions: str, max_tokens: int) -> None:
        super().__init__()
        signature = LKSignature.with_instructions(student_instructions)
        self.rewriter = dspy.Predict(signature=signature, max_tokens=max_tokens)

    def forward(self, current_block: str) -> dspy.Prediction:
        return self.rewriter(current_block=current_block)


def build_metric(
    label_prefix: str,
    default_split: str,
    default_repeats: int,
    default_timeout: Optional[float],
    environment: Optional[Dict[str, Any]],
    run_root: Optional[Path],
) -> callable:
    """Create a GEPA metric that wraps evaluate_and_score."""

    def metric(
        example: Example,
        prediction: dspy.Prediction,
        trace=None,
        pred_name: Optional[str] = None,
        pred_trace=None,
    ) -> ScoreWithFeedback:
        # Reuse cached result if available (needed because GEPA expects
        # feedback score to match the module-level score precisely).
        cached_score = getattr(prediction, "_gepa_score", None)
        cached_feedback = getattr(prediction, "_gepa_feedback_text", None)
        cached_result = getattr(prediction, "_gepa_result", None)
        block = getattr(prediction, "heuristic_block", None)
        if cached_score is not None and cached_feedback is not None and cached_result is not None:
            swf = ScoreWithFeedback(score=cached_score, feedback=cached_feedback)
            swf.result = cached_result  # type: ignore[attr-defined]
            return swf

        if not isinstance(block, str) or not block.strip():
            feedback = "Candidate did not return a heuristic_block string."
            return ScoreWithFeedback(score=-1e9, feedback=feedback)

        label = f"{label_prefix}_{uuid.uuid4().hex[:8]}"
        split = getattr(example, "split", default_split)
        repeats = getattr(example, "repeats", default_repeats)
        timeout = getattr(example, "timeout", default_timeout)
        env_payload = getattr(example, "environment", environment)

        score, feedback, result = evaluate_and_score(
            code=block,
            label=label,
            split=split,
            repeats=repeats,
            timeout=timeout,
            environment=env_payload,
            run_root=run_root,
        )

        # Attach artifacts to the feedback for the reflector to inspect.
        run_dir = result.get("run_dir")
        if run_dir:
            feedback = f"{feedback}\nArtifacts: {run_dir}"

        swf = ScoreWithFeedback(score=score, feedback=feedback)
        swf.result = result  # type: ignore[attr-defined]

        # Cache on prediction for future reflective calls (must match module score exactly).
        setattr(prediction, "_gepa_score", score)
        setattr(prediction, "_gepa_feedback_text", feedback)
        setattr(prediction, "_gepa_result", result)

        return swf

    return metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA on the Lin-Kernighan heuristic.")
    parser.add_argument("--student-model", default="openai/gpt-5-nano", help="Student model (OpenRouter ID).")
    parser.add_argument("--reflector-model", default="openai/gpt-5-mini", help="Reflector model (OpenRouter ID).")
    parser.add_argument(
        "--student-max-tokens",
        type=int,
        default=6000,
        help="Maximum decoding tokens for the student LM (<=0 lets GEPA choose).",
    )
    parser.add_argument(
        "--reflector-max-tokens",
        type=int,
        default=4000,
        help="Maximum decoding tokens for the reflector LM (<=0 lets GEPA choose).",
    )
    parser.add_argument("--split", default="toy20", help="Dataset split for evaluation.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per instance during evaluation.")
    parser.add_argument(
        "--baseline-repeats",
        type=int,
        default=5,
        help="Repeats for the baseline evaluation (averaged).",
    )
    parser.add_argument("--timeout", type=float, default=None, help="Optional per-instance timeout (seconds).")
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Budget for GEPA metric calls. Use 0 to only evaluate the baseline block.",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=3,
        help="Number of train examples to feed GEPA (all using the provided split/repeats/timeout).",
    )
    parser.add_argument(
        "--reflection-batch",
        type=int,
        default=2,
        help="Reflection minibatch size (see GEPA docs).",
    )
    parser.add_argument("--label-prefix", default="gepa", help="Prefix for metric run labels.")
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Optional JSON string merged into metric environment metadata.",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("runs/gepa"), help="Directory for GEPA summaries.")
    return parser.parse_args()


def configure_lms(
    args: argparse.Namespace,
    headers: Dict[str, str],
) -> tuple[dspy.LM, dspy.LM]:
    api_key = os.environ["OPENROUTER_API_KEY"]
    api_base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

    # gpt-5 reasoning models require temperature=1.0 and >=16000 tokens.
    def lm_kwargs(model: str, requested: int) -> Dict[str, Any]:
        is_reasoning = model.startswith("openai/gpt-5") or "/gpt-5" in model
        if requested and requested > 0:
            max_tokens = requested
        else:
            max_tokens = 20000 if is_reasoning else 6000
        if is_reasoning:
            max_tokens = max(max_tokens, 16000)
        return {
            "temperature": 1.0 if is_reasoning else 0.3,
            "max_tokens": max_tokens,
            "api_key": api_key,
            "api_base": api_base,
            "extra_headers": headers or None,
        }

    student_lm = dspy.LM(
        args.student_model,
        **lm_kwargs(args.student_model, args.student_max_tokens),
    )
    reflector_lm = dspy.LM(
        args.reflector_model,
        **lm_kwargs(args.reflector_model, args.reflector_max_tokens),
    )
    return student_lm, reflector_lm


def baseline_evaluation(
    args: argparse.Namespace,
    environment: Optional[Dict[str, Any]],
    run_root: Path,
) -> Dict[str, Any]:
    label = f"{args.label_prefix}_baseline"
    baseline_repeats = max(1, args.baseline_repeats)
    score, feedback, result = evaluate_and_score(
        code=default_block(),
        label=label,
        split=args.split,
        repeats=baseline_repeats,
        timeout=args.timeout,
        environment=environment,
        run_root=run_root,
    )

    run_root.mkdir(parents=True, exist_ok=True)
    baseline_path = run_root / "baseline.json"
    payload = {
        "config": config_to_json(args),
        "score": score,
        "feedback": feedback,
        "result": result,
        "baseline_repeats": baseline_repeats,
    }
    baseline_path.write_text(json.dumps(payload, indent=2))
    print(f"Baseline evaluation saved to {baseline_path}")
    return payload


def main() -> None:
    args = parse_args()

    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY must be set (consider storing it in .env).")

    environment = None
    if args.environment:
        try:
            environment = json.loads(args.environment)
        except json.JSONDecodeError as err:
            raise SystemExit(f"Failed to parse --environment JSON: {err}") from err

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = args.save_dir / f"{timestamp}_{args.label_prefix}"

    if args.steps <= 0:
        baseline_evaluation(args, environment, run_root)
        return

    baseline_payload = baseline_evaluation(args, environment, run_root)

    student_prompt = load_prompt("Student Prompt")
    reflector_prompt = load_prompt("Reflector Prompt")

    headers: Dict[str, str] = {}
    if os.environ.get("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
    if os.environ.get("OPENROUTER_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_TITLE"]

    student_lm, reflector_lm = configure_lms(args, headers)
    dspy.configure(lm=student_lm)

    student_block_tokens = args.student_max_tokens if args.student_max_tokens > 0 else 6000
    program = LinKernighanProgram(student_prompt, max_tokens=student_block_tokens)

    def make_example() -> Example:
        return Example(
            current_block=default_block(),
            split=args.split,
            repeats=args.repeats,
            timeout=args.timeout,
            environment=environment,
        ).with_inputs("current_block")

    trainset = [make_example() for _ in range(max(1, args.train_examples))]

    metric_fn = build_metric(
        label_prefix=args.label_prefix,
        default_split=args.split,
        default_repeats=args.repeats,
        default_timeout=args.timeout,
        environment=environment,
        run_root=run_root,
    )
    # Track hashes of evaluated heuristic blocks to avoid duplicate evaluations.
    evaluated_hashes = set()

    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    gepa = GEPA(
        metric=metric_fn,
        reflection_lm=reflector_lm,
        max_metric_calls=args.steps,
        reflection_minibatch_size=args.reflection_batch,
        candidate_selection_strategy="pareto",
        track_stats=True,
        log_dir=str(log_dir),
        failure_score=-1e9,
        perfect_score=0.0,
        deduplicate=True,
    )

    try:
        def dedup_wrapper(module, *args, **kwargs):
            # Hash heuristic_block string to skip duplicates
            candidate_block = getattr(module, "heuristic_block", None)
            if isinstance(candidate_block, str):
                h = hash(candidate_block)
                if h in evaluated_hashes:
                    # Return dummy prediction with cached score/feedback to avoid re-eval
                    pred = dspy.Prediction()
                    setattr(pred, "_gepa_score", -1e9)
                    setattr(pred, "_gepa_feedback_text", "Duplicate candidate skipped by dedup.")
                    setattr(pred, "_gepa_result", {"status": "skipped_duplicate"})
                    setattr(pred, "heuristic_block", candidate_block)
                    return pred
                evaluated_hashes.add(h)
            return module(*args, **kwargs)

        optimized_program = gepa.compile(program, trainset=trainset)
        # Wrap predictors to enforce dedup (best-effort)
        for name, pred in optimized_program.named_predictors():
            orig_forward = pred.forward
            pred.forward = lambda *a, **kw: dedup_wrapper(pred, *a, **kw)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"GEPA failed: {exc}") from exc

    # Run the optimized program once more to capture the final candidate code.
    final_prediction = optimized_program(current_block=train_example.current_block)
    final_block = getattr(final_prediction, "heuristic_block", "")

    final_score, final_feedback, final_result = evaluate_and_score(
        code=final_block,
        label=f"{args.label_prefix}_final",
        split=args.split,
        repeats=args.repeats,
        timeout=args.timeout,
        environment=environment,
        run_root=run_root,
    )

    final_dir = run_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_run_dir = final_result.get("run_dir")
    if final_run_dir:
        candidate_src = Path(final_run_dir) / "candidate_linkern_block.c"
        if candidate_src.exists():
            shutil.copy2(candidate_src, final_dir / "candidate_linkern_block.c")
        summary_src = Path(final_run_dir) / "summary.json"
        if summary_src.exists():
            shutil.copy2(summary_src, final_dir / "evaluation_summary.json")

    summary = {
        "timestamp": timestamp,
        "config": config_to_json(args),
        "baseline": baseline_payload,
        "final_score": final_score,
        "final_feedback": final_feedback,
        "final_result": final_result,
        "optimized_instructions": optimized_program.rewriter.signature.instructions,
        "gepa_logs": str(log_dir),
    }

    detailed = getattr(optimized_program, "detailed_results", None)
    if detailed is not None:
        try:
            summary["gepa_details"] = detailed.to_dict()
        except AttributeError:
            summary["gepa_details"] = str(detailed)

    lm_usage = collect_lm_usage([student_lm, reflector_lm])
    if lm_usage:
        summary["lm_usage"] = lm_usage
        usage_path = log_dir / "lm_usage.json"
        usage_path.write_text(json.dumps(lm_usage, indent=2))
        print("LM usage (tokens):")
        for model_name, stats in lm_usage.items():
            print(
                f"  {model_name}: input={stats['input_tokens']} output={stats['output_tokens']} "
                f"total={stats['total_tokens']} calls={stats['calls']}"
            )
    else:
        print("LM usage (tokens): none recorded")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_root / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"GEPA summary saved to {out_path}")
    if final_result.get("run_dir"):
        print(f"Final evaluation artifacts: {final_result['run_dir']}")


if __name__ == "__main__":
    main()
