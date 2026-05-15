from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .agent_bench_bridge import (
    DEFAULT_BENCHMARK_SPECS,
    parse_benchmark_specs,
    read_benchmark_spec_file,
    run_agent_bench_suite,
)
from .benchmarks import FixtureAdapter, HarveyLabAdapter
from .benchmarks.harvey import default_harvey_root
from .benchmarks.harvey_eval import evaluate_prepared_harvey_run, prepare_harvey_eval_package
from .config import load_config
from .diagnosis import diagnose_harvey_scores_file
from .doctor import run_doctor
from .env import load_dotenv_if_present
from .experiments import close_experiment, open_experiment, read_experiment
from .harvey_pipeline import HarveyPipelineResult, run_harvey_batch
from .product import DEFAULT_PRODUCT_MAX_FILES, DEFAULT_PRODUCT_TOP_K, run_product_matter
from .product_ui import serve_product_ui
from .state import RunState
from .trace import TraceWriter, attach_harvey_scores, diagnose_trace, load_trace, save_trace, trace_summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a benchmark task")
    run.add_argument("--benchmark", default="fixture")
    run.add_argument("--task-id", default="smoke")
    run.add_argument("--config", default=None)
    run.add_argument("--trace-dir", default="traces")
    run.add_argument("--output-dir", default="outputs")
    run.add_argument("--live-synthesis", action="store_true")
    run.set_defaults(func=cmd_run)

    list_tasks = sub.add_parser("list-tasks", help="List benchmark tasks")
    list_tasks.add_argument("--benchmark", default="fixture")
    list_tasks.add_argument("--split", default="sample")
    list_tasks.add_argument("--limit", type=int, default=20)
    list_tasks.set_defaults(func=cmd_list_tasks)

    inspect = sub.add_parser("inspect", help="Summarize one trace")
    inspect.add_argument("--trace", required=True)
    inspect.set_defaults(func=cmd_inspect)

    diagnose = sub.add_parser("diagnose", help="Diagnose one trace")
    diagnose.add_argument("--trace", required=True)
    diagnose.set_defaults(func=cmd_diagnose)

    diagnose_scores = sub.add_parser("diagnose-scores", help="Diagnose a benchmark scores.json file")
    diagnose_scores.add_argument("--scores", required=True)
    diagnose_scores.set_defaults(func=cmd_diagnose_scores)

    attach_scores = sub.add_parser("attach-scores", help="Attach a benchmark scores.json file to a trace")
    attach_scores.add_argument("--trace", required=True)
    attach_scores.add_argument("--scores", required=True)
    attach_scores.set_defaults(func=cmd_attach_scores)

    refresh_harvey_diagnostics = sub.add_parser(
        "refresh-harvey-diagnostics",
        help="Refresh Harvey trace diagnosis fields from saved LAB scores.json files",
    )
    refresh_harvey_diagnostics.add_argument("--trace-dir", required=True)
    refresh_harvey_diagnostics.add_argument("--harvey-root", default=None)
    refresh_harvey_diagnostics.set_defaults(func=cmd_refresh_harvey_diagnostics)

    summarize = sub.add_parser("summarize", help="Summarize all traces under a run directory")
    summarize.add_argument("--run", required=True)
    summarize.set_defaults(func=cmd_summarize)

    compare = sub.add_parser("compare", help="Compare two trace directories")
    compare.add_argument("--run-a", required=True)
    compare.add_argument("--run-b", required=True)
    compare.set_defaults(func=cmd_compare)

    doctor = sub.add_parser("doctor", help="Check local harness environment")
    doctor.set_defaults(func=cmd_doctor)

    product_run = sub.add_parser("product-run", help="Run the product matter flow over user-provided documents")
    product_run.add_argument("--objective", required=True)
    product_run.add_argument("--path", action="append", default=[])
    product_run.add_argument("--matter-id", default="local-matter")
    product_run.add_argument("--chat-id", default="main")
    product_run.add_argument("--trace-dir", default="traces/product")
    product_run.add_argument("--output-dir", default="outputs/product")
    product_run.add_argument("--config", default=None)
    product_run.add_argument("--top-k", type=int, default=DEFAULT_PRODUCT_TOP_K)
    product_run.add_argument("--max-files", type=int, default=DEFAULT_PRODUCT_MAX_FILES)
    product_run.add_argument("--worker-source-planning", action="store_true")
    product_run.add_argument("--live-synthesis", action="store_true")
    product_run.set_defaults(func=cmd_product_run)

    product_ui = sub.add_parser("product-ui", help="Serve the local product matter UI")
    product_ui.add_argument("--host", default="127.0.0.1")
    product_ui.add_argument("--port", type=int, default=8765)
    product_ui.add_argument("--trace-dir", default="traces/product")
    product_ui.add_argument("--output-dir", default="outputs/product")
    product_ui.add_argument("--config", default=None)
    product_ui.set_defaults(func=cmd_product_ui)

    harvey_eval = sub.add_parser("prepare-harvey-eval", help="Package a Harvey trace for Harvey LAB evaluator")
    harvey_eval.add_argument("--trace", required=True)
    harvey_eval.add_argument("--harvey-root", default=None)
    harvey_eval.add_argument("--run-id", default=None)
    harvey_eval.set_defaults(func=cmd_prepare_harvey_eval)

    harvey_score = sub.add_parser("score-harvey", help="Score a prepared Harvey LAB run")
    harvey_score.add_argument("--run-id", required=True)
    harvey_score.add_argument("--task-id", required=True)
    harvey_score.add_argument("--harvey-root", default=None)
    harvey_score.add_argument("--judge-model", default=None)
    harvey_score.add_argument("--parallel", type=int, default=24)
    harvey_score.add_argument("--execute", action="store_true")
    harvey_score.set_defaults(func=cmd_score_harvey)

    harvey_smoke = sub.add_parser("harvey-smoke", help="Run multiple Harvey tasks with optional concurrent scoring")
    harvey_smoke.add_argument("--task-id", action="append", default=[])
    harvey_smoke.add_argument("--task-file", default=None)
    harvey_smoke.add_argument("--split", default=None)
    harvey_smoke.add_argument("--limit", type=int, default=None)
    harvey_smoke.add_argument("--trace-dir", default="traces")
    harvey_smoke.add_argument("--output-dir", default="outputs")
    harvey_smoke.add_argument("--harvey-root", default=None)
    harvey_smoke.add_argument("--run-prefix", default="irys-smoke")
    harvey_smoke.add_argument("--config", default=None)
    harvey_smoke.add_argument("--workers", type=int, default=24)
    harvey_smoke.add_argument("--score-parallel", type=int, default=24)
    harvey_smoke.add_argument("--resume", action="store_true")
    harvey_smoke.add_argument("--summary-path", default=None)
    harvey_smoke.add_argument("--execute-score", action="store_true")
    harvey_smoke.add_argument("--no-live-synthesis", action="store_true")
    harvey_smoke.set_defaults(func=cmd_harvey_smoke)

    agent_bench = sub.add_parser(
        "agent-bench-smoke",
        help="Run Irys through the sibling agent-bench benchmark harness",
    )
    agent_bench.add_argument("--agent-bench-root", default=None)
    agent_bench.add_argument("--data-dir", default=None)
    agent_bench.add_argument("--benchmark", action="append", default=[])
    agent_bench.add_argument("--benchmark-file", default=None)
    agent_bench.add_argument("--limit", type=int, default=10)
    agent_bench.add_argument("--concurrency", type=int, default=18)
    agent_bench.add_argument("--benchmark-workers", type=int, default=4)
    agent_bench.add_argument("--checkpoint-every", type=int, default=5)
    agent_bench.add_argument("--results-dir", default="scratch/agent_bench_irys")
    agent_bench.add_argument("--trace-dir", default=None)
    agent_bench.add_argument("--config", default=None)
    agent_bench.add_argument("--backend-mode", choices=["adaptive", "direct", "three-tier"], default="three-tier")
    agent_bench.add_argument("--resume", action="store_true")
    agent_bench.add_argument("--dry-run", action="store_true")
    agent_bench.set_defaults(func=cmd_agent_bench_smoke)

    experiment = sub.add_parser("experiment", help="Open or close improvement experiments")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_open = experiment_sub.add_parser("open", help="Open an improvement experiment")
    experiment_open.add_argument("--baseline", required=True)
    experiment_open.add_argument("--hypothesis", required=True)
    experiment_open.add_argument(
        "--target",
        choices=["quality", "cost", "latency", "routing", "generalization"],
        required=True,
    )
    experiment_open.add_argument("--experiments-dir", default="experiments")
    experiment_open.set_defaults(func=cmd_experiment_open)

    experiment_close = experiment_sub.add_parser("close", help="Close an improvement experiment")
    experiment_close.add_argument("--experiment", required=True)
    experiment_close.add_argument("--run", required=True)
    decision = experiment_close.add_mutually_exclusive_group(required=True)
    decision.add_argument("--accept", action="store_true")
    decision.add_argument("--reject", action="store_true")
    experiment_close.add_argument("--decision-reason", required=True)
    experiment_close.set_defaults(func=cmd_experiment_close)

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    config = load_config(args.config)
    adapter = get_adapter(args.benchmark)
    if hasattr(adapter, "live_synthesis"):
        adapter.live_synthesis = bool(args.live_synthesis)
    task = adapter.load_task(args.task_id)
    output_dir = Path(args.output_dir) / task.benchmark / task.task_id
    state = RunState(task=task, config=config, output_dir=str(output_dir))
    state = adapter.run(state)
    path = TraceWriter(args.trace_dir).write(state)
    print(f"[SAVE] trace={path}")
    if state.scoring_result and state.scoring_result.passed is False:
        return 1
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    print_json(trace_summary(load_trace(args.trace)))
    return 0


def cmd_list_tasks(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.benchmark)
    if not hasattr(adapter, "list_tasks"):
        print_json({"benchmark": args.benchmark, "tasks": []})
        return 0
    refs = adapter.list_tasks(args.split)  # type: ignore[attr-defined]
    print_json(
        {
            "benchmark": args.benchmark,
            "split": args.split,
            "count": len(refs),
            "tasks": [
                {
                    "task_id": ref.task_id,
                    "practice_area": ref.practice_area,
                    "slug": ref.slug,
                }
                for ref in refs[: args.limit]
            ],
        }
    )
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    print_json(diagnose_trace(load_trace(args.trace)))
    return 0


def cmd_diagnose_scores(args: argparse.Namespace) -> int:
    print_json(diagnose_harvey_scores_file(args.scores))
    return 0


def cmd_attach_scores(args: argparse.Namespace) -> int:
    trace = load_trace(args.trace)
    with Path(args.scores).open("r", encoding="utf-8") as handle:
        scores = json.load(handle)
    updated = attach_harvey_scores(trace, scores)
    save_trace(args.trace, updated)
    print_json(trace_summary(updated))
    return 0


def cmd_refresh_harvey_diagnostics(args: argparse.Namespace) -> int:
    print_json(
        refresh_harvey_diagnostics(
            trace_dir=args.trace_dir,
            harvey_root=args.harvey_root or default_harvey_root(),
        )
    )
    return 0


def refresh_harvey_diagnostics(*, trace_dir: str | Path, harvey_root: str | Path) -> dict[str, Any]:
    root = Path(trace_dir)
    harvey_results = Path(harvey_root) / "results"
    updated: list[str] = []
    skipped: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for trace_path in sorted(root.rglob("*.json")) if root.exists() else []:
        if trace_path.name.startswith("summary"):
            continue
        try:
            trace = load_trace(trace_path)
            if trace.get("benchmark") != "harvey_lab_sample":
                skipped.append({"trace": str(trace_path), "reason": "not_harvey_trace"})
                continue
            run_id = (
                (trace.get("scoring_result") or {})
                .get("details", {})
                .get("run_id")
            )
            if not run_id:
                skipped.append({"trace": str(trace_path), "reason": "missing_run_id"})
                continue
            scores_path = harvey_results / Path(str(run_id)) / "scores.json"
            if not scores_path.exists():
                skipped.append({"trace": str(trace_path), "reason": "missing_scores"})
                continue
            with scores_path.open("r", encoding="utf-8") as handle:
                scores = json.load(handle)
            save_trace(trace_path, attach_harvey_scores(trace, scores))
            updated.append(str(trace_path))
        except Exception as exc:  # noqa: BLE001 - report per-trace diagnosis refresh failures.
            errors.append({"trace": str(trace_path), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "trace_dir": str(root),
        "harvey_root": str(harvey_root),
        "updated": len(updated),
        "skipped": len(skipped),
        "errors": len(errors),
        "updated_traces": updated,
        "skipped_traces": skipped[:50],
        "error_traces": errors,
    }


def median_number(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value or 0.0) for value in values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def max_cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    item = max(rows, key=lambda row: float(row.get("estimated_cost") or 0.0))
    return {
        "benchmark": item.get("benchmark"),
        "task_id": item.get("task_id"),
        "estimated_cost": float(item.get("estimated_cost") or 0.0),
        "total_tokens": int(item.get("total_tokens") or 0),
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    traces = load_trace_dir(args.run)
    summaries = [trace_summary(trace) for trace in traces]
    total = len(summaries)
    passed = sum(1 for item in summaries if item.get("passed") is True)
    total_tokens = sum(int(item.get("total_tokens") or 0) for item in summaries)
    total_cost = sum(float(item.get("estimated_cost") or 0.0) for item in summaries)
    task_costs = [float(item.get("estimated_cost") or 0.0) for item in summaries]
    tokens_by_tier = aggregate_tokens_by_tier(traces)
    print_json(
        {
            "tasks": total,
            "passed": passed,
            "score_rate": passed / total if total else 0.0,
            "total_tokens": total_tokens,
            "estimated_cost": total_cost,
            "average_cost_per_task": total_cost / total if total else 0.0,
            "median_cost_per_task": median_number(task_costs),
            "max_cost_task": max_cost_summary(summaries),
            "tokens_by_tier": tokens_by_tier,
            "token_share_by_tier": token_share(tokens_by_tier),
            "traces": summaries,
        }
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    print_json(compare_run_dirs(args.run_a, args.run_b))
    return 0


def compare_run_dirs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    a = {trace_key(trace): trace_summary(trace) for trace in load_trace_dir(run_a)}
    b = {trace_key(trace): trace_summary(trace) for trace in load_trace_dir(run_b)}
    keys = sorted(set(a) | set(b))
    rows = []
    for key in keys:
        before = a.get(key)
        after = b.get(key)
        before_rate = before.get("rubric_pass_rate") if before else None
        after_rate = after.get("rubric_pass_rate") if after else None
        before_rubric_passed = before.get("rubric_passed") if before else None
        after_rubric_passed = after.get("rubric_passed") if after else None
        rows.append(
            {
                "task": key,
                "before_passed": before.get("passed") if before else None,
                "after_passed": after.get("passed") if after else None,
                "before_rubric_pass_rate": before_rate,
                "after_rubric_pass_rate": after_rate,
                "rubric_pass_rate_delta": (
                    float(after_rate) - float(before_rate)
                    if before_rate is not None and after_rate is not None
                    else None
                ),
                "before_rubric_passed": before_rubric_passed,
                "after_rubric_passed": after_rubric_passed,
                "rubric_passed_delta": (
                    int(after_rubric_passed) - int(before_rubric_passed)
                    if before_rubric_passed is not None and after_rubric_passed is not None
                    else None
                ),
                "before_rubric_total": before.get("rubric_total") if before else None,
                "after_rubric_total": after.get("rubric_total") if after else None,
                "before_cost": before.get("estimated_cost") if before else None,
                "after_cost": after.get("estimated_cost") if after else None,
                "before_tokens": before.get("total_tokens") if before else None,
                "after_tokens": after.get("total_tokens") if after else None,
            }
        )
    before_passed = sum(1 for item in a.values() if item.get("passed") is True)
    after_passed = sum(1 for item in b.values() if item.get("passed") is True)
    before_cost = sum(float(item.get("estimated_cost") or 0.0) for item in a.values())
    after_cost = sum(float(item.get("estimated_cost") or 0.0) for item in b.values())
    before_tokens = sum(int(item.get("total_tokens") or 0) for item in a.values())
    after_tokens = sum(int(item.get("total_tokens") or 0) for item in b.values())
    common = [key for key in keys if key in a and key in b]
    common_scored = [
        key
        for key in common
        if a[key].get("rubric_pass_rate") is not None and b[key].get("rubric_pass_rate") is not None
    ]
    before_common_macro = (
        sum(float(a[key]["rubric_pass_rate"]) for key in common_scored) / len(common_scored)
        if common_scored
        else None
    )
    after_common_macro = (
        sum(float(b[key]["rubric_pass_rate"]) for key in common_scored) / len(common_scored)
        if common_scored
        else None
    )
    before_common_rubric_passed = sum(int(a[key].get("rubric_passed") or 0) for key in common_scored)
    before_common_rubric_total = sum(int(a[key].get("rubric_total") or 0) for key in common_scored)
    after_common_rubric_passed = sum(int(b[key].get("rubric_passed") or 0) for key in common_scored)
    after_common_rubric_total = sum(int(b[key].get("rubric_total") or 0) for key in common_scored)
    before_common_passed = sum(1 for key in common if a[key].get("passed") is True)
    after_common_passed = sum(1 for key in common if b[key].get("passed") is True)
    before_common_cost = sum(float(a[key].get("estimated_cost") or 0.0) for key in common)
    after_common_cost = sum(float(b[key].get("estimated_cost") or 0.0) for key in common)
    before_common_tokens = sum(int(a[key].get("total_tokens") or 0) for key in common)
    after_common_tokens = sum(int(b[key].get("total_tokens") or 0) for key in common)
    scored_rows = [row for row in rows if row["rubric_pass_rate_delta"] is not None]
    family_deltas = compare_family_deltas(scored_rows)
    top_gains = sorted(
        scored_rows,
        key=lambda item: (float(item["rubric_pass_rate_delta"]), int(item["rubric_passed_delta"] or 0)),
        reverse=True,
    )[:20]
    top_regressions = sorted(
        scored_rows,
        key=lambda item: (float(item["rubric_pass_rate_delta"]), int(item["rubric_passed_delta"] or 0)),
    )[:20]
    return {
        "summary": {
            "before_tasks": len(a),
            "after_tasks": len(b),
            "common_tasks": len(common),
            "common_scored_tasks": len(common_scored),
            "passed_delta": after_passed - before_passed,
            "before_common_passed": before_common_passed,
            "after_common_passed": after_common_passed,
            "common_passed_delta": after_common_passed - before_common_passed,
            "before_common_macro_rubric_pass_rate": before_common_macro,
            "after_common_macro_rubric_pass_rate": after_common_macro,
            "common_macro_rubric_pass_rate_delta": (
                after_common_macro - before_common_macro
                if before_common_macro is not None and after_common_macro is not None
                else None
            ),
            "before_common_rubric_passed": before_common_rubric_passed,
            "before_common_rubric_total": before_common_rubric_total,
            "after_common_rubric_passed": after_common_rubric_passed,
            "after_common_rubric_total": after_common_rubric_total,
            "common_rubric_passed_delta": after_common_rubric_passed - before_common_rubric_passed,
            "cost_delta": after_cost - before_cost,
            "before_common_cost": before_common_cost,
            "after_common_cost": after_common_cost,
            "common_cost_delta": after_common_cost - before_common_cost,
            "before_common_average_cost_per_task": (
                before_common_cost / len(common) if common else 0.0
            ),
            "after_common_average_cost_per_task": (
                after_common_cost / len(common) if common else 0.0
            ),
            "common_average_cost_per_task_delta": (
                (after_common_cost - before_common_cost) / len(common) if common else 0.0
            ),
            "token_delta": after_tokens - before_tokens,
            "before_common_tokens": before_common_tokens,
            "after_common_tokens": after_common_tokens,
            "common_token_delta": after_common_tokens - before_common_tokens,
        },
        "top_gains": top_gains,
        "top_regressions": top_regressions,
        "family_deltas": family_deltas,
        "tasks": rows,
    }


def compare_family_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "tasks": 0,
            "before_rubric_passed": 0,
            "before_rubric_total": 0,
            "after_rubric_passed": 0,
            "after_rubric_total": 0,
            "rubric_passed_delta": 0,
            "macro_rubric_pass_rate_delta": 0.0,
            "top_gain": None,
            "top_regression": None,
        }
    )
    for row in rows:
        family = task_family_from_key(str(row.get("task") or ""))
        bucket = families[family]
        bucket["tasks"] += 1
        bucket["before_rubric_passed"] += int(row.get("before_rubric_passed") or 0)
        bucket["before_rubric_total"] += int(row.get("before_rubric_total") or 0)
        bucket["after_rubric_passed"] += int(row.get("after_rubric_passed") or 0)
        bucket["after_rubric_total"] += int(row.get("after_rubric_total") or 0)
        bucket["rubric_passed_delta"] += int(row.get("rubric_passed_delta") or 0)
        bucket["macro_rubric_pass_rate_delta"] += float(row.get("rubric_pass_rate_delta") or 0.0)
        if row.get("rubric_passed_delta") is not None:
            gain = bucket["top_gain"]
            regression = bucket["top_regression"]
            if gain is None or int(row["rubric_passed_delta"]) > int(gain["rubric_passed_delta"]):
                bucket["top_gain"] = row
            if regression is None or int(row["rubric_passed_delta"]) < int(regression["rubric_passed_delta"]):
                bucket["top_regression"] = row
    results = []
    for family, bucket in families.items():
        tasks = int(bucket["tasks"])
        before_total = int(bucket["before_rubric_total"])
        after_total = int(bucket["after_rubric_total"])
        results.append(
            {
                "family": family,
                "tasks": tasks,
                "before_rubric_passed": bucket["before_rubric_passed"],
                "before_rubric_total": before_total,
                "after_rubric_passed": bucket["after_rubric_passed"],
                "after_rubric_total": after_total,
                "before_micro_rubric_pass_rate": (
                    bucket["before_rubric_passed"] / before_total if before_total else None
                ),
                "after_micro_rubric_pass_rate": (
                    bucket["after_rubric_passed"] / after_total if after_total else None
                ),
                "rubric_passed_delta": bucket["rubric_passed_delta"],
                "macro_rubric_pass_rate_delta": (
                    bucket["macro_rubric_pass_rate_delta"] / tasks if tasks else None
                ),
                "top_gain": bucket["top_gain"],
                "top_regression": bucket["top_regression"],
            }
        )
    return sorted(
        results,
        key=lambda item: (int(item["rubric_passed_delta"]), float(item["macro_rubric_pass_rate_delta"] or 0.0)),
        reverse=True,
    )


def task_family_from_key(key: str) -> str:
    task_id = key.split("::", 1)[-1]
    parts = [part for part in task_id.split("/") if part]
    leaf = parts[-1] if parts else ""
    if leaf.startswith("scenario-") and len(parts) >= 2:
        leaf = parts[-2]
    prefixes = [
        ("draft-", "draft/package"),
        ("extract-", "extract"),
        ("identify-", "identify/issues"),
        ("compare-", "compare"),
        ("review-", "review"),
        ("analyze-", "analyze"),
        ("assess-", "assess"),
        ("summarize-", "summarize"),
        ("synthesize-", "synthesize"),
        ("prepare-", "prepare"),
        ("conduct-", "conduct"),
    ]
    for prefix, family in prefixes:
        if leaf.startswith(prefix):
            return family
    return leaf.split("-", 1)[0] if leaf else "unknown"


def cmd_doctor(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    checks = run_doctor()
    print_json({"checks": [check.to_dict() for check in checks]})
    return 0 if all(check.passed for check in checks) else 1


def cmd_product_run(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    config = load_config(args.config)
    result = run_product_matter(
        objective=args.objective,
        paths=args.path,
        matter_id=args.matter_id,
        chat_id=args.chat_id,
        config=config,
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        live_synthesis=bool(args.live_synthesis),
        use_llm_planning=bool(args.worker_source_planning),
        top_k=args.top_k,
        max_files=args.max_files,
    )
    print_json(result.to_dict() | {"summary": trace_summary(result.state.to_trace())})
    return 0


def cmd_product_ui(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    config = load_config(args.config)
    serve_product_ui(
        host=args.host,
        port=args.port,
        config=config,
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
    )
    return 0


def cmd_experiment_open(args: argparse.Namespace) -> int:
    path = open_experiment(
        baseline_run=args.baseline,
        hypothesis=args.hypothesis,
        target=args.target,
        experiments_dir=args.experiments_dir,
    )
    print_json({"experiment": str(path)})
    return 0


def cmd_experiment_close(args: argparse.Namespace) -> int:
    baseline = read_experiment(args.experiment).baseline_run
    record = close_experiment(
        args.experiment,
        experiment_run=args.run,
        accepted=bool(args.accept),
        decision_reason=args.decision_reason,
        comparison=compare_run_dirs(baseline, args.run),
    )
    print_json(record.to_dict())
    return 0


def cmd_prepare_harvey_eval(args: argparse.Namespace) -> int:
    trace = load_trace(args.trace)
    package = prepare_harvey_eval_package(
        trace,
        harvey_root=args.harvey_root or default_harvey_root(),
        run_id=args.run_id,
    )
    print_json(package.to_dict())
    return 0 if package.copied_files else 1


def cmd_score_harvey(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    config = load_config()
    judge_model = args.judge_model or config.judge_model
    harvey_root = args.harvey_root or default_harvey_root()
    if not args.execute:
        print_json(
            {
                "dry_run": True,
                "run_id": args.run_id,
                "task_id": args.task_id,
                "harvey_root": str(harvey_root),
                "judge_model": judge_model,
                "parallel": args.parallel,
                "execute_with": "--execute",
            }
        )
        return 0
    scores = evaluate_prepared_harvey_run(
        harvey_root=harvey_root,
        run_id=args.run_id,
        task_id=args.task_id,
        judge_model=judge_model,
        parallel=args.parallel,
    )
    print_json(scores)
    return 0


def cmd_harvey_smoke(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    config = load_config(args.config)
    task_ids = resolve_harvey_smoke_task_ids(
        task_ids=list(args.task_id or []),
        task_file=args.task_file,
        split=args.split,
        limit=args.limit,
        harvey_root=args.harvey_root,
    )
    results = run_harvey_batch(
        task_ids=task_ids,
        config=config,
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        harvey_root=args.harvey_root,
        run_prefix=args.run_prefix,
        live_synthesis=not args.no_live_synthesis,
        execute_score=bool(args.execute_score),
        judge_model=config.judge_model,
        score_parallel=args.score_parallel,
        workers=args.workers,
        resume=bool(args.resume),
    )
    report = build_harvey_batch_report(
        results,
        requested_task_ids=task_ids,
        trace_dir=args.trace_dir,
        run_prefix=args.run_prefix,
        split=args.split,
        limit=args.limit,
    )
    tracking_paths = write_harvey_batch_tracking(report, summary_path=args.summary_path)
    report["tracking_paths"] = tracking_paths
    print_json(report)
    return 0 if all(item.error is None for item in results) else 1


def resolve_harvey_smoke_task_ids(
    *,
    task_ids: list[str],
    task_file: str | Path | None,
    split: str | None,
    limit: int | None,
    harvey_root: str | Path | None,
) -> list[str]:
    explicit_ids = list(task_ids or [])
    if task_file:
        explicit_ids.extend(read_task_file(task_file))
    if explicit_ids:
        return dedupe(explicit_ids)
    if split:
        adapter = HarveyLabAdapter(root=harvey_root or default_harvey_root())
        refs = adapter.list_tasks(split)
        selected_limit = limit if limit is not None else len(refs)
        return dedupe([ref.task_id for ref in refs[:selected_limit]])
    return []


def cmd_agent_bench_smoke(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    config = load_config(args.config)
    raw_specs = list(args.benchmark or [])
    if args.benchmark_file:
        raw_specs.extend(read_benchmark_spec_file(args.benchmark_file))
    specs = parse_benchmark_specs(raw_specs or DEFAULT_BENCHMARK_SPECS)
    if args.dry_run:
        print_json(
            {
                "dry_run": True,
                "backend_mode": args.backend_mode,
                "agent_bench_root": args.agent_bench_root,
                "data_dir": args.data_dir,
                "results_dir": args.results_dir,
                "trace_dir": args.trace_dir,
                "limit": args.limit,
                "concurrency": args.concurrency,
                "benchmark_workers": args.benchmark_workers,
                "benchmarks": [spec.__dict__ for spec in specs],
            }
        )
        return 0
    report = asyncio.run(
        run_agent_bench_suite(
            config=config,
            specs=specs,
            agent_bench_root=args.agent_bench_root,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            trace_dir=args.trace_dir,
            backend_mode=args.backend_mode,
            limit=args.limit,
            concurrency=args.concurrency,
            benchmark_workers=args.benchmark_workers,
            checkpoint_every=args.checkpoint_every,
            resume=bool(args.resume),
        )
    )
    print_json(report)
    return 0 if not report.get("error_benchmarks") else 1


def build_harvey_batch_report(
    results: list[HarveyPipelineResult],
    *,
    requested_task_ids: list[str],
    trace_dir: str | Path,
    run_prefix: str,
    split: str | None,
    limit: int | None,
) -> dict[str, Any]:
    passed = sum(1 for item in results if item.passed is True)
    evaluated = [item for item in results if item.rubric_pass_rate is not None]
    average_rubric_pass_rate = (
        sum(float(item.rubric_pass_rate or 0.0) for item in evaluated) / len(evaluated)
        if evaluated
        else None
    )
    tokens_by_tier: dict[str, float] = {}
    for item in results:
        for tier, share in item.token_share_by_tier.items():
            tokens_by_tier[tier] = tokens_by_tier.get(tier, 0.0) + float(share or 0.0)
    average_token_share_by_tier = {
        tier: value / len(results) for tier, value in sorted(tokens_by_tier.items())
    } if results else {}
    task_costs = [float(item.estimated_cost or 0.0) for item in results]
    evaluated_task_costs = [float(item.estimated_cost or 0.0) for item in evaluated]
    task_tokens = [int(item.total_tokens or 0) for item in results]
    total_cost = sum(task_costs)
    total_tokens = sum(task_tokens)
    max_cost_item = max(results, key=lambda item: float(item.estimated_cost or 0.0), default=None)
    max_token_item = max(results, key=lambda item: int(item.total_tokens or 0), default=None)
    max_cost_task = (
        {
            "task_id": max_cost_item.task_id,
            "estimated_cost": float(max_cost_item.estimated_cost or 0.0),
            "total_tokens": int(max_cost_item.total_tokens or 0),
        }
        if max_cost_item
        else None
    )
    max_token_task = (
        {
            "task_id": max_token_item.task_id,
            "estimated_cost": float(max_token_item.estimated_cost or 0.0),
            "total_tokens": int(max_token_item.total_tokens or 0),
        }
        if max_token_item
        else None
    )

    status_counts = Counter(item.status for item in results)
    failed_tasks = [
        item.task_id
        for item in results
        if item.error is None and item.rubric_pass_rate is not None and item.passed is not True
    ]
    error_tasks = [item.task_id for item in results if item.error is not None]
    incomplete_tasks = [
        item.task_id
        for item in results
        if item.error is None and item.rubric_pass_rate is None
    ]
    non_pass_tasks = dedupe(failed_tasks + error_tasks + incomplete_tasks)

    by_practice_area: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "tasks": 0,
            "passed": 0,
            "evaluated": 0,
            "errors": 0,
            "rubric_passed": 0,
            "rubric_total": 0,
            "estimated_cost": 0.0,
            "total_tokens": 0,
            "average_rubric_pass_rate": None,
            "average_cost_per_task": 0.0,
            "average_cost_per_evaluated_task": None,
            "average_tokens_per_task": 0.0,
            "failure_tag_counts": {},
            "suspected_module_counts": {},
            "suspected_actor_counts": {},
        }
    )
    failure_tag_counts: Counter[str] = Counter()
    suspected_module_counts: Counter[str] = Counter()
    suspected_actor_counts: Counter[str] = Counter()
    task_diagnostics: dict[str, dict[str, Any]] = {}
    for item in results:
        area = item.task_id.split("/", 1)[0]
        row = by_practice_area[area]
        row["tasks"] += 1
        row["estimated_cost"] += float(item.estimated_cost or 0.0)
        row["total_tokens"] += int(item.total_tokens or 0)
        if item.passed is True:
            row["passed"] += 1
        if item.error is not None:
            row["errors"] += 1
        if item.rubric_passed is not None and item.rubric_total is not None:
            row["evaluated"] += 1
            row["rubric_passed"] += int(item.rubric_passed)
            row["rubric_total"] += int(item.rubric_total)
        diagnosis = load_result_diagnosis(item)
        if diagnosis and (
            diagnosis.get("failure_tags")
            or diagnosis.get("suspected_module")
            or diagnosis.get("suspected_actor")
            or diagnosis.get("recommended_experiment")
        ):
            task_diagnostics[item.task_id] = diagnosis
        area_failure_tags: Counter[str] = Counter(row["failure_tag_counts"])
        area_modules: Counter[str] = Counter(row["suspected_module_counts"])
        area_actors: Counter[str] = Counter(row["suspected_actor_counts"])
        for tag in diagnosis.get("failure_tags", []):
            failure_tag_counts[str(tag)] += 1
            area_failure_tags[str(tag)] += 1
        module = diagnosis.get("suspected_module")
        if module:
            suspected_module_counts[str(module)] += 1
            area_modules[str(module)] += 1
        actor = diagnosis.get("suspected_actor")
        if actor:
            suspected_actor_counts[str(actor)] += 1
            area_actors[str(actor)] += 1
        row["failure_tag_counts"] = dict(sorted(area_failure_tags.items()))
        row["suspected_module_counts"] = dict(sorted(area_modules.items()))
        row["suspected_actor_counts"] = dict(sorted(area_actors.items()))
    for row in by_practice_area.values():
        row["average_rubric_pass_rate"] = (
            row["rubric_passed"] / row["rubric_total"] if row["rubric_total"] else None
        )
        row["average_cost_per_task"] = row["estimated_cost"] / row["tasks"] if row["tasks"] else 0.0
        row["average_cost_per_evaluated_task"] = (
            row["estimated_cost"] / row["evaluated"] if row["evaluated"] else None
        )
        row["average_tokens_per_task"] = row["total_tokens"] / row["tasks"] if row["tasks"] else 0.0

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_prefix": run_prefix,
        "trace_dir": str(trace_dir),
        "split": split,
        "limit": limit,
        "requested_tasks": len(requested_task_ids),
        "tasks": len(results),
        "passed": passed,
        "evaluated": len(evaluated),
        "errors": len(error_tasks),
        "incomplete": len(incomplete_tasks),
        "failed": len(failed_tasks),
        "average_rubric_pass_rate": average_rubric_pass_rate,
        "estimated_cost": total_cost,
        "average_cost_per_task": total_cost / len(results) if results else 0.0,
        "average_cost_per_evaluated_task": (
            sum(evaluated_task_costs) / len(evaluated_task_costs) if evaluated_task_costs else None
        ),
        "median_cost_per_task": median_number(task_costs),
        "max_cost_task": max_cost_task,
        "total_tokens": total_tokens,
        "average_tokens_per_task": total_tokens / len(results) if results else 0.0,
        "median_tokens_per_task": median_number([float(value) for value in task_tokens]),
        "max_token_task": max_token_task,
        "average_token_share_by_tier": average_token_share_by_tier,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_tag_counts": dict(sorted(failure_tag_counts.items())),
        "suspected_module_counts": dict(sorted(suspected_module_counts.items())),
        "suspected_actor_counts": dict(sorted(suspected_actor_counts.items())),
        "by_practice_area": dict(sorted(by_practice_area.items())),
        "task_diagnostics": task_diagnostics,
        "failed_tasks": failed_tasks,
        "error_tasks": error_tasks,
        "incomplete_tasks": incomplete_tasks,
        "non_pass_tasks": non_pass_tasks,
        "results": [item.to_dict() for item in results],
    }


def load_result_diagnosis(item: HarveyPipelineResult) -> dict[str, Any]:
    if not item.trace_path:
        return {}
    path = Path(item.trace_path)
    if not path.exists():
        return {}
    try:
        trace = load_trace(path)
    except (OSError, json.JSONDecodeError):
        return {}
    diagnosis = trace.get("diagnosis") or {}
    return {
        "failure_tags": trace.get("failure_tags", []),
        "suspected_module": diagnosis.get("suspected_module"),
        "suspected_actor": diagnosis.get("suspected_actor"),
        "recommended_experiment": diagnosis.get("recommended_experiment"),
    }


def write_harvey_batch_tracking(
    report: dict[str, Any],
    *,
    summary_path: str | Path | None = None,
) -> dict[str, str]:
    base = Path(summary_path) if summary_path else Path(str(report["trace_dir"])) / "harvey_batch_summary.json"
    base.parent.mkdir(parents=True, exist_ok=True)
    with base.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    paths = {"summary": str(base)}
    task_files = {
        "failed_task_file": report.get("failed_tasks", []),
        "error_task_file": report.get("error_tasks", []),
        "incomplete_task_file": report.get("incomplete_tasks", []),
        "non_pass_task_file": report.get("non_pass_tasks", []),
    }
    for label, tasks in task_files.items():
        path = base.parent / f"{label.replace('_file', 's')}.txt"
        write_task_lines(path, list(tasks))
        paths[label] = str(path)
    return paths


def write_task_lines(path: str | Path, tasks: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(f"{task}\n")


def get_adapter(name: str) -> FixtureAdapter | HarveyLabAdapter:
    if name == "fixture":
        return FixtureAdapter()
    if name in {"harvey_lab_sample", "harvey_lab"}:
        return HarveyLabAdapter()
    raise SystemExit(f"Unknown benchmark: {name}")


def load_trace_dir(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    files = sorted(root.rglob("*.json")) if root.exists() else []
    traces: list[dict[str, Any]] = []
    for file in files:
        trace = load_trace(file)
        if not is_trace_json(trace):
            continue
        traces.append(trace)
    return traces


def is_trace_json(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("benchmark"), str)
        and bool(value.get("benchmark"))
        and isinstance(value.get("task_id"), str)
        and bool(value.get("task_id"))
    )


def trace_key(trace: dict[str, Any]) -> str:
    return f"{trace.get('benchmark')}::{trace.get('task_id')}"


def aggregate_tokens_by_tier(traces: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for trace in traces:
        for tier, tokens in (trace.get("metrics", {}).get("tokens_by_tier", {}) or {}).items():
            totals[tier] = totals.get(tier, 0) + int(tokens or 0)
    return totals


def token_share(tokens_by_tier: dict[str, int]) -> dict[str, float]:
    total = sum(tokens_by_tier.values())
    if total == 0:
        return {tier: 0.0 for tier in sorted(tokens_by_tier)}
    return {tier: tokens / total for tier, tokens in sorted(tokens_by_tier.items())}


def read_task_file(path: str | Path) -> list[str]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            tasks.append(value)
    return tasks


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
