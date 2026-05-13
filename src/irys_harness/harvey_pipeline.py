from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmarks.harvey import HarveyLabAdapter, default_harvey_root
from .benchmarks.harvey_eval import evaluate_prepared_harvey_run, prepare_harvey_eval_package
from .config import HarnessConfig
from .state import RunState
from .trace import TraceWriter, attach_harvey_scores, load_trace, save_trace, trace_summary


@dataclass(frozen=True)
class HarveyPipelineResult:
    task_id: str
    trace_path: str | None
    run_id: str
    passed: bool | None
    rubric_passed: int | None
    rubric_total: int | None
    rubric_pass_rate: float | None
    token_share_by_tier: dict[str, float]
    error: str | None = None
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trace_path": self.trace_path,
            "run_id": self.run_id,
            "passed": self.passed,
            "rubric_passed": self.rubric_passed,
            "rubric_total": self.rubric_total,
            "rubric_pass_rate": self.rubric_pass_rate,
            "token_share_by_tier": self.token_share_by_tier,
            "error": self.error,
            "status": self.status,
        }


def run_harvey_batch(
    *,
    task_ids: list[str],
    config: HarnessConfig,
    trace_dir: str | Path,
    output_dir: str | Path,
    harvey_root: str | Path | None = None,
    run_prefix: str = "irys-smoke",
    live_synthesis: bool = True,
    execute_score: bool = False,
    judge_model: str | None = None,
    score_parallel: int = 24,
    workers: int = 24,
    resume: bool = False,
) -> list[HarveyPipelineResult]:
    if not task_ids:
        return []
    resumed: list[HarveyPipelineResult] = []
    pending: list[str] = []
    for task_id in task_ids:
        if resume:
            existing = maybe_resume_harvey_task(
                task_id=task_id,
                trace_dir=trace_dir,
                run_prefix=run_prefix,
                execute_score=execute_score,
                harvey_root=harvey_root,
                judge_model=judge_model or config.judge_model,
                score_parallel=score_parallel,
            )
            if existing is not None:
                resumed.append(existing)
                continue
        pending.append(task_id)
    if not pending:
        return sorted(resumed, key=lambda item: item.task_id)
    max_workers = max(1, min(workers, len(pending)))
    results: list[HarveyPipelineResult] = list(resumed)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_harvey_task_pipeline,
                task_id=task_id,
                config=config,
                trace_dir=trace_dir,
                output_dir=output_dir,
                harvey_root=harvey_root,
                run_prefix=run_prefix,
                live_synthesis=live_synthesis,
                execute_score=execute_score,
                judge_model=judge_model,
                score_parallel=score_parallel,
            )
            for task_id in pending
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.task_id)


def maybe_resume_harvey_task(
    *,
    task_id: str,
    trace_dir: str | Path,
    run_prefix: str,
    execute_score: bool,
    harvey_root: str | Path | None,
    judge_model: str,
    score_parallel: int,
) -> HarveyPipelineResult | None:
    trace_path = expected_harvey_trace_path(trace_dir, task_id)
    if not trace_path.exists():
        return None
    trace = load_trace(trace_path)
    summary = trace_summary(trace)
    run_id = f"{run_prefix}\\{task_id.replace('/', '--')}"
    if execute_score and summary.get("rubric_pass_rate") is None:
        try:
            package = prepare_harvey_eval_package(
                trace,
                harvey_root=harvey_root or default_harvey_root(),
                run_id=run_id,
            )
            scores = evaluate_prepared_harvey_run(
                harvey_root=harvey_root or default_harvey_root(),
                run_id=package.run_id,
                task_id=task_id,
                judge_model=judge_model,
                parallel=score_parallel,
            )
            updated = attach_harvey_scores(trace, scores)
            save_trace(trace_path, updated)
            summary = trace_summary(updated)
            return result_from_summary(
                task_id=task_id,
                trace_path=trace_path,
                run_id=run_id,
                summary=summary,
                status="resumed_scored",
            )
        except Exception as exc:  # noqa: BLE001 - resume should not hide per-task scoring failures.
            return HarveyPipelineResult(
                task_id=task_id,
                trace_path=str(trace_path),
                run_id=run_id,
                passed=False,
                rubric_passed=None,
                rubric_total=None,
                rubric_pass_rate=None,
                token_share_by_tier=summary.get("token_share_by_tier", {}),
                error=f"{type(exc).__name__}: {exc}",
                status="resume_score_error",
            )
    if execute_score and summary.get("rubric_pass_rate") is None:
        return None
    return result_from_summary(
        task_id=task_id,
        trace_path=trace_path,
        run_id=run_id,
        summary=summary,
        status="resumed",
    )


def expected_harvey_trace_path(trace_dir: str | Path, task_id: str) -> Path:
    return Path(trace_dir) / "harvey_lab_sample" / f"{task_id}.json"


def result_from_summary(
    *,
    task_id: str,
    trace_path: str | Path,
    run_id: str,
    summary: dict[str, Any],
    status: str,
) -> HarveyPipelineResult:
    return HarveyPipelineResult(
        task_id=task_id,
        trace_path=str(trace_path),
        run_id=run_id,
        passed=summary.get("passed"),
        rubric_passed=summary.get("rubric_passed"),
        rubric_total=summary.get("rubric_total"),
        rubric_pass_rate=summary.get("rubric_pass_rate"),
        token_share_by_tier=summary.get("token_share_by_tier", {}),
        status=status,
    )


def run_harvey_task_pipeline(
    *,
    task_id: str,
    config: HarnessConfig,
    trace_dir: str | Path,
    output_dir: str | Path,
    harvey_root: str | Path | None = None,
    run_prefix: str = "irys-smoke",
    live_synthesis: bool = True,
    execute_score: bool = False,
    judge_model: str | None = None,
    score_parallel: int = 24,
) -> HarveyPipelineResult:
    try:
        adapter = HarveyLabAdapter(root=harvey_root or default_harvey_root(), live_synthesis=live_synthesis)
        task = adapter.load_task(task_id)
        task_output_dir = Path(output_dir) / task.benchmark / task.task_id
        state = RunState(task=task, config=config, output_dir=str(task_output_dir))
        state = adapter.run(state)
        trace_path = TraceWriter(trace_dir).write(state)
        safe_task_id = task_id.replace("/", "--")
        run_id = f"{run_prefix}\\{safe_task_id}"
        package = prepare_harvey_eval_package(
            load_trace(trace_path),
            harvey_root=harvey_root or default_harvey_root(),
            run_id=run_id,
        )
        if not execute_score:
            summary = trace_summary(load_trace(trace_path))
            return HarveyPipelineResult(
                task_id=task_id,
                trace_path=str(trace_path),
                run_id=package.run_id,
                passed=summary.get("passed"),
                rubric_passed=None,
                rubric_total=None,
                rubric_pass_rate=None,
                token_share_by_tier=summary.get("token_share_by_tier", {}),
            )
        scores = evaluate_prepared_harvey_run(
            harvey_root=harvey_root or default_harvey_root(),
            run_id=run_id,
            task_id=task_id,
            judge_model=judge_model or config.judge_model,
            parallel=score_parallel,
        )
        updated = attach_harvey_scores(load_trace(trace_path), scores)
        save_trace(trace_path, updated)
        summary = trace_summary(updated)
        return HarveyPipelineResult(
            task_id=task_id,
            trace_path=str(trace_path),
            run_id=run_id,
            passed=summary.get("passed"),
            rubric_passed=summary.get("rubric_passed"),
            rubric_total=summary.get("rubric_total"),
            rubric_pass_rate=summary.get("rubric_pass_rate"),
            token_share_by_tier=summary.get("token_share_by_tier", {}),
        )
    except Exception as exc:  # noqa: BLE001 - batch runner must report per-task failures.
        return HarveyPipelineResult(
            task_id=task_id,
            trace_path=None,
            run_id=f"{run_prefix}\\{task_id.replace('/', '--')}",
            passed=False,
            rubric_passed=None,
            rubric_total=None,
            rubric_pass_rate=None,
            token_share_by_tier={},
            error=f"{type(exc).__name__}: {exc}",
        )
