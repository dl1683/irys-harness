from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HarveyEvalPackage:
    run_id: str
    task_id: str
    run_dir: str
    output_dir: str
    copied_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "run_dir": self.run_dir,
            "output_dir": self.output_dir,
            "copied_files": self.copied_files,
        }


def prepare_harvey_eval_package(
    trace: dict[str, Any],
    *,
    harvey_root: str | Path,
    run_id: str | None = None,
) -> HarveyEvalPackage:
    if trace.get("benchmark") != "harvey_lab_sample":
        raise ValueError("Trace is not a Harvey LAB trace")
    task_id = str(trace["task_id"])
    resolved_run_id = run_id or f"irys/{trace['run_id']}"
    run_dir = Path(harvey_root) / "results" / resolved_run_id
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for artifact in trace.get("artifacts", []):
        source = Path(artifact["path"])
        if not source.exists():
            continue
        target = output_dir / source.name
        shutil.copy2(source, target)
        copied.append(str(target))

    return HarveyEvalPackage(
        run_id=resolved_run_id,
        task_id=task_id,
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        copied_files=copied,
    )


def evaluate_prepared_harvey_run(
    *,
    harvey_root: str | Path,
    run_id: str,
    task_id: str,
    judge_model: str,
    parallel: int = 24,
) -> dict[str, Any]:
    root = Path(harvey_root).resolve()
    sys.path.insert(0, str(root))
    try:
        from evaluation.judge import Judge  # type: ignore
        from evaluation.run_eval import evaluate_run  # type: ignore

        judge = Judge(model=judge_model)
        return evaluate_run(run_id=run_id, task=task_id, judge=judge, parallel=parallel)
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
