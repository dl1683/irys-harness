from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .diagnosis import diagnose_harvey_scores
from .failures import FailureTag
from .state import RunState


class TraceWriter:
    def __init__(self, trace_dir: str | Path) -> None:
        self.trace_dir = Path(trace_dir)

    def write(self, state: RunState) -> Path:
        target = self.trace_dir / state.task.benchmark / f"{state.task.task_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(state.to_trace(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return target


def load_trace(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_trace(path: str | Path, trace: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2, sort_keys=True)
        handle.write("\n")


def attach_harvey_scores(trace: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    updated = dict(trace)
    n_passed = scores.get("n_passed")
    n_criteria = scores.get("n_criteria")
    updated["scoring_result"] = {
        "score": scores.get("score"),
        "passed": bool(scores.get("all_pass")),
        "details": {
            "run_id": scores.get("run_id"),
            "task": scores.get("task"),
            "summary": scores.get("summary"),
            "judge_model": scores.get("judge_model"),
            "scored_at": scores.get("scored_at"),
            "n_passed": n_passed,
            "n_criteria": n_criteria,
            "criteria_results": scores.get("criteria_results", []),
        },
    }
    metrics = dict(updated.get("metrics", {}) or {})
    quality = dict(metrics.get("quality", {}) or {})
    quality.update(
        {
            "rubric_passed": n_passed,
            "rubric_total": n_criteria,
            "rubric_pass_rate": n_passed / n_criteria if n_criteria else None,
        }
    )
    metrics["quality"] = quality
    updated["metrics"] = metrics
    diagnosis = diagnose_harvey_scores(scores)
    updated["failure_tags"] = diagnosis.get("failure_tags", [])
    updated["diagnosis"] = diagnosis
    return updated


def trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    metrics = trace.get("metrics", {})
    quality = metrics.get("quality", {}) or {}
    scoring_result = trace.get("scoring_result") or {}
    passed = scoring_result.get("passed")
    scoring_details = scoring_result.get("details") or {}
    if passed is None and scoring_details.get("mode") == "metadata_readiness":
        rubric_passed = None
        rubric_total = None
    else:
        rubric_passed = quality.get("rubric_passed")
        rubric_total = quality.get("rubric_total")
    return {
        "run_id": trace.get("run_id"),
        "benchmark": trace.get("benchmark"),
        "task_id": trace.get("task_id"),
        "score": scoring_result.get("score"),
        "passed": passed,
        "rubric_passed": rubric_passed,
        "rubric_total": rubric_total,
        "rubric_pass_rate": rubric_passed / rubric_total if rubric_passed is not None and rubric_total else None,
        "total_tokens": metrics.get("total_tokens", 0),
        "estimated_cost": metrics.get("estimated_cost", 0.0),
        "token_share_by_tier": metrics.get("token_share_by_tier", {}),
        "failure_tags": trace.get("failure_tags", []),
    }


def diagnose_trace(trace: dict[str, Any]) -> dict[str, Any]:
    missing_sections = [
        section
        for section in [
            "answer_contract_versions",
            "retrieval_iterations",
            "extraction_records",
            "verification_records",
            "final_packet",
            "metrics",
        ]
        if not trace.get(section)
    ]
    scoring = trace.get("scoring_result") or {}
    passed = scoring.get("passed")
    failure_tags = list(trace.get("failure_tags", []))
    if passed is False and not failure_tags:
        failure_tags.append(FailureTag.UNCLASSIFIED_FAILURE.value)
    if missing_sections:
        failure_tags.append(FailureTag.TRACE_INCOMPLETE.value)

    suspected_module = None
    if FailureTag.FORMAT_ERROR.value in failure_tags:
        suspected_module = "renderer"
    elif FailureTag.RETRIEVAL_MISS.value in failure_tags:
        suspected_module = "retriever"
    elif FailureTag.BAD_EXTRACTION.value in failure_tags:
        suspected_module = "extractor"
    elif FailureTag.TRACE_INCOMPLETE.value in failure_tags:
        suspected_module = "trace_writer"

    return {
        "task_id": trace.get("task_id"),
        "failed": passed is False,
        "failure_tags": sorted(set(failure_tags)),
        "suspected_module": suspected_module,
        "missing_trace_sections": missing_sections,
        "supporting_trace_refs": [
            {"path": section, "reason": "Section is empty or missing"}
            for section in missing_sections
        ],
        "recommended_experiment": {
            "change_type": "diagnosis",
            "target": suspected_module,
            "validation": ["same_task"],
        }
        if suspected_module
        else None,
    }
