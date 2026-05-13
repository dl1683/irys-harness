from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .config import HarnessConfig
from .metrics import RunMetrics


@dataclass
class BenchmarkTask:
    benchmark: str
    task_id: str
    question: str
    context_files: list[str] = field(default_factory=list)
    answer_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "question": self.question,
            "context_files": self.context_files,
            "answer_schema": self.answer_schema,
            "metadata": self.metadata,
        }


@dataclass
class ScoreResult:
    score: float | None = None
    max_score: float | None = None
    passed: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "max_score": self.max_score,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class RunState:
    task: BenchmarkTask
    config: HarnessConfig
    run_id: str = field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    output_dir: str | None = None
    documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    answer_contract_versions: list[dict[str, Any]] = field(default_factory=list)
    retrieval_iterations: list[dict[str, Any]] = field(default_factory=list)
    extraction_records: list[dict[str, Any]] = field(default_factory=list)
    critic_records: list[dict[str, Any]] = field(default_factory=list)
    verification_records: list[dict[str, Any]] = field(default_factory=list)
    computation_records: list[dict[str, Any]] = field(default_factory=list)
    final_packet: dict[str, Any] | None = None
    draft_answer: str | None = None
    rendered_answer: str | None = None
    scoring_result: ScoreResult | None = None
    failure_tags: list[str] = field(default_factory=list)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_trace(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark": self.task.benchmark,
            "task_id": self.task.task_id,
            "started_at": self.started_at,
            "output_dir": self.output_dir,
            "config": self.config.to_dict(),
            "isolation": {
                "memory_enabled": self.config.run.memory_enabled,
                "web_enabled": self.config.run.web_enabled,
            },
            "task": self.task.to_dict(),
            "documents": self.documents,
            "chunks": self.chunks,
            "answer_contract_versions": self.answer_contract_versions,
            "retrieval_iterations": self.retrieval_iterations,
            "extraction_records": self.extraction_records,
            "critic_records": self.critic_records,
            "verification_records": self.verification_records,
            "computation_records": self.computation_records,
            "final_packet": self.final_packet,
            "draft_answer": self.draft_answer,
            "rendered_answer": self.rendered_answer,
            "scoring_result": self.scoring_result.to_dict() if self.scoring_result else None,
            "failure_tags": self.failure_tags,
            "diagnosis": self.diagnosis,
            "artifacts": self.artifacts,
            "metrics": self.metrics.to_dict(),
            "events": self.events,
        }
