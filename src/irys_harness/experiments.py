from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ExperimentRecord:
    experiment_id: str
    hypothesis: str
    target: str
    baseline_run: str
    status: str = "open"
    experiment_run: str | None = None
    accepted: bool | None = None
    decision_reason: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    closed_at: str | None = None
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "target": self.target,
            "baseline_run": self.baseline_run,
            "status": self.status,
            "experiment_run": self.experiment_run,
            "accepted": self.accepted,
            "decision_reason": self.decision_reason,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "comparison": self.comparison,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=raw["experiment_id"],
            hypothesis=raw["hypothesis"],
            target=raw["target"],
            baseline_run=raw["baseline_run"],
            status=raw.get("status", "open"),
            experiment_run=raw.get("experiment_run"),
            accepted=raw.get("accepted"),
            decision_reason=raw.get("decision_reason"),
            created_at=raw.get("created_at", datetime.now(UTC).isoformat()),
            closed_at=raw.get("closed_at"),
            comparison=raw.get("comparison", {}),
        )


def open_experiment(
    *,
    baseline_run: str,
    hypothesis: str,
    target: str,
    experiments_dir: str | Path = "experiments",
) -> Path:
    record = ExperimentRecord(
        experiment_id=f"exp_{uuid4().hex[:12]}",
        hypothesis=hypothesis,
        target=target,
        baseline_run=baseline_run,
    )
    directory = Path(experiments_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.experiment_id}.json"
    write_experiment(path, record)
    return path


def close_experiment(
    path: str | Path,
    *,
    experiment_run: str,
    accepted: bool,
    decision_reason: str,
    comparison: dict[str, Any],
) -> ExperimentRecord:
    record = read_experiment(path)
    record.status = "accepted" if accepted else "rejected"
    record.experiment_run = experiment_run
    record.accepted = accepted
    record.decision_reason = decision_reason
    record.closed_at = datetime.now(UTC).isoformat()
    record.comparison = comparison
    write_experiment(path, record)
    return record


def read_experiment(path: str | Path) -> ExperimentRecord:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ExperimentRecord.from_dict(json.load(handle))


def write_experiment(path: str | Path, record: ExperimentRecord) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
