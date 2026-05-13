from __future__ import annotations

from abc import ABC, abstractmethod

from irys_harness.state import BenchmarkTask, RunState, ScoreResult


class BenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def load_task(self, task_id: str) -> BenchmarkTask:
        raise NotImplementedError

    @abstractmethod
    def run(self, state: RunState) -> RunState:
        raise NotImplementedError

    @abstractmethod
    def score(self, state: RunState) -> ScoreResult:
        raise NotImplementedError

