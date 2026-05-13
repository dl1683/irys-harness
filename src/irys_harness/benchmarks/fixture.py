from __future__ import annotations

from irys_harness.benchmarks.base import BenchmarkAdapter
from irys_harness.events import EventLogger
from irys_harness.failures import FailureTag
from irys_harness.metrics import ModelCallRecord, QualityMetrics
from irys_harness.state import BenchmarkTask, RunState, ScoreResult


class FixtureAdapter(BenchmarkAdapter):
    name = "fixture"

    def load_task(self, task_id: str) -> BenchmarkTask:
        return BenchmarkTask(
            benchmark=self.name,
            task_id=task_id,
            question="Answer exactly: PASS",
            answer_schema={"type": "exact_string", "expected": "PASS"},
            metadata={"purpose": "smoke_test"},
        )

    def run(self, state: RunState) -> RunState:
        log = EventLogger(state)
        log.emit("RUN", "started", benchmark=state.task.benchmark, task=state.task.task_id)

        contract_model = state.config.model_for_module("contract")
        state.metrics.add_call(
            ModelCallRecord.from_usage(
                module="contract",
                model_config=contract_model,
                input_tokens=25,
                output_tokens=10,
                latency_seconds=0.01,
            )
        )
        state.answer_contract_versions.append(
            {
                "version": 1,
                "interpreted_goal": "Return the exact sentinel answer.",
                "required_output_format": "exact_string",
                "needed_information": ["expected sentinel"],
                "search_queries": ["PASS sentinel"],
                "verification_requirements": ["rendered answer must equal expected string"],
                "scoring_risks": ["extra prose causes format failure"],
            }
        )
        log.emit("CONTRACT", "built", format="exact_string")

        extraction_model = state.config.model_for_module("extraction")
        state.metrics.add_call(
            ModelCallRecord.from_usage(
                module="extraction",
                model_config=extraction_model,
                input_tokens=900,
                output_tokens=30,
                latency_seconds=0.01,
            )
        )
        state.retrieval_iterations.append(
            {
                "iteration": 1,
                "queries": ["PASS sentinel"],
                "retrieved_chunks": ["fixture_chunk_1"],
                "reason": "Fixture task has one synthetic expected chunk.",
            }
        )
        state.extraction_records.append(
            {
                "evidence_items": [
                    {
                        "claim": "The exact expected answer is PASS.",
                        "raw_support": "expected=PASS",
                        "source": {"doc_id": "fixture_doc", "chunk_id": "fixture_chunk_1"},
                        "confidence": "high",
                        "directness": "direct",
                    }
                ]
            }
        )
        log.emit("EXTRACT", "candidate evidence extracted", candidates=1)

        verification_model = state.config.model_for_module("verification")
        state.metrics.add_call(
            ModelCallRecord.from_usage(
                module="verification",
                model_config=verification_model,
                input_tokens=15,
                output_tokens=5,
                latency_seconds=0.01,
            )
        )
        state.verification_records.append(
            {
                "accepted": ["The exact expected answer is PASS."],
                "rejected": [],
                "weak": [],
            }
        )
        log.emit("VERIFY", "evidence verified", accepted=1, rejected=0)

        state.final_packet = {
            "verified_evidence": ["The exact expected answer is PASS."],
            "unresolved": [],
        }

        synthesis_model = state.config.model_for_module("synthesis")
        state.metrics.add_call(
            ModelCallRecord.from_usage(
                module="synthesis",
                model_config=synthesis_model,
                input_tokens=15,
                output_tokens=5,
                latency_seconds=0.01,
            )
        )
        state.draft_answer = "PASS"
        state.rendered_answer = "PASS"
        log.emit("RENDER", "answer rendered", schema="exact_string")

        state.scoring_result = self.score(state)
        state.metrics.quality = QualityMetrics(
            score=state.scoring_result.score,
            max_score=state.scoring_result.max_score,
            passed=state.scoring_result.passed,
        )
        if not state.scoring_result.passed:
            state.failure_tags.append(FailureTag.FORMAT_ERROR.value)
        log.emit("SCORE", "scored", passed=state.scoring_result.passed, score=state.scoring_result.score)
        return state

    def score(self, state: RunState) -> ScoreResult:
        expected = state.task.answer_schema.get("expected")
        passed = state.rendered_answer == expected
        return ScoreResult(
            score=1.0 if passed else 0.0,
            max_score=1.0,
            passed=passed,
            details={"expected": expected, "actual": state.rendered_answer},
        )
