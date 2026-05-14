from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from irys_harness.cli import main
from irys_harness.cli import (
    aggregate_tokens_by_tier,
    build_harvey_batch_report,
    compare_run_dirs,
    dedupe,
    load_trace_dir,
    read_task_file,
    refresh_harvey_diagnostics,
    resolve_harvey_smoke_task_ids,
    token_share,
    write_harvey_batch_tracking,
)
from irys_harness.harvey_pipeline import HarveyPipelineResult, maybe_resume_harvey_task
from irys_harness.trace import attach_harvey_scores, diagnose_trace, load_trace, trace_summary


def json_trace(
    *,
    task_id: str,
    passed: bool,
    rubric_passed: int,
    rubric_total: int,
    cost: float,
    tokens: int,
) -> str:
    return json.dumps(
        {
            "benchmark": "harvey_lab_sample",
            "task_id": task_id,
            "scoring_result": {"passed": passed, "score": 0.0},
            "metrics": {
                "quality": {"rubric_passed": rubric_passed, "rubric_total": rubric_total},
                "estimated_cost": cost,
                "total_tokens": tokens,
            },
        }
    )


class TraceCliTests(unittest.TestCase):
    def test_fixture_run_writes_trace_with_cost_and_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = main(["run", "--benchmark", "fixture", "--task-id", "smoke", "--trace-dir", tmp])
            self.assertEqual(result, 0)
            trace_path = Path(tmp) / "fixture" / "smoke.json"
            self.assertTrue(trace_path.exists())
            trace = load_trace(trace_path)
            summary = trace_summary(trace)
            self.assertEqual(summary["benchmark"], "fixture")
            self.assertTrue(summary["passed"])
            self.assertGreater(summary["total_tokens"], 0)
            self.assertGreaterEqual(summary["token_share_by_tier"]["cheap_worker"], 0.9)

    def test_diagnose_trace_flags_missing_sections(self) -> None:
        diagnosis = diagnose_trace({"task_id": "bad", "scoring_result": {"passed": False}})
        self.assertTrue(diagnosis["failed"])
        self.assertIn("trace_incomplete", diagnosis["failure_tags"])
        self.assertEqual(diagnosis["suspected_module"], "trace_writer")

    def test_compare_run_dirs_reports_rubric_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before" / "harvey_lab_sample" / "area"
            after = root / "after" / "harvey_lab_sample" / "area"
            before.mkdir(parents=True)
            after.mkdir(parents=True)
            before.joinpath("task.json").write_text(
                json_trace(
                    task_id="area/task",
                    passed=False,
                    rubric_passed=6,
                    rubric_total=10,
                    cost=1.0,
                    tokens=100,
                ),
                encoding="utf-8",
            )
            after.joinpath("task.json").write_text(
                json_trace(
                    task_id="area/task",
                    passed=False,
                    rubric_passed=9,
                    rubric_total=10,
                    cost=1.5,
                    tokens=120,
                ),
                encoding="utf-8",
            )

            comparison = compare_run_dirs(root / "before", root / "after")

            self.assertEqual(comparison["summary"]["common_scored_tasks"], 1)
            self.assertEqual(comparison["summary"]["common_passed_delta"], 0)
            self.assertAlmostEqual(comparison["summary"]["common_macro_rubric_pass_rate_delta"], 0.3)
            self.assertEqual(comparison["summary"]["common_rubric_passed_delta"], 3)
            self.assertEqual(comparison["summary"]["common_token_delta"], 20)
            self.assertAlmostEqual(comparison["summary"]["common_average_cost_per_task_delta"], 0.5)
            self.assertEqual(comparison["top_gains"][0]["task"], "harvey_lab_sample::area/task")
            self.assertAlmostEqual(comparison["top_gains"][0]["rubric_pass_rate_delta"], 0.3)
            self.assertEqual(comparison["family_deltas"][0]["family"], "task")
            self.assertEqual(comparison["family_deltas"][0]["rubric_passed_delta"], 3)
            self.assertAlmostEqual(comparison["family_deltas"][0]["macro_rubric_pass_rate_delta"], 0.3)

    def test_trace_loader_accepts_bom_and_trace_dir_ignores_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            trace_dir = run / "harvey_lab_sample" / "area"
            trace_dir.mkdir(parents=True)
            trace_path = trace_dir / "task.json"
            trace_path.write_text(
                json_trace(
                    task_id="area/task",
                    passed=False,
                    rubric_passed=6,
                    rubric_total=10,
                    cost=1.0,
                    tokens=100,
                ),
                encoding="utf-8-sig",
            )
            (run / "summary.json").write_text(json.dumps({"tasks": 1}), encoding="utf-8")

            trace = load_trace(trace_path)
            traces = load_trace_dir(run)

            self.assertEqual(trace["task_id"], "area/task")
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["task_id"], "area/task")

    def test_compare_run_dirs_groups_task_family_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before" / "harvey_lab_sample" / "area"
            after = root / "after" / "harvey_lab_sample" / "area"
            before.mkdir(parents=True)
            after.mkdir(parents=True)
            before.joinpath("extract-clauses.json").write_text(
                json_trace(
                    task_id="area/extract-clauses/scenario-01",
                    passed=False,
                    rubric_passed=5,
                    rubric_total=10,
                    cost=1.0,
                    tokens=100,
                ),
                encoding="utf-8",
            )
            after.joinpath("extract-clauses.json").write_text(
                json_trace(
                    task_id="area/extract-clauses/scenario-01",
                    passed=False,
                    rubric_passed=8,
                    rubric_total=10,
                    cost=1.0,
                    tokens=100,
                ),
                encoding="utf-8",
            )
            before.joinpath("draft-memo.json").write_text(
                json_trace(
                    task_id="area/draft-memo",
                    passed=False,
                    rubric_passed=9,
                    rubric_total=10,
                    cost=1.0,
                    tokens=100,
                ),
                encoding="utf-8",
            )
            after.joinpath("draft-memo.json").write_text(
                json_trace(
                    task_id="area/draft-memo",
                    passed=False,
                    rubric_passed=7,
                    rubric_total=10,
                    cost=1.0,
                    tokens=100,
                ),
                encoding="utf-8",
            )

            comparison = compare_run_dirs(root / "before", root / "after")
            families = {item["family"]: item for item in comparison["family_deltas"]}

            self.assertEqual(families["extract"]["rubric_passed_delta"], 3)
            self.assertEqual(families["draft/package"]["rubric_passed_delta"], -2)
            self.assertEqual(
                families["extract"]["top_gain"]["task"],
                "harvey_lab_sample::area/extract-clauses/scenario-01",
            )
            self.assertEqual(
                families["draft/package"]["top_regression"]["task"],
                "harvey_lab_sample::area/draft-memo",
            )

    def test_refresh_harvey_diagnostics_updates_trace_from_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            trace_path = trace_dir / "harvey_lab_sample" / "area" / "task.json"
            trace_path.parent.mkdir(parents=True)
            run_id = "run\\area--task"
            trace_path.write_text(
                json.dumps(
                    {
                        "benchmark": "harvey_lab_sample",
                        "task_id": "area/task",
                        "scoring_result": {"details": {"run_id": run_id}, "passed": False},
                        "failure_tags": ["wrong_computation"],
                        "metrics": {"quality": {"rubric_passed": 1, "rubric_total": 2}},
                    }
                ),
                encoding="utf-8",
            )
            scores_path = root / "harvey" / "results" / Path(run_id) / "scores.json"
            scores_path.parent.mkdir(parents=True)
            scores_path.write_text(
                json.dumps(
                    {
                        "task": "area/task",
                        "all_pass": False,
                        "score": 0.0,
                        "n_passed": 1,
                        "n_criteria": 2,
                        "run_id": run_id,
                        "criteria_results": [
                            {
                                "id": "C-1",
                                "title": "Identifies second transaction",
                                "verdict": "fail",
                                "reasoning": "The output focuses exclusively on one transaction and does not discuss the other deal from the source documents.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = refresh_harvey_diagnostics(trace_dir=trace_dir, harvey_root=root / "harvey")
            refreshed = load_trace(trace_path)

            self.assertEqual(report["updated"], 1)
            self.assertIn("distractor_confusion", refreshed["failure_tags"])
            self.assertEqual(refreshed["diagnosis"]["suspected_module"], "final_packet_synthesizer")

    def test_aggregate_token_share(self) -> None:
        traces = [
            {"metrics": {"tokens_by_tier": {"cheap_worker": 90, "strong_synthesizer": 10}}},
            {"metrics": {"tokens_by_tier": {"cheap_worker": 10, "mid_orchestrator": 10}}},
        ]
        totals = aggregate_tokens_by_tier(traces)
        self.assertEqual(totals["cheap_worker"], 100)
        shares = token_share(totals)
        self.assertAlmostEqual(shares["cheap_worker"], 100 / 120)

    def test_attach_harvey_scores_updates_quality_and_diagnosis(self) -> None:
        trace = {"benchmark": "harvey_lab_sample", "task_id": "area/task", "metrics": {"quality": {}}}
        updated = attach_harvey_scores(
            trace,
            {
                "run_id": "run-1",
                "task": "area/task",
                "all_pass": False,
                "score": 0.0,
                "n_passed": 26,
                "n_criteria": 27,
                "criteria_results": [
                    {
                        "id": "C-25",
                        "title": "Correctly identifies expert letter writers whose letters are present",
                        "verdict": "fail",
                        "reasoning": "The memo does not list Johansson, Tsai, Al-Rashidi, and Moriarty.",
                    }
                ],
            },
        )
        self.assertFalse(updated["scoring_result"]["passed"])
        self.assertEqual(updated["metrics"]["quality"]["rubric_passed"], 26)
        self.assertEqual(updated["diagnosis"]["suspected_actor"], "strong_synthesizer")

    def test_metadata_readiness_trace_is_not_treated_as_scored_failure(self) -> None:
        summary = trace_summary(
            {
                "benchmark": "harvey_lab_sample",
                "task_id": "area/task",
                "scoring_result": {
                    "passed": None,
                    "details": {"mode": "metadata_readiness"},
                },
                "metrics": {"quality": {"rubric_passed": 0, "rubric_total": 92}},
            }
        )
        self.assertIsNone(summary["rubric_pass_rate"])
        self.assertIsNone(summary["rubric_passed"])
        self.assertIsNone(summary["rubric_total"])

    def test_task_file_reader_ignores_comments_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.txt"
            path.write_text("# comment\narea/task\n\narea/task-2\n", encoding="utf-8")
            self.assertEqual(read_task_file(path), ["area/task", "area/task-2"])
            self.assertEqual(dedupe(["a", "b", "a"]), ["a", "b"])

    def test_harvey_task_file_does_not_expand_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.txt"
            path.write_text("area/task\narea/task-2\n", encoding="utf-8")
            task_ids = resolve_harvey_smoke_task_ids(
                task_ids=[],
                task_file=path,
                split="sample",
                limit=120,
                harvey_root=None,
            )
            self.assertEqual(task_ids, ["area/task", "area/task-2"])

    def test_harvey_explicit_task_id_does_not_expand_split(self) -> None:
        task_ids = resolve_harvey_smoke_task_ids(
            task_ids=["area/task"],
            task_file=None,
            split="sample",
            limit=120,
            harvey_root=None,
        )
        self.assertEqual(task_ids, ["area/task"])

    def test_harvey_batch_tracking_writes_restart_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = [
                HarveyPipelineResult(
                    task_id="area/pass",
                    trace_path="trace-pass.json",
                    run_id="run\\area--pass",
                    passed=True,
                    rubric_passed=10,
                    rubric_total=10,
                    rubric_pass_rate=1.0,
                    estimated_cost=0.30,
                    total_tokens=3000,
                    token_share_by_tier={"cheap_worker": 0.9},
                    status="completed",
                ),
                HarveyPipelineResult(
                    task_id="area/fail",
                    trace_path="trace-fail.json",
                    run_id="run\\area--fail",
                    passed=False,
                    rubric_passed=8,
                    rubric_total=10,
                    rubric_pass_rate=0.8,
                    estimated_cost=0.50,
                    total_tokens=5000,
                    token_share_by_tier={"cheap_worker": 0.8},
                    status="completed",
                ),
                HarveyPipelineResult(
                    task_id="other/error",
                    trace_path=None,
                    run_id="run\\other--error",
                    passed=False,
                    rubric_passed=None,
                    rubric_total=None,
                    rubric_pass_rate=None,
                    estimated_cost=0.0,
                    total_tokens=0,
                    token_share_by_tier={},
                    error="RuntimeError: boom",
                    status="error",
                ),
            ]
            report = build_harvey_batch_report(
                results,
                requested_task_ids=["area/pass", "area/fail", "other/error"],
                trace_dir=tmp,
                run_prefix="run",
                split="sample",
                limit=3,
            )
            self.assertEqual(report["failed_tasks"], ["area/fail"])
            self.assertEqual(report["error_tasks"], ["other/error"])
            self.assertEqual(report["non_pass_tasks"], ["area/fail", "other/error"])
            self.assertAlmostEqual(report["estimated_cost"], 0.80)
            self.assertAlmostEqual(report["average_cost_per_task"], 0.80 / 3)
            self.assertAlmostEqual(report["median_cost_per_task"], 0.30)
            self.assertEqual(report["max_cost_task"]["task_id"], "area/fail")
            self.assertEqual(report["total_tokens"], 8000)
            self.assertEqual(report["max_token_task"]["task_id"], "area/fail")
            self.assertAlmostEqual(report["by_practice_area"]["area"]["estimated_cost"], 0.80)
            self.assertAlmostEqual(report["results"][0]["cost_per_rubric_criterion"], 0.03)
            paths = write_harvey_batch_tracking(report)
            self.assertTrue(Path(paths["summary"]).exists())
            self.assertEqual(Path(paths["failed_task_file"]).read_text(encoding="utf-8").strip(), "area/fail")
            self.assertEqual(
                Path(paths["non_pass_task_file"]).read_text(encoding="utf-8").splitlines(),
                ["area/fail", "other/error"],
            )

    def test_harvey_resume_reuses_scored_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "harvey_lab_sample" / "area" / "task.json"
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text(
                """{
  "benchmark": "harvey_lab_sample",
  "task_id": "area/task",
  "scoring_result": {"passed": true, "score": 1.0},
  "metrics": {
    "quality": {"rubric_passed": 3, "rubric_total": 3},
    "token_share_by_tier": {"cheap_worker": 0.91}
  }
}
""",
                encoding="utf-8",
            )
            result = maybe_resume_harvey_task(
                task_id="area/task",
                trace_dir=tmp,
                run_prefix="run",
                execute_score=True,
                harvey_root=None,
                judge_model="judge",
                score_parallel=24,
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.status, "resumed")
            self.assertTrue(result.passed)
            self.assertEqual(result.rubric_pass_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
