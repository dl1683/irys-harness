from __future__ import annotations

from pathlib import Path
from http.server import ThreadingHTTPServer
import json
import threading
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from irys_harness.config import load_config
from irys_harness.product import (
    build_answer_source_map,
    build_product_synthesis_prompt,
    compare_product_traces,
    discover_corpus_paths,
    run_product_matter,
    sanitize_matter_id,
)
from irys_harness.product_ui import (
    INDEX_HTML,
    build_handler,
    list_product_traces,
    parse_paths,
    pick_local_paths,
    rerun_from_trace,
    resolve_trace_path,
    summarize_trace_rows,
)


class ProductMatterTests(unittest.TestCase):
    def test_discover_corpus_paths_filters_supported_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = root / "contract.txt"
            wanted.write_text("payment covenant notice cure period", encoding="utf-8")
            ignored = root / "archive.bin"
            ignored.write_bytes(b"ignored")

            paths = discover_corpus_paths([str(root)])

        self.assertEqual([path.name for path in paths], ["contract.txt"])

    def test_discover_corpus_paths_default_handles_large_matter_folder(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(90):
                (root / f"doc-{index:03d}.txt").write_text("matter document", encoding="utf-8")

            paths = discover_corpus_paths([str(root)])

        self.assertEqual(len(paths), 90)

    def test_run_product_matter_writes_trace_over_user_corpus(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text(
                "The borrower must deliver quarterly financial statements within 45 days. "
                "A 10 day cure period applies after written notice.",
                encoding="utf-8",
            )
            trace_dir = root / "traces"
            result = run_product_matter(
                objective="What notice and cure period applies to late financial statements?",
                paths=[str(doc)],
                matter_id="Acme Matter",
                config=load_config(),
                trace_dir=trace_dir,
                live_synthesis=False,
                verbose=False,
            )

            trace = result.state.to_trace()
            self.assertTrue(result.trace_path.exists())
            self.assertEqual(trace["benchmark"], "product_matter")
            self.assertEqual(trace["task_id"], "Acme-Matter")
            self.assertEqual(trace["documents"][0]["source_posture"], "user_provided_corpus")
            self.assertTrue(trace["retrieval_iterations"][0]["retrieved_chunks"])
            self.assertIn("10 day cure period", trace["rendered_answer"])
            self.assertEqual(trace["diagnosis"]["status"], "ready_for_review")
            self.assertEqual(trace["diagnosis"]["evidence_count"], 1)
            self.assertGreaterEqual(trace["diagnosis"]["answer_source_map_count"], 1)
            self.assertTrue(Path(trace["artifacts"][0]["path"]).exists())
            self.assertTrue(trace["artifacts"][0]["diagnostic"])
            self.assertEqual(result.to_dict()["artifacts"][0]["filename"], "answer.md")

    def test_run_product_matter_flags_no_matching_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "source.txt"
            doc.write_text("ordinary payment terms only", encoding="utf-8")

            result = run_product_matter(
                objective="zebra orbital indemnity",
                paths=[str(doc)],
                matter_id="No Evidence",
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            trace = result.state.to_trace()

            self.assertEqual(trace["diagnosis"]["status"], "needs_attention")
            self.assertEqual(trace["diagnosis"]["evidence_count"], 0)
            self.assertIn("No chunks matched the objective", trace["final_packet"]["unresolved"][0])

    def test_run_product_matter_does_not_truncate_local_documents_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = "UNTRUNCATED_MARKER_NOTICE_CURE"
            doc = root / "large_agreement.txt"
            doc.write_text(("A" * 210_000) + f" {marker}", encoding="utf-8")

            result = run_product_matter(
                objective=marker,
                paths=[str(doc)],
                matter_id="Large Matter",
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            trace = result.state.to_trace()

            self.assertGreater(trace["documents"][0]["text_chars"], 200_000)
            self.assertIn(marker, trace["rendered_answer"])

    def test_sanitize_matter_id_is_path_safe(self) -> None:
        self.assertEqual(sanitize_matter_id("Client / Matter: 01"), "Client-Matter-01")

    def test_resolve_trace_path_stays_under_trace_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            trace_dir.mkdir()
            trace = trace_dir / "product_matter" / "matter.json"
            trace.parent.mkdir()
            trace.write_text("{}", encoding="utf-8")

            self.assertEqual(resolve_trace_path(str(trace), trace_dir), trace.resolve())
            with self.assertRaises(ValueError):
                resolve_trace_path(str(root / "outside.json"), trace_dir)

    def test_product_ui_renders_answer_markdown_online(self) -> None:
        self.assertIn('class="answer" id="answer"', INDEX_HTML)
        self.assertIn("function renderMarkdown", INDEX_HTML)
        self.assertNotIn('<pre id="answer"></pre>', INDEX_HTML)
        self.assertIn('id="chat"', INDEX_HTML)
        self.assertIn("conversation_history", INDEX_HTML)
        self.assertIn('id="chatHistory"', INDEX_HTML)
        self.assertIn("function renderChatHistory", INDEX_HTML)
        self.assertIn('id="traceList"', INDEX_HTML)
        self.assertIn("/api/traces", INDEX_HTML)
        self.assertIn('id="matterCost"', INDEX_HTML)
        self.assertIn('id="matterMessages"', INDEX_HTML)
        self.assertIn("function pathPayload", INDEX_HTML)
        self.assertIn("Paste local file or folder paths", INDEX_HTML)
        self.assertIn("No artificial file-count or per-document character cap", INDEX_HTML)
        self.assertIn("Top K is the number of retrieved chunks", INDEX_HTML)
        self.assertIn("/api/pick-path", INDEX_HTML)
        self.assertIn('id="chooseFolder"', INDEX_HTML)
        self.assertIn('id="chooseFiles"', INDEX_HTML)
        self.assertIn("function appendPaths", INDEX_HTML)
        self.assertIn('id="liveEvents"', INDEX_HTML)
        self.assertIn("/api/run-async", INDEX_HTML)
        self.assertIn("/api/rerun-async", INDEX_HTML)
        self.assertIn("/api/run-status", INDEX_HTML)
        self.assertIn("/api/cancel-run", INDEX_HTML)
        self.assertIn("function pollRunJob", INDEX_HTML)
        self.assertIn("What Irys Is Doing", INDEX_HTML)
        self.assertIn('id="currentStep"', INDEX_HTML)
        self.assertIn('id="stopRun"', INDEX_HTML)
        self.assertIn("Advanced run details", INDEX_HTML)
        self.assertNotIn("webkitdirectory", INDEX_HTML)
        self.assertNotIn("function uploadFiles", INDEX_HTML)

    def test_pick_local_paths_rejects_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            pick_local_paths(mode="invalid")

    def test_pick_path_endpoint_returns_selected_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            output_dir = root / "outputs"
            handler = build_handler(
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch(
                    "irys_harness.product_ui.pick_local_paths",
                    return_value=[str(root / "selected")],
                ):
                    request = Request(
                        f"http://127.0.0.1:{server.server_port}/api/pick-path",
                        data=json.dumps({"mode": "folder"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=5) as response:  # noqa: S310 - local test server.
                        data = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(data["paths"], [str(root / "selected")])

    def test_async_run_endpoint_streams_live_events_until_completion(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            trace_dir = root / "traces"
            output_dir = root / "outputs"
            handler = build_handler(
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/run-async",
                    data=json.dumps(
                        {
                            "matter_id": "Live Matter",
                            "paths": [str(doc)],
                            "objective": "What cure period applies?",
                            "live_synthesis": False,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:  # noqa: S310 - local test server.
                    started = json.loads(response.read().decode("utf-8"))

                status = started
                for _ in range(20):
                    with urlopen(  # noqa: S310 - local test server.
                        f"http://127.0.0.1:{server.server_port}/api/run-status?job_id={started['job_id']}",
                        timeout=5,
                    ) as response:
                        status = json.loads(response.read().decode("utf-8"))
                    if status["status"] != "running":
                        break
                    time.sleep(0.05)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(status["status"], "completed")
            self.assertIn("5 day cure period", status["result"]["rendered_answer"])
            labels = [event["label"] for event in status["events"]]
            self.assertIn("SCOPE", labels)
            self.assertIn("READ", labels)
            self.assertIn("LOAD", labels)
            self.assertIn("SEARCH", labels)
            self.assertIn("SAVE", labels)
            summaries = [event.get("fields", {}).get("summary", "") for event in status["events"]]
            self.assertTrue(any("Reading document" in summary for summary in summaries))

    def test_list_product_traces_filters_by_matter_and_chat(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            trace_dir = root / "traces"
            run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Matter A",
                chat_id="main",
                config=load_config(),
                trace_dir=trace_dir,
                live_synthesis=False,
                verbose=False,
            )
            run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Matter A",
                chat_id="side",
                config=load_config(),
                trace_dir=trace_dir,
                live_synthesis=False,
                verbose=False,
            )

            all_rows = list_product_traces(trace_dir, matter_id="Matter A")
            side_rows = list_product_traces(trace_dir, matter_id="Matter A", chat_id="side")

            self.assertEqual(len(all_rows), 2)
            self.assertEqual(len(side_rows), 1)
            self.assertEqual(side_rows[0]["chat_id"], "side")
            self.assertTrue(Path(side_rows[0]["path"]).exists())

    def test_summarize_trace_rows_reports_matter_cost(self) -> None:
        summary = summarize_trace_rows(
            [
                {"chat_id": "main", "estimated_cost": 0.25, "total_tokens": 10},
                {"chat_id": "side", "estimated_cost": 0.75, "total_tokens": 30},
            ]
        )

        self.assertEqual(summary["messages"], 2)
        self.assertEqual(summary["chats"], 2)
        self.assertEqual(summary["total_tokens"], 40)
        self.assertAlmostEqual(summary["estimated_cost"], 1.0)
        self.assertAlmostEqual(summary["average_cost_per_message"], 0.5)

    def test_rerun_from_trace_links_parent_and_nudge(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            trace_dir = root / "traces"
            parent = run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Parent Matter",
                config=load_config(),
                trace_dir=trace_dir,
                live_synthesis=False,
                verbose=False,
            )

            response = rerun_from_trace(
                {
                    "trace_path": str(parent.trace_path),
                    "nudge": "Focus only on payment defaults.",
                    "live_synthesis": False,
                    "top_k": 3,
                },
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=root / "outputs",
            )
            trace = response["trace"]

            self.assertTrue(response["trace_path"])
            self.assertIn("User steering note: Focus only on payment defaults.", trace["task"]["question"])
            self.assertEqual(trace["task"]["metadata"]["parent_trace_path"], str(parent.trace_path.resolve()))
            self.assertEqual(trace["task"]["metadata"]["user_nudge"], "Focus only on payment defaults.")
            self.assertEqual(response["comparison"]["parent_task_id"], "Parent-Matter")
            self.assertTrue(response["comparison"]["objective_changed"])
            history = trace["final_packet"]["conversation_history"]
            self.assertEqual(set(history[-1]), {"user", "assistant"})
            self.assertIn("What cure period applies?", history[-1]["user"])
            self.assertIn("5 day cure period", history[-1]["assistant"])

    def test_rerun_from_trace_can_add_local_path_corpus(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            guaranty = root / "guaranty.txt"
            guaranty.write_text(
                "The guarantor cure period is 2 business days after written notice.",
                encoding="utf-8",
            )
            trace_dir = root / "traces"
            output_dir = root / "outputs"
            parent = run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Corpus Matter",
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
                live_synthesis=False,
                verbose=False,
            )

            response = rerun_from_trace(
                {
                    "trace_path": str(parent.trace_path),
                    "nudge": "Also consider the guaranty.",
                    "paths": [str(guaranty)],
                },
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
            )
            trace = response["trace"]

            self.assertEqual(len(trace["task"]["context_files"]), 2)
            self.assertEqual(response["comparison"]["document_delta"]["kept_count"], 1)
            self.assertEqual(len(response["comparison"]["document_delta"]["added"]), 1)

    def test_parse_paths_accepts_textarea_or_list(self) -> None:
        self.assertEqual(parse_paths(" a.txt \n\n b.txt "), ["a.txt", "b.txt"])
        self.assertEqual(parse_paths([" a.txt ", ""]), ["a.txt"])

    def test_conversation_history_is_synthesis_only_not_retrieval_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            result = run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Chat Matter",
                chat_id="follow-up",
                conversation_history=[
                    {
                        "user": "Prior question with NEVER_RETRIEVE_HISTORY_MARKER.",
                        "assistant": "Prior final answer only.",
                        "intermediate": "must not survive",
                    }
                ],
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            trace = result.state.to_trace()
            queries = "\n".join(trace["retrieval_iterations"][0]["queries"])
            history = trace["final_packet"]["conversation_history"]
            prompt = build_product_synthesis_prompt(result.state)

            self.assertEqual(trace["task"]["metadata"]["chat_id"], "follow-up")
            self.assertIn("--chat-follow-up", trace["task_id"])
            self.assertEqual(history, [{"user": "Prior question with NEVER_RETRIEVE_HISTORY_MARKER.", "assistant": "Prior final answer only."}])
            self.assertNotIn("NEVER_RETRIEVE_HISTORY_MARKER", queries)
            self.assertFalse(trace["retrieval_iterations"][0]["conversation_history_used"])
            self.assertIn("NEVER_RETRIEVE_HISTORY_MARKER", prompt)

    def test_history_dependent_followup_is_flagged_without_using_history_for_retrieval(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Guaranty enforcement requires a continuing payment default.", encoding="utf-8")
            result = run_product_matter(
                objective="What about that?",
                paths=[str(doc)],
                matter_id="Followup Matter",
                conversation_history=[{"user": "Can lender accelerate?", "assistant": "Only after default."}],
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            trace = result.state.to_trace()

            self.assertIn("may depend on prior chat context", trace["final_packet"]["unresolved"][0])
            self.assertFalse(trace["retrieval_iterations"][0]["conversation_history_used"])

    def test_product_synthesis_prompt_includes_worker_analysis(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            result = run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Analysis Matter",
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            result.state.final_packet["worker_analysis"] = "MATERIAL_FACTS: payment default cure is 5 days."

            prompt = build_product_synthesis_prompt(result.state)

            self.assertIn("Worker analysis:", prompt)
            self.assertIn("MATERIAL_FACTS: payment default cure is 5 days.", prompt)

    def test_build_answer_source_map_links_answer_sections_to_evidence(self) -> None:
        evidence = [
            {
                "raw_support": "The payment default cure period is 5 business days after notice.",
                "source": {"doc_id": "doc_0001", "chunk_id": "doc_0001_chunk_0000"},
            }
        ]
        answer = "Payment default has a 5 business day cure period (doc_0001_chunk_0000)."

        source_map = build_answer_source_map(answer, evidence)

        self.assertEqual(len(source_map), 1)
        self.assertEqual(source_map[0]["source_refs"], ["doc_0001_chunk_0000"])
        self.assertEqual(source_map[0]["support"][0]["doc_id"], "doc_0001")

    def test_compare_product_traces_reports_evidence_and_metric_deltas(self) -> None:
        parent = {
            "run_id": "parent",
            "task_id": "matter",
            "task": {"question": "Question"},
            "documents": [{"path": "a.txt"}],
            "final_packet": {
                "verified_evidence": [
                    {
                        "claim": "Old claim",
                        "raw_support": "Old support",
                        "source": {"doc_id": "doc_1", "chunk_id": "chunk_1"},
                    }
                ],
                "unresolved": ["old gap"],
            },
            "rendered_answer": "Old answer",
            "metrics": {"total_tokens": 10, "estimated_cost": 0.25},
        }
        child = {
            "run_id": "child",
            "task_id": "matter-nudge",
            "task": {"question": "Question\n\nUser steering note: focus"},
            "documents": [{"path": "a.txt"}, {"path": "b.txt"}],
            "final_packet": {
                "verified_evidence": [
                    {
                        "claim": "New claim",
                        "raw_support": "New support",
                        "source": {"doc_id": "doc_2", "chunk_id": "chunk_2"},
                    }
                ],
                "unresolved": ["new gap"],
            },
            "rendered_answer": "New answer",
            "metrics": {"total_tokens": 17, "estimated_cost": 0.5},
        }

        comparison = compare_product_traces(parent, child)

        self.assertTrue(comparison["objective_changed"])
        self.assertTrue(comparison["answer_changed"])
        self.assertEqual(comparison["document_delta"]["added"], ["b.txt"])
        self.assertEqual(comparison["evidence_delta"]["added_count"], 1)
        self.assertEqual(comparison["evidence_delta"]["removed_count"], 1)
        self.assertEqual(comparison["unresolved_delta"]["added"], ["new gap"])
        self.assertEqual(comparison["metrics_delta"]["total_tokens"], 7)
        self.assertAlmostEqual(comparison["metrics_delta"]["estimated_cost"], 0.25)


if __name__ == "__main__":
    unittest.main()
