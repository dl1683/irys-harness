from __future__ import annotations

from pathlib import Path
import base64
from tempfile import TemporaryDirectory
import unittest

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
    parse_paths,
    rerun_from_trace,
    resolve_trace_path,
    safe_upload_filename,
    save_uploaded_files,
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

    def test_save_uploaded_files_sanitizes_and_writes_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = [
                {
                    "filename": "../credit terms.txt",
                    "content_base64": base64.b64encode(b"notice and cure").decode("ascii"),
                },
                {
                    "filename": "../credit terms.txt",
                    "content_base64": base64.b64encode(b"second copy").decode("ascii"),
                },
            ]

            paths = save_uploaded_files(uploads, upload_root=root / "uploads")

            self.assertEqual([path.name for path in paths], ["credit terms.txt", "credit terms-2.txt"])
            self.assertEqual(paths[0].read_text(encoding="utf-8"), "notice and cure")
            self.assertEqual(paths[1].read_text(encoding="utf-8"), "second copy")

    def test_safe_upload_filename_removes_path_and_control_chars(self) -> None:
        self.assertEqual(safe_upload_filename(r"..\folder/bad:name?.txt"), "bad-name-.txt")

    def test_product_ui_renders_answer_markdown_online(self) -> None:
        self.assertIn('class="answer" id="answer"', INDEX_HTML)
        self.assertIn("function renderMarkdown", INDEX_HTML)
        self.assertNotIn('<pre id="answer"></pre>', INDEX_HTML)
        self.assertIn('id="chat"', INDEX_HTML)
        self.assertIn("conversation_history", INDEX_HTML)
        self.assertIn('id="chatHistory"', INDEX_HTML)
        self.assertIn("function renderChatHistory", INDEX_HTML)

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

    def test_rerun_from_trace_can_add_uploaded_corpus(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
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
                    "uploads": [
                        {
                            "filename": "guaranty.txt",
                            "content_base64": base64.b64encode(
                                b"The guarantor cure period is 2 business days after written notice."
                            ).decode("ascii"),
                        }
                    ],
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
