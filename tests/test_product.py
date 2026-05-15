from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from irys_harness.config import load_config
from irys_harness.product import discover_corpus_paths, run_product_matter, sanitize_matter_id
from irys_harness.product_ui import resolve_trace_path


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


if __name__ == "__main__":
    unittest.main()
