from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from docx import Document
from openpyxl import load_workbook

from irys_harness.artifacts import render_deliverables


class ArtifactRenderingTests(unittest.TestCase):
    def test_xlsx_renderer_uses_deliverable_contract_sheets(self) -> None:
        packet = {
            "deliverable_contract": {
                "deliverables": [
                    {
                        "filename": "section-382-analysis-workbook.xlsx",
                        "workbook_sheets": [
                            {"name": "Shareholder Register", "purpose": "Owner register"},
                            {"name": "Ownership Shift Calculations", "purpose": "Shift math"},
                            {"name": "Section 382 Limit Computation", "purpose": "Limit math"},
                        ],
                    }
                ]
            },
            "draft_answer": """
| Shareholder | Date | Percentage |
| --- | --- | --- |
| Alpha Fund | 2024-01-01 | 6.2% |
""",
            "verified_evidence": [
                {
                    "claim": "Alpha Fund exceeded five percent.",
                    "raw_support": "Alpha Fund held 6.2 percent.",
                    "source": {"doc_id": "doc_1", "chunk_id": "chunk_1"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = render_deliverables(
                output_dir=tmp,
                deliverables=["section-382-analysis-workbook.xlsx"],
                title="Section 382",
                packet=packet,
            )
            path = Path(artifacts[0]["path"])
            workbook = load_workbook(path, read_only=True)
            try:
                self.assertIn("Shareholder Register", workbook.sheetnames)
                self.assertIn("Ownership Shift Calculations", workbook.sheetnames)
                self.assertIn("Section 382 Limit Computation", workbook.sheetnames)
                self.assertIn("Evidence", workbook.sheetnames)
                self.assertGreater(workbook["Shareholder Register"].max_row, 4)
            finally:
                workbook.close()

    def test_docx_renderer_replaces_encoded_artifact_with_readable_fallback(self) -> None:
        packet = {
            "draft_answer": "```xml\n<base64_file><filename>memo.docx</filename><content>"
            + ("A" * 5000)
            + "</content></base64_file>\n```",
            "cheap_worker_summary": "Class 1 interest underpayment and Class 6 cash pool issue.",
            "verified_evidence": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = render_deliverables(
                output_dir=tmp,
                deliverables=["memo.docx"],
                title="Memo",
                packet=packet,
            )
            doc = Document(Path(artifacts[0]["path"]))
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            self.assertIn("Class 1 interest underpayment", text)
            self.assertNotIn("<base64_file>", text)

    def test_docx_renderer_preserves_structured_appendix(self) -> None:
        packet = {
            "draft_answer": "Clean memo narrative.",
            "artifact_appendix": "Score-critical row: 2023 Merger Guidelines threshold.",
            "verified_evidence": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = render_deliverables(
                output_dir=tmp,
                deliverables=["memo.docx"],
                title="Memo",
                packet=packet,
            )
            doc = Document(Path(artifacts[0]["path"]))
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            self.assertIn("Structured Findings Appendix", text)
            self.assertIn("2023 Merger Guidelines threshold", text)


if __name__ == "__main__":
    unittest.main()
