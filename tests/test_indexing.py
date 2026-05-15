from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from irys_harness.indexing import chunk_text, load_documents, normalize_extracted_text


class IndexingTests(unittest.TestCase):
    def test_chunk_text_links_neighbors(self) -> None:
        chunks = chunk_text("abcdefghij", doc_id="doc", chunk_size_chars=4, chunk_overlap_chars=1)
        self.assertEqual([chunk.text for chunk in chunks], ["abcd", "defg", "ghij"])
        self.assertIsNone(chunks[0].prev_chunk)
        self.assertEqual(chunks[0].next_chunk, chunks[1].chunk_id)
        self.assertEqual(chunks[1].prev_chunk, chunks[0].chunk_id)

    def test_load_documents_reads_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.txt"
            path.write_text("hello world", encoding="utf-8")
            docs, chunks = load_documents([str(path)], chunk_size_chars=20, chunk_overlap_chars=0)
            self.assertEqual(docs[0]["filename"], "source.txt")
            self.assertEqual(docs[0]["text_chars"], 11)
            self.assertEqual(chunks[0]["text"], "hello world")

    def test_load_documents_renders_xlsx_rows_without_nan_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Covenants"
            sheet.append(["Metric", "Value", "Note"])
            sheet.append(["EBITDA", 20790000, None])
            workbook.save(path)
            workbook.close()
            docs, chunks = load_documents([str(path)], chunk_size_chars=2000, chunk_overlap_chars=0)
            self.assertEqual(docs[0]["filename"], "source.xlsx")
            text = chunks[0]["text"]
            self.assertIn("## Sheet: Covenants", text)
            self.assertIn("row 2: EBITDA | 20790000", text)
            self.assertNotIn("NaN", text)

    def test_normalize_extracted_text_repairs_common_pdf_glyphs(self) -> None:
        text = normalize_extracted_text(
            "Net income per share - diluted $ \ue534.\ue536\ue538 and free cash \ue38dow e\ue389ects"
        )
        self.assertEqual(text, "Net income per share - diluted $ 0.13 and free cash flow effects")


if __name__ == "__main__":
    unittest.main()
