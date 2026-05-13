from __future__ import annotations

import unittest

from irys_harness.benchmarks.harvey import merge_priority_chunks


class HarveyPriorityChunkTests(unittest.TestCase):
    def test_merge_priority_chunks_includes_checklist_file(self) -> None:
        retrieved = [{"chunk_id": "a", "doc_id": "doc1", "text": "already"}]
        chunks = [
            {"chunk_id": "a", "doc_id": "doc1", "text": "already"},
            {"chunk_id": "b", "doc_id": "doc2", "text": "checklist details"},
        ]
        documents = [
            {"doc_id": "doc1", "filename": "memo.docx"},
            {"doc_id": "doc2", "filename": "filing-checklist.docx"},
        ]
        merged = merge_priority_chunks(
            retrieved=retrieved,
            chunks=chunks,
            documents=documents,
            max_chunks=5,
        )
        self.assertEqual([item["chunk_id"] for item in merged], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
