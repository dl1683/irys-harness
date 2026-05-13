from __future__ import annotations

import unittest

from irys_harness.agents import build_final_packet, extract_evidence_from_retrieval


class EvidenceTests(unittest.TestCase):
    def test_extract_evidence_from_retrieved_chunks(self) -> None:
        items = extract_evidence_from_retrieval(
            [
                {
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "text": "Important fact.\nMore support.",
                }
            ]
        )
        self.assertEqual(items[0].claim, "Important fact.")
        self.assertEqual(items[0].doc_id, "doc1")

    def test_build_final_packet(self) -> None:
        items = extract_evidence_from_retrieval(
            [{"doc_id": "doc1", "chunk_id": "chunk1", "text": "Fact."}]
        )
        packet = build_final_packet(question="Q", deliverables=["memo.docx"], evidence_items=items)
        self.assertEqual(packet["deliverables"], ["memo.docx"])
        self.assertEqual(len(packet["verified_evidence"]), 1)


if __name__ == "__main__":
    unittest.main()
