from __future__ import annotations

import unittest

from irys_harness.indexing import retrieve_chunks, tokenize


class RetrievalTests(unittest.TestCase):
    def test_tokenize_normalizes_words(self) -> None:
        self.assertEqual(tokenize("HSR filing, market-share"), ["hsr", "filing", "market-share"])

    def test_retrieve_chunks_ranks_matching_chunk(self) -> None:
        chunks = [
            {"chunk_id": "a", "doc_id": "doc1", "text": "unrelated tax memo"},
            {"chunk_id": "b", "doc_id": "doc2", "text": "HSR filing antitrust market share"},
        ]
        results = retrieve_chunks(chunks, ["antitrust HSR"], top_k=1)
        self.assertEqual(results[0].chunk_id, "b")


if __name__ == "__main__":
    unittest.main()

