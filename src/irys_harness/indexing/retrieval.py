from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "text": self.text,
        }


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def retrieve_chunks(
    chunks: list[dict[str, Any]],
    queries: list[str],
    *,
    top_k: int = 12,
) -> list[RetrievedChunk]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    query_terms = set(tokenize(" ".join(queries)))
    if not query_terms:
        return []

    doc_freq: dict[str, int] = {}
    chunk_terms: list[tuple[dict[str, Any], list[str]]] = []
    for chunk in chunks:
        terms = tokenize(str(chunk.get("text", "")))
        chunk_terms.append((chunk, terms))
        for term in set(terms):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    n_chunks = max(1, len(chunks))
    results: list[RetrievedChunk] = []
    for chunk, terms in chunk_terms:
        if not terms:
            continue
        term_counts: dict[str, int] = {}
        for term in terms:
            if term in query_terms:
                term_counts[term] = term_counts.get(term, 0) + 1
        if not term_counts:
            continue
        score = 0.0
        length_norm = math.sqrt(len(terms))
        for term, count in term_counts.items():
            idf = math.log((n_chunks + 1) / (doc_freq.get(term, 0) + 1)) + 1
            score += (count / length_norm) * idf
        results.append(
            RetrievedChunk(
                chunk_id=str(chunk["chunk_id"]),
                doc_id=str(chunk["doc_id"]),
                score=score,
                text=str(chunk.get("text", "")),
            )
        )

    return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

