from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    claim: str
    raw_support: str
    doc_id: str
    chunk_id: str
    confidence: str
    directness: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "raw_support": self.raw_support,
            "source": {"doc_id": self.doc_id, "chunk_id": self.chunk_id},
            "confidence": self.confidence,
            "directness": self.directness,
        }


def extract_evidence_from_retrieval(
    retrieved_chunks: list[dict[str, Any]],
    *,
    max_items: int = 8,
    support_chars: int = 800,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for chunk in retrieved_chunks[:max_items]:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        support = text[:support_chars]
        claim = summarize_support(support)
        items.append(
            EvidenceItem(
                claim=claim,
                raw_support=support,
                doc_id=str(chunk.get("doc_id")),
                chunk_id=str(chunk.get("chunk_id")),
                confidence="medium",
                directness="candidate",
            )
        )
    return items


def summarize_support(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        first_line = text.strip()
    return first_line[:240]


def build_final_packet(
    *,
    question: str,
    deliverables: list[str],
    evidence_items: list[EvidenceItem],
    unresolved: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "deliverables": deliverables,
        "verified_evidence": [item.to_dict() for item in evidence_items],
        "unresolved": unresolved or [],
    }

