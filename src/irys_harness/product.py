from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .config import HarnessConfig
from .events import EventLogger
from .indexing import load_documents, retrieve_chunks, tokenize
from .models.gemini import GeminiModelRouter
from .state import BenchmarkTask, RunState
from .trace import TraceWriter


SUPPORTED_PRODUCT_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".eml",
    ".html",
    ".htm",
    ".json",
    ".md",
    ".pdf",
    ".pptx",
    ".txt",
    ".xlsm",
    ".xlsx",
}


@dataclass(frozen=True)
class ProductRunResult:
    state: RunState
    trace_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": str(self.trace_path),
            "run_id": self.state.run_id,
            "matter_id": self.state.task.task_id,
            "rendered_answer": self.state.rendered_answer,
            "documents": self.state.documents,
            "events": self.state.events,
            "metrics": self.state.metrics.to_dict(),
        }


def run_product_matter(
    *,
    objective: str,
    paths: list[str],
    config: HarnessConfig,
    matter_id: str = "matter",
    trace_dir: str | Path = "traces/product",
    output_dir: str | Path = "outputs/product",
    live_synthesis: bool = False,
    top_k: int = 12,
    max_files: int = 80,
    max_chars_per_doc: int | None = 200_000,
    verbose: bool = True,
) -> ProductRunResult:
    if not objective.strip():
        raise ValueError("objective is required")
    corpus_paths = discover_corpus_paths(paths, max_files=max_files)
    if not corpus_paths:
        raise ValueError("at least one supported document path is required")

    task_id = sanitize_matter_id(matter_id)
    task = BenchmarkTask(
        benchmark="product_matter",
        task_id=task_id,
        question=objective.strip(),
        context_files=[str(path) for path in corpus_paths],
        answer_schema={
            "type": "product_legal_work_product",
            "requires_citations": True,
            "source": "user_defined_corpus",
        },
        metadata={
            "matter_id": matter_id,
            "document_boundary": "user_defined_corpus",
            "live_synthesis": live_synthesis,
        },
    )
    state = RunState(task=task, config=config, output_dir=str(Path(output_dir) / task_id))
    log = EventLogger(state, verbose=verbose)
    log.emit("RUN", "started product matter run", matter=task_id)

    state.documents, state.chunks = load_documents(
        [str(path) for path in corpus_paths],
        max_chars_per_doc=max_chars_per_doc,
        source_posture="user_provided_corpus",
    )
    log.emit("LOAD", "loaded user corpus", documents=len(state.documents), chunks=len(state.chunks))

    contract = build_product_contract(state, top_k=top_k)
    state.answer_contract_versions.append(contract)
    log.emit("CONTRACT", "built matter contract", needed=len(contract["needed_information"]))

    queries = build_product_queries(objective, state.documents)
    retrieved = retrieve_chunks(state.chunks, queries, top_k=top_k)
    state.retrieval_iterations.append(
        {
            "iteration": 1,
            "queries": queries,
            "retrieved_chunks": [item.to_dict() for item in retrieved],
            "reason": "Product matter retrieval from user objective, filenames, and corpus inventory.",
        }
    )
    log.emit("SEARCH", "retrieved candidate chunks", chunks=len(retrieved), top_k=top_k)

    evidence_items = build_product_evidence_items(retrieved)
    state.extraction_records.append(
        {
            "mode": "deterministic_product_snippet_extraction",
            "evidence_items": evidence_items,
        }
    )
    state.verification_records.append(
        {
            "mode": "source_chunk_presence_check",
            "accepted": [item["claim"] for item in evidence_items],
            "rejected": [],
            "weak": [],
        }
    )
    log.emit("EXTRACT", "built source evidence packet", evidence=len(evidence_items))

    state.final_packet = {
        "mode": "product_user_corpus_packet",
        "question": objective.strip(),
        "document_boundary": "user_defined_corpus",
        "documents": state.documents,
        "retrieved_chunks": [item.to_dict() for item in retrieved],
        "verified_evidence": evidence_items,
        "unresolved": build_unresolved_items(state),
    }
    state.diagnosis = build_product_diagnosis(state)

    if live_synthesis:
        router = GeminiModelRouter(config)
        prompt = build_product_synthesis_prompt(state)
        result = router.generate(
            module="synthesis",
            prompt=prompt,
            temperature=0.0,
            max_output_tokens=8192,
        )
        state.metrics.add_call(result.usage)
        state.draft_answer = result.text
        state.rendered_answer = result.text
        log.emit("SYNTH", "generated live answer", model=result.usage.model)
    else:
        state.draft_answer = build_deterministic_product_answer(state)
        state.rendered_answer = state.draft_answer
        log.emit("SYNTH", "generated deterministic preview")

    trace_path = TraceWriter(trace_dir).write(state)
    log.emit("SAVE", "trace saved", trace=str(trace_path))
    return ProductRunResult(state=state, trace_path=trace_path)


def discover_corpus_paths(paths: list[str], *, max_files: int = 80) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        if not str(raw).strip():
            continue
        path = Path(str(raw).strip()).expanduser()
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in SUPPORTED_PRODUCT_EXTENSIONS:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)
            if len(discovered) >= max_files:
                return discovered
    return discovered


def sanitize_matter_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip() or "matter").strip("-")
    return cleaned[:80] or "matter"


def build_product_contract(state: RunState, *, top_k: int) -> dict[str, Any]:
    return {
        "version": 1,
        "interpreted_goal": state.task.question,
        "required_output_format": "clean product answer or draft work product",
        "document_boundary": "user_defined_corpus",
        "needed_information": [
            "user objective",
            "active corpus inventory",
            "most relevant source chunks",
            "source-grounded facts and gaps",
            "clean answer or artifact draft",
        ],
        "search_queries": build_product_queries(state.task.question, state.documents),
        "verification_requirements": [
            "Use only user-provided corpus unless external tools are explicitly enabled.",
            "Expose which documents and chunks were used.",
            "Mark gaps instead of inventing facts.",
        ],
        "scoring_risks": [
            "Product run has no benchmark gold answer.",
            "Quality must be inspected through trace, evidence packet, and user review.",
        ],
        "retrieval": {"top_k": top_k},
    }


def build_product_queries(objective: str, documents: list[dict[str, Any]]) -> list[str]:
    filename_terms = " ".join(str(doc.get("filename", "")) for doc in documents)
    objective_terms = " ".join(tokenize(objective))
    return [objective, objective_terms, filename_terms]


def build_product_evidence_items(retrieved: list[Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(retrieved, 1):
        snippet = compact_snippet(item.text)
        evidence.append(
            {
                "claim": f"Relevant source excerpt {index} from {item.doc_id}.",
                "raw_support": snippet,
                "source": {
                    "doc_id": item.doc_id,
                    "chunk_id": item.chunk_id,
                    "score": item.score,
                },
                "confidence": "medium",
                "directness": "direct",
            }
        )
    return evidence


def compact_snippet(text: str, *, max_chars: int = 900) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def build_unresolved_items(state: RunState) -> list[str]:
    items: list[str] = []
    if any(doc.get("load_error") for doc in state.documents):
        items.append("One or more documents failed to load; inspect document inventory.")
    if not state.final_packet and not state.chunks:
        items.append("No searchable text was extracted from the active corpus.")
    if state.chunks and not state.retrieval_iterations:
        items.append("Retrieval has not run.")
    if state.chunks and state.retrieval_iterations:
        retrieved = state.retrieval_iterations[-1].get("retrieved_chunks", [])
        if not retrieved:
            items.append("No chunks matched the objective; revise the objective or add relevant documents.")
    return items


def build_product_diagnosis(state: RunState) -> dict[str, Any]:
    packet = state.final_packet or {}
    retrieved = []
    if state.retrieval_iterations:
        retrieved = list(state.retrieval_iterations[-1].get("retrieved_chunks", []) or [])
    evidence = list(packet.get("verified_evidence", []) or [])
    unresolved = list(packet.get("unresolved", []) or [])
    load_errors = [doc for doc in state.documents if doc.get("load_error")]
    status = "ready_for_review"
    if load_errors or not state.chunks or not evidence or unresolved:
        status = "needs_attention"
    return {
        "mode": "product_matter_diagnosis",
        "status": status,
        "document_count": len(state.documents),
        "load_error_count": len(load_errors),
        "chunk_count": len(state.chunks),
        "retrieved_chunk_count": len(retrieved),
        "evidence_count": len(evidence),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "supporting_trace_refs": [
            {"path": "documents", "reason": "Active user corpus and load errors."},
            {"path": "retrieval_iterations[-1].retrieved_chunks", "reason": "Chunks selected from the corpus."},
            {"path": "final_packet.verified_evidence", "reason": "Evidence passed to synthesis or preview."},
            {"path": "final_packet.unresolved", "reason": "Gaps that should be resolved before relying on output."},
        ],
    }


def build_product_synthesis_prompt(state: RunState) -> str:
    packet = state.final_packet or {}
    return f"""Produce the requested legal work product or answer using only the user-provided corpus.

User objective:
{state.task.question}

Active corpus:
{format_documents_for_prompt(state.documents)}

Evidence packet:
{format_evidence_for_prompt(packet.get("verified_evidence", []))}

Unresolved gaps:
{format_list(packet.get("unresolved", []))}

Write a clean answer. Cite source doc IDs and chunk IDs when relying on evidence. If the corpus does not contain enough support, state the gap plainly."""


def build_deterministic_product_answer(state: RunState) -> str:
    packet = state.final_packet or {}
    lines = [
        "# Product Matter Preview",
        "",
        "## Objective",
        state.task.question,
        "",
        "## Active Corpus",
        *[
            f"- {doc.get('doc_id')}: {doc.get('filename')} ({doc.get('text_chars', 0)} chars)"
            for doc in state.documents
        ],
        "",
        "## Retrieved Evidence",
    ]
    evidence = packet.get("verified_evidence", [])
    if evidence:
        for item in evidence:
            source = item.get("source", {})
            lines.extend(
                [
                    f"- {source.get('doc_id')} / {source.get('chunk_id')}: {item.get('raw_support')}",
                ]
            )
    else:
        lines.append("- No matching evidence was retrieved.")
    gaps = packet.get("unresolved", [])
    if gaps:
        lines.extend(["", "## Open Gaps", *[f"- {gap}" for gap in gaps]])
    return "\n".join(lines)


def format_documents_for_prompt(documents: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {doc.get('doc_id')}: {doc.get('filename')} path={doc.get('path')} chars={doc.get('text_chars')}"
        for doc in documents
    )


def format_evidence_for_prompt(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "- No evidence retrieved."
    parts = []
    for item in evidence:
        source = item.get("source", {})
        parts.append(
            "\n".join(
                [
                    f"- Source: {source.get('doc_id')} / {source.get('chunk_id')} score={source.get('score')}",
                    f"  Support: {item.get('raw_support')}",
                ]
            )
        )
    return "\n".join(parts)


def format_list(items: list[str]) -> str:
    if not items:
        return "- None."
    return "\n".join(f"- {item}" for item in items)
