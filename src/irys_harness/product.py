from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Callable

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

DEFAULT_PRODUCT_MAX_FILES: int | None = None


class ProductRunCancelled(RuntimeError):
    pass


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
            "artifacts": self.state.artifacts,
            "metrics": self.state.metrics.to_dict(),
        }


def run_product_matter(
    *,
    objective: str,
    paths: list[str],
    config: HarnessConfig,
    matter_id: str = "matter",
    chat_id: str = "main",
    conversation_history: list[dict[str, Any]] | None = None,
    trace_dir: str | Path = "traces/product",
    output_dir: str | Path = "outputs/product",
    live_synthesis: bool = False,
    top_k: int = 12,
    max_files: int | None = DEFAULT_PRODUCT_MAX_FILES,
    max_chars_per_doc: int | None = None,
    verbose: bool = True,
    parent_trace_path: str | None = None,
    user_nudge: str | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ProductRunResult:
    if not objective.strip():
        raise ValueError("objective is required")
    pre_events: list[dict[str, Any]] = []
    emit_pre_event(
        pre_events,
        event_callback,
        "SCOPE",
        "Checking selected corpus",
        summary="Checking your selected files and folders.",
        selected_paths=[str(path) for path in paths],
        next_step="Find supported documents to read.",
    )
    check_product_cancel(should_cancel)
    corpus_paths = discover_corpus_paths(paths, max_files=max_files)
    if not corpus_paths:
        raise ValueError("at least one supported document path is required")
    emit_pre_event(
        pre_events,
        event_callback,
        "SCOPE",
        "Found supported documents",
        summary=f"Found {len(corpus_paths)} supported document(s).",
        document_count=len(corpus_paths),
        sample_documents=[path.name for path in corpus_paths[:12]],
        omitted_document_count=max(0, len(corpus_paths) - 12),
        next_step="Read documents and extract searchable text.",
    )
    check_product_cancel(should_cancel)

    normalized_chat_id = sanitize_chat_id(chat_id)
    normalized_history = normalize_conversation_history(conversation_history)
    task_id = build_product_task_id(matter_id, normalized_chat_id)
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
            "chat_id": normalized_chat_id,
            "document_boundary": "user_defined_corpus",
            "live_synthesis": live_synthesis,
            "parent_trace_path": parent_trace_path,
            "user_nudge": user_nudge,
            "conversation_history_turns": len(normalized_history),
            "conversation_history_policy": "synthesis_only_user_question_and_final_answer",
        },
    )
    state = RunState(task=task, config=config, output_dir=str(Path(output_dir) / task_id))
    state.events.extend(pre_events)
    log = EventLogger(state, verbose=verbose, event_callback=event_callback)
    log.emit(
        "RUN",
        "started product matter run",
        matter=task_id,
        summary=f"Started matter run for {matter_id}.",
        next_step="Read the selected corpus.",
    )

    state.documents, state.chunks = load_documents(
        [str(path) for path in corpus_paths],
        max_chars_per_doc=max_chars_per_doc,
        source_posture="user_provided_corpus",
        progress_callback=lambda event: log.emit(
            str(event.get("label") or "READ"),
            str(event.get("message") or ""),
            **dict(event.get("fields") or {}),
        ),
        should_cancel=should_cancel,
    )
    check_product_cancel(should_cancel)
    log.emit(
        "LOAD",
        "loaded user corpus",
        documents=len(state.documents),
        chunks=len(state.chunks),
        summary=f"Loaded {len(state.documents)} document(s) into {len(state.chunks)} searchable chunk(s).",
        next_step="Build the answer target and decide what evidence to search for.",
    )

    contract = build_product_contract(state, top_k=top_k)
    state.answer_contract_versions.append(contract)
    log.emit(
        "PLAN",
        "built matter contract",
        needed=len(contract["needed_information"]),
        summary="Clarified what the answer needs to establish.",
        needed_information=contract["needed_information"],
        search_queries=contract["search_queries"],
        next_step="Search the loaded corpus for source support.",
    )

    queries = build_product_queries(objective, state.documents)
    log.emit(
        "SEARCH",
        "searching corpus",
        summary="Searching the corpus for relevant passages.",
        queries=queries,
        top_k=top_k,
        steer_hint="If these search targets miss the point, stop and rephrase the question or add a steering note.",
    )
    check_product_cancel(should_cancel)
    retrieved = retrieve_chunks(state.chunks, queries, top_k=top_k)
    state.retrieval_iterations.append(
        {
            "iteration": 1,
            "queries": queries,
            "retrieved_chunks": [item.to_dict() for item in retrieved],
            "reason": "Product matter retrieval from current objective, filenames, and corpus inventory. Conversation history is intentionally excluded from retrieval queries.",
            "conversation_history_used": False,
        }
    )
    log.emit(
        "SEARCH",
        "retrieved candidate chunks",
        chunks=len(retrieved),
        top_k=top_k,
        summary=f"Selected {len(retrieved)} candidate passage(s) for evidence review.",
        selected_sources=summarize_retrieved_sources(retrieved, state.documents),
        next_step="Extract usable evidence from the selected passages.",
    )

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
    log.emit(
        "EVIDENCE",
        "built source evidence packet",
        evidence=len(evidence_items),
        summary=f"Built an evidence packet with {len(evidence_items)} item(s).",
        evidence_preview=summarize_evidence_items(evidence_items),
        next_step="Draft the answer from the evidence packet.",
    )

    state.final_packet = {
        "mode": "product_user_corpus_packet",
        "question": objective.strip(),
        "chat_id": normalized_chat_id,
        "conversation_history": normalized_history,
        "conversation_history_policy": "synthesis_only_user_question_and_final_answer",
        "document_boundary": "user_defined_corpus",
        "documents": state.documents,
        "retrieved_chunks": [item.to_dict() for item in retrieved],
        "verified_evidence": evidence_items,
        "unresolved": build_unresolved_items(state),
    }

    if live_synthesis:
        check_product_cancel(should_cancel)
        router = GeminiModelRouter(config)
        log.emit(
            "ANALYZE",
            "asking worker model to organize evidence",
            summary="Asking a worker model to organize the evidence for drafting.",
            next_step="Use those notes for the final answer.",
        )
        analysis_prompt = build_product_worker_analysis_prompt(state)
        analysis = router.generate(
            module="extraction",
            prompt=analysis_prompt,
            temperature=0.0,
            max_output_tokens=4096,
        )
        state.metrics.add_call(analysis.usage)
        state.extraction_records.append(
            {
                "mode": "live_product_worker_analysis",
                "model": analysis.usage.model,
                "analysis": analysis.text,
            }
        )
        state.final_packet["worker_analysis"] = analysis.text
        log.emit(
            "ANALYZE",
            "generated worker analysis",
            model=analysis.usage.model,
            summary="Worker analysis is ready.",
            analysis_preview=compact_compare_text(analysis.text, max_chars=900),
            next_step="Draft the final answer.",
        )
        check_product_cancel(should_cancel)
        prompt = build_product_synthesis_prompt(state)
        log.emit(
            "SYNTH",
            "drafting final answer",
            summary="Drafting the final answer from the evidence packet.",
            next_step="Save the answer and source trace.",
        )
        result = router.generate(
            module="synthesis",
            prompt=prompt,
            temperature=0.0,
            max_output_tokens=8192,
        )
        state.metrics.add_call(result.usage)
        state.draft_answer = result.text
        state.rendered_answer = result.text
        log.emit(
            "SYNTH",
            "generated live answer",
            model=result.usage.model,
            summary="Final answer generated.",
            answer_preview=compact_compare_text(result.text, max_chars=900),
        )
    else:
        state.draft_answer = build_deterministic_product_answer(state)
        state.rendered_answer = state.draft_answer
        log.emit(
            "SYNTH",
            "generated deterministic preview",
            summary="Generated a deterministic preview without model synthesis.",
            answer_preview=compact_compare_text(state.rendered_answer, max_chars=900),
        )

    state.final_packet["answer_source_map"] = build_answer_source_map(
        state.rendered_answer or "",
        state.final_packet.get("verified_evidence", []),
    )
    state.artifacts.append(write_product_answer_artifact(state, diagnostic=not live_synthesis))
    state.diagnosis = build_product_diagnosis(state)
    trace_path = TraceWriter(trace_dir).write(state)
    log.emit(
        "SAVE",
        "trace saved",
        trace=str(trace_path),
        summary="Saved the answer, evidence packet, and diagnostic trace.",
    )
    return ProductRunResult(state=state, trace_path=trace_path)


def emit_pre_event(
    pre_events: list[dict[str, Any]],
    event_callback: Callable[[dict[str, Any]], None] | None,
    label: str,
    message: str,
    **fields: Any,
) -> None:
    event = {
        "ts": datetime.now(UTC).isoformat(),
        "label": label,
        "message": message,
        "fields": fields,
    }
    pre_events.append(event)
    if event_callback:
        event_callback(event)


def check_product_cancel(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel and should_cancel():
        raise ProductRunCancelled("run stopped by user")


def summarize_retrieved_sources(retrieved: list[Any], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filenames_by_doc_id = {str(doc.get("doc_id")): str(doc.get("filename") or "") for doc in documents}
    rows = []
    for item in retrieved[:8]:
        rows.append(
            {
                "document": filenames_by_doc_id.get(item.doc_id, item.doc_id),
                "chunk_id": item.chunk_id,
                "score": round(float(item.score), 4),
                "preview": compact_compare_text(item.text, max_chars=220),
            }
        )
    return rows


def summarize_evidence_items(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in evidence_items[:8]:
        source = item.get("source") or {}
        rows.append(
            {
                "source": source.get("chunk_id") or source.get("doc_id"),
                "claim": compact_compare_text(item.get("claim"), max_chars=180),
                "support": compact_compare_text(item.get("raw_support"), max_chars=260),
            }
        )
    return rows


def discover_corpus_paths(paths: list[str], *, max_files: int | None = DEFAULT_PRODUCT_MAX_FILES) -> list[Path]:
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
            if max_files is not None and len(discovered) >= max_files:
                return discovered
    return discovered


def sanitize_matter_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip() or "matter").strip("-")
    return cleaned[:80] or "matter"


def sanitize_chat_id(value: str) -> str:
    return sanitize_matter_id(value or "main")


def build_product_task_id(matter_id: str, chat_id: str) -> str:
    matter = sanitize_matter_id(matter_id)
    chat = sanitize_chat_id(chat_id)
    if chat == "main":
        return matter
    return sanitize_matter_id(f"{matter}--chat-{chat}")


def normalize_conversation_history(raw_history: list[dict[str, Any]] | None, *, max_turns: int = 12) -> list[dict[str, str]]:
    if raw_history is None:
        return []
    if not isinstance(raw_history, list):
        raise ValueError("conversation_history must be a list")
    normalized: list[dict[str, str]] = []
    for item in raw_history[-max_turns:]:
        if not isinstance(item, dict):
            continue
        user = compact_history_text(
            item.get("user")
            or item.get("question")
            or item.get("objective")
            or item.get("user_question")
            or ""
        )
        assistant = compact_history_text(
            item.get("assistant")
            or item.get("answer")
            or item.get("final_answer")
            or item.get("rendered_answer")
            or ""
        )
        if user or assistant:
            normalized.append({"user": user, "assistant": assistant})
    return normalized


def compact_history_text(value: Any, *, max_chars: int = 6000) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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
            "Conversation history may inform synthesis, but must not be used as retrieval context.",
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
    if (
        state.task.metadata.get("conversation_history_turns")
        and objective_may_depend_on_history(state.task.question)
    ):
        items.append(
            "The current objective may depend on prior chat context. Conversation history will inform synthesis, but retrieval uses only the current objective and active corpus; restate the target issue if retrieval looks thin."
        )
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


def objective_may_depend_on_history(objective: str) -> bool:
    tokens = tokenize(objective)
    if len(tokens) <= 4:
        return True
    history_dependent_terms = {
        "above",
        "earlier",
        "it",
        "previous",
        "same",
        "that",
        "them",
        "these",
        "this",
        "those",
    }
    return any(token in history_dependent_terms for token in tokens)


def build_product_diagnosis(state: RunState) -> dict[str, Any]:
    packet = state.final_packet or {}
    retrieved = []
    if state.retrieval_iterations:
        retrieved = list(state.retrieval_iterations[-1].get("retrieved_chunks", []) or [])
    evidence = list(packet.get("verified_evidence", []) or [])
    source_map = list(packet.get("answer_source_map", []) or [])
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
        "answer_source_map_count": len(source_map),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "supporting_trace_refs": [
            {"path": "documents", "reason": "Active user corpus and load errors."},
            {"path": "retrieval_iterations[-1].retrieved_chunks", "reason": "Chunks selected from the corpus."},
            {"path": "final_packet.verified_evidence", "reason": "Evidence passed to synthesis or preview."},
            {"path": "final_packet.answer_source_map", "reason": "Answer sections linked back to cited evidence."},
            {"path": "final_packet.unresolved", "reason": "Gaps that should be resolved before relying on output."},
        ],
    }


def compare_product_traces(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    parent_evidence = extract_trace_evidence(parent)
    child_evidence = extract_trace_evidence(child)
    parent_keys = {item["key"] for item in parent_evidence}
    child_keys = {item["key"] for item in child_evidence}
    parent_unresolved = set(extract_unresolved(parent))
    child_unresolved = set(extract_unresolved(child))
    parent_docs = set(extract_document_paths(parent))
    child_docs = set(extract_document_paths(child))
    parent_metrics = parent.get("metrics") or {}
    child_metrics = child.get("metrics") or {}
    return {
        "mode": "product_trace_comparison",
        "parent_run_id": parent.get("run_id"),
        "child_run_id": child.get("run_id"),
        "parent_task_id": parent.get("task_id"),
        "child_task_id": child.get("task_id"),
        "objective_changed": (parent.get("task") or {}).get("question") != (child.get("task") or {}).get("question"),
        "answer_changed": normalize_compare_text(parent.get("rendered_answer")) != normalize_compare_text(child.get("rendered_answer")),
        "document_delta": {
            "added": sorted(child_docs - parent_docs),
            "removed": sorted(parent_docs - child_docs),
            "kept_count": len(parent_docs & child_docs),
        },
        "evidence_delta": {
            "added_count": len(child_keys - parent_keys),
            "removed_count": len(parent_keys - child_keys),
            "kept_count": len(parent_keys & child_keys),
            "added": [item for item in child_evidence if item["key"] in child_keys - parent_keys][:20],
            "removed": [item for item in parent_evidence if item["key"] in parent_keys - child_keys][:20],
        },
        "unresolved_delta": {
            "added": sorted(child_unresolved - parent_unresolved),
            "removed": sorted(parent_unresolved - child_unresolved),
            "kept_count": len(parent_unresolved & child_unresolved),
        },
        "metrics_delta": {
            "total_tokens": int(child_metrics.get("total_tokens") or 0) - int(parent_metrics.get("total_tokens") or 0),
            "estimated_cost": float(child_metrics.get("estimated_cost") or 0.0)
            - float(parent_metrics.get("estimated_cost") or 0.0),
        },
    }


def extract_trace_evidence(trace: dict[str, Any]) -> list[dict[str, str]]:
    packet = trace.get("final_packet") or {}
    rows: list[dict[str, str]] = []
    for item in packet.get("verified_evidence", []) or []:
        source = item.get("source") or {}
        support = normalize_compare_text(item.get("raw_support"))
        key = "|".join(
            [
                str(source.get("doc_id") or ""),
                str(source.get("chunk_id") or ""),
                support[:220],
            ]
        )
        rows.append(
            {
                "key": key,
                "doc_id": str(source.get("doc_id") or ""),
                "chunk_id": str(source.get("chunk_id") or ""),
                "claim": compact_compare_text(item.get("claim")),
                "support": compact_compare_text(item.get("raw_support")),
            }
        )
    return rows


def extract_unresolved(trace: dict[str, Any]) -> list[str]:
    packet = trace.get("final_packet") or {}
    return [normalize_compare_text(item) for item in packet.get("unresolved", []) or [] if normalize_compare_text(item)]


def extract_document_paths(trace: dict[str, Any]) -> list[str]:
    return [
        str(doc.get("path") or doc.get("filename") or doc.get("doc_id") or "")
        for doc in trace.get("documents", []) or []
        if str(doc.get("path") or doc.get("filename") or doc.get("doc_id") or "").strip()
    ]


def normalize_compare_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def compact_compare_text(value: Any, *, max_chars: int = 280) -> str:
    text = normalize_compare_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_product_synthesis_prompt(state: RunState) -> str:
    packet = state.final_packet or {}
    return f"""Produce the requested legal work product or answer using only the user-provided corpus.

Conversation history:
{format_conversation_history(packet.get("conversation_history", []))}

Use conversation history for continuity only. Do not treat prior answers as source evidence unless the current retrieved corpus also supports them.

User objective:
{state.task.question}

Active corpus:
{format_documents_for_prompt(state.documents)}

Evidence packet:
{format_evidence_for_prompt(packet.get("verified_evidence", []))}

Worker analysis:
{packet.get("worker_analysis") or "- Not run."}

Unresolved gaps:
{format_list(packet.get("unresolved", []))}

Write a clean answer. Cite source doc IDs and chunk IDs when relying on evidence. If the corpus does not contain enough support, state the gap plainly."""


def build_product_worker_analysis_prompt(state: RunState) -> str:
    packet = state.final_packet or {}
    return f"""Analyze the retrieved user-corpus evidence for the product objective.

Return compact structured notes for the final drafter. Use only the provided evidence.

Conversation history:
{format_conversation_history(packet.get("conversation_history", []))}

Use conversation history for continuity only. Do not treat prior answers as source evidence unless the current evidence packet supports them.

User objective:
{state.task.question}

Evidence packet:
{format_evidence_for_prompt(packet.get("verified_evidence", []))}

Return:
- ANSWER_TARGET: the exact question or artifact to satisfy.
- MATERIAL_FACTS: concise source-grounded facts with doc/chunk IDs.
- CONDITIONS_OR_REQUIREMENTS: notice, timing, approval, threshold, document, or procedural requirements.
- GAPS: facts missing from the corpus that the drafter should not invent.
- DRAFTING_NOTES: constraints for a clean user-facing answer."""


def write_product_answer_artifact(state: RunState, *, diagnostic: bool) -> dict[str, Any]:
    if not state.output_dir:
        raise ValueError("state.output_dir is required for product artifact writing")
    output_dir = Path(state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "answer.md"
    path.write_text(state.rendered_answer or "", encoding="utf-8")
    return {
        "type": "diagnostic_preview_markdown" if diagnostic else "answer_markdown",
        "path": str(path),
        "filename": path.name,
        "diagnostic": diagnostic,
        "chars": len(state.rendered_answer or ""),
    }


def format_conversation_history(history: list[dict[str, Any]]) -> str:
    turns = normalize_conversation_history(history)
    if not turns:
        return "- None."
    lines: list[str] = []
    for index, turn in enumerate(turns, 1):
        lines.append(f"Turn {index} user: {turn['user']}")
        lines.append(f"Turn {index} final answer: {turn['assistant']}")
    return "\n".join(lines)


def build_answer_source_map(answer: str, evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_by_ref: dict[str, dict[str, Any]] = {}
    for item in evidence_items:
        source = item.get("source") or {}
        for ref in [source.get("doc_id"), source.get("chunk_id")]:
            if ref:
                evidence_by_ref[str(ref)] = item
    rows: list[dict[str, Any]] = []
    for index, section in enumerate(split_answer_sections(answer), 1):
        refs = sorted(set(re.findall(r"doc_\d{4}(?:_chunk_\d{4})?", section)))
        if not refs:
            continue
        support = []
        for ref in refs:
            item = evidence_by_ref.get(ref)
            if item is None and "_chunk_" in ref:
                item = evidence_by_ref.get(ref.split("_chunk_", 1)[0])
            if item is None:
                continue
            source = item.get("source") or {}
            support.append(
                {
                    "ref": ref,
                    "doc_id": source.get("doc_id"),
                    "chunk_id": source.get("chunk_id"),
                    "support": compact_compare_text(item.get("raw_support")),
                }
            )
        rows.append(
            {
                "section_index": index,
                "answer_excerpt": compact_compare_text(section, max_chars=420),
                "source_refs": refs,
                "support": support,
            }
        )
    return rows


def split_answer_sections(answer: str) -> list[str]:
    sections = [section.strip() for section in re.split(r"\n\s*\n", answer or "") if section.strip()]
    if len(sections) <= 1:
        sections = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    return sections


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
