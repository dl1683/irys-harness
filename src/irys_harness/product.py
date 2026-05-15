from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Callable

from .config import HarnessConfig
from .events import EventLogger
from .indexing import RetrievedChunk, load_documents, retrieve_chunks, tokenize
from .metrics import ModelCallRecord
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
DEFAULT_PRODUCT_TOP_K = 36
MAX_SOURCE_PLANNER_PATHS = 4000
MAX_SOURCE_PLANNER_CANDIDATE_PATHS = 240
MAX_SOURCE_PLANNER_DIRECTORY_ROWS = 80
MAX_SOURCE_PLANNER_OUTPUT_TOKENS = 4096
MAX_SUPPLEMENTAL_FIRST_READ_DOCS = 12


class ProductRunCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusScopeDecision:
    selected_paths: list[Path]
    discovered_paths: list[Path]
    reason: str
    signals: dict[str, Any]
    scored_paths: list[dict[str, Any]]
    model_calls: list[ModelCallRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_paths": [str(path) for path in self.selected_paths],
            "discovered_paths": [str(path) for path in self.discovered_paths],
            "reason": self.reason,
            "signals": self.signals,
            "scored_paths": self.scored_paths,
            "model_calls": [call.to_dict() for call in self.model_calls],
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
    top_k: int = DEFAULT_PRODUCT_TOP_K,
    max_files: int | None = DEFAULT_PRODUCT_MAX_FILES,
    max_chars_per_doc: int | None = None,
    verbose: bool = True,
    parent_trace_path: str | None = None,
    user_nudge: str | None = None,
    selected_paths: list[str] | None = None,
    pinned_paths: list[str] | None = None,
    plan_note: str | None = None,
    use_llm_planning: bool = False,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ProductRunResult:
    if not objective.strip():
        raise ValueError("objective is required")
    pre_events: list[dict[str, Any]] = []
    if user_nudge:
        emit_pre_event(
            pre_events,
            event_callback,
            "STEER",
            "Applying user steering",
            summary="Applying your steering note before choosing sources.",
            user_nudge=user_nudge,
            source_selection_mode="locked_by_user" if selected_paths else "replan_from_nudge",
            next_step="Re-check the document inventory against the updated objective.",
        )
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
        next_step="Choose the best first-read set before extracting text.",
    )
    emit_pre_event(
        pre_events,
        event_callback,
        "SCOPE",
        "Choosing first-read documents",
        summary="Ranking the file inventory against your question before reading full documents.",
        document_count=len(corpus_paths),
        planner="cheap_worker" if use_llm_planning else "path_scoring",
        next_step="Select the most relevant files to read first, or keep the full corpus if narrowing is unsafe.",
    )
    scope_decision = plan_product_corpus_scope(
        objective=objective,
        paths=corpus_paths,
        selected_paths=selected_paths,
        plan_note=plan_note,
        config=config,
        use_llm_planning=use_llm_planning,
    )
    pinned_corpus_paths = resolve_pinned_paths(pinned_paths, corpus_paths)
    if pinned_corpus_paths:
        emit_pre_event(
            pre_events,
            event_callback,
            "STEER",
            "Pinned sources for synthesis",
            summary=f"Keeping {len(pinned_corpus_paths)} user-pinned source document(s) in the synthesis packet.",
            pinned_documents=[path.name for path in pinned_corpus_paths[:12]],
            omitted_document_count=max(0, len(pinned_corpus_paths) - 12),
            next_step="Include pinned sources even if the automatic first-read plan would not select them.",
        )
    if len(scope_decision.selected_paths) < len(scope_decision.discovered_paths):
        emit_pre_event(
            pre_events,
            event_callback,
            "SCOPE",
            "Selected first-read documents",
            summary=(
                f"Selected {len(scope_decision.selected_paths)} likely relevant document(s) "
                f"before reading the full corpus."
            ),
            reason=scope_decision.reason,
            selected_documents=[path.name for path in scope_decision.selected_paths[:12]],
            omitted_document_count=max(0, len(scope_decision.selected_paths) - 12),
            skipped_document_count=len(scope_decision.discovered_paths) - len(scope_decision.selected_paths),
            planner=scope_decision.signals.get("source_planner", {}).get("status"),
            steer_hint=(
                "Use a steering note like 'focus on the 2024 10-K' or 'include quarterly reports too' "
                "if this first-read set is wrong."
            ),
            next_step="Read only the selected first-read documents.",
        )
    else:
        emit_pre_event(
            pre_events,
            event_callback,
            "SCOPE",
            "Using full corpus",
            summary="No narrow document target was clear, so the run will read the full selected corpus.",
            reason=scope_decision.reason,
            next_step="Read documents and extract searchable text.",
        )
    active_corpus_paths = unique_paths([*scope_decision.selected_paths, *pinned_corpus_paths])
    check_product_cancel(should_cancel)
    emit_pre_event(
        pre_events,
        event_callback,
        "SCOPE",
        "Ready to read active corpus",
        summary=f"Active first-read corpus: {len(active_corpus_paths)} document(s).",
        document_count=len(active_corpus_paths),
        sample_documents=[path.name for path in active_corpus_paths[:12]],
        omitted_document_count=max(0, len(active_corpus_paths) - 12),
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
        context_files=[str(path) for path in active_corpus_paths],
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
            "plan_note": plan_note,
            "pinned_context_files": [str(path) for path in pinned_corpus_paths],
            "conversation_history_turns": len(normalized_history),
            "conversation_history_policy": "synthesis_only_user_question_and_final_answer",
            "discovered_context_files": [str(path) for path in corpus_paths],
            "corpus_scope_decision": scope_decision.to_dict(),
        },
    )
    state = RunState(task=task, config=config, output_dir=str(Path(output_dir) / task_id))
    for call in scope_decision.model_calls:
        state.metrics.add_call(call)
    state.events.extend(pre_events)
    log = EventLogger(state, verbose=verbose, event_callback=event_callback)
    log.emit(
        "RUN",
        "started product matter run",
        matter=task_id,
        summary=f"Started matter run for {matter_id}.",
        next_step="Read the active first-read corpus.",
    )

    state.documents, state.chunks = load_documents(
        [str(path) for path in active_corpus_paths],
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
    retrieved = retrieve_product_chunks(state.chunks, queries, state.documents, top_k=top_k)
    source_coverage = build_source_coverage(state.documents, retrieved)
    state.retrieval_iterations.append(
        {
            "iteration": 1,
            "queries": queries,
            "retrieved_chunks": [item.to_dict() for item in retrieved],
            "reason": "Product matter retrieval from the current objective and answer target. Conversation history is intentionally excluded from retrieval queries.",
            "conversation_history_used": False,
            "strategy": "global_plus_source_diverse_retrieval",
            "source_coverage": source_coverage,
        }
    )
    log.emit(
        "SEARCH",
        "retrieved candidate chunks",
        chunks=len(retrieved),
        top_k=top_k,
        summary=f"Selected {len(retrieved)} candidate passage(s) for evidence review.",
        selected_sources=summarize_retrieved_sources(retrieved, state.documents),
        source_coverage=source_coverage,
        next_step="Extract usable evidence from the selected passages.",
    )

    evidence_items = build_product_evidence_items(retrieved, queries=queries)
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

    packet_review: dict[str, Any] | None = None
    if live_synthesis:
        packet_review = review_product_packet_with_cheap_worker(
            state=state,
            queries=queries,
            evidence_items=evidence_items,
            source_coverage=source_coverage,
            log=log,
            should_cancel=should_cancel,
        )
        revised_queries = [
            str(query).strip()
            for query in (packet_review.get("revised_queries", []) if packet_review else [])
            if str(query).strip()
        ][:8]
        if packet_review and packet_review.get("continue_retrieval") is True and revised_queries:
            check_product_cancel(should_cancel)
            supplemental_paths, supplemental_profile, supplemental_calls = select_supplemental_product_paths(
                objective=objective,
                discovered_paths=scope_decision.discovered_paths,
                active_paths=active_corpus_paths,
                packet_review=packet_review,
                config=config,
                use_llm_planning=use_llm_planning,
                limit=MAX_SUPPLEMENTAL_FIRST_READ_DOCS,
            )
            for call in supplemental_calls:
                state.metrics.add_call(call)
            if supplemental_paths:
                active_corpus_paths = unique_paths([*active_corpus_paths, *supplemental_paths, *pinned_corpus_paths])
                state.task.context_files = [str(path) for path in active_corpus_paths]
                state.task.metadata["supplemental_context_files"] = [str(path) for path in supplemental_paths]
                state.task.metadata["supplemental_source_planner"] = supplemental_profile
                log.emit(
                    "SCOPE",
                    "adding supplemental documents",
                    summary=(
                        f"The packet reviewer found a gap, so Irys is adding "
                        f"{len(supplemental_paths)} held-back document(s) before drafting."
                    ),
                    selected_documents=[path.name for path in supplemental_paths[:12]],
                    omitted_document_count=max(0, len(supplemental_paths) - 12),
                    missing_information=packet_review.get("missing_information", []),
                    queries=revised_queries,
                    reason=supplemental_profile.get("reason"),
                    planner=(supplemental_profile.get("source_planner") or {}).get("status")
                    or supplemental_profile.get("status"),
                    next_step="Read the expanded active corpus and search again.",
                )
                state.documents, state.chunks = load_documents(
                    [str(path) for path in active_corpus_paths],
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
                    "loaded expanded user corpus",
                    documents=len(state.documents),
                    chunks=len(state.chunks),
                    summary=(
                        f"Loaded expanded corpus: {len(state.documents)} document(s), "
                        f"{len(state.chunks)} searchable chunk(s)."
                    ),
                    next_step="Run the reviewer-requested evidence search.",
                )
            supplemental = retrieve_product_chunks(
                state.chunks,
                revised_queries,
                state.documents,
                top_k=top_k,
            )
            retrieved = merge_retrieved_chunks(retrieved, supplemental, limit=top_k)
            source_coverage = build_source_coverage(state.documents, retrieved)
            state.retrieval_iterations.append(
                {
                    "iteration": len(state.retrieval_iterations) + 1,
                    "queries": revised_queries,
                    "retrieved_chunks": [item.to_dict() for item in retrieved],
                    "reason": "Cheap worker packet reviewer requested a second retrieval pass before synthesis.",
                    "conversation_history_used": False,
                    "strategy": "worker_review_retrieval_expansion",
                    "source_coverage": source_coverage,
                    "supplemental_context_files": [str(path) for path in supplemental_paths],
                }
            )
            log.emit(
                "SEARCH",
                "expanded evidence after packet review",
                chunks=len(retrieved),
                top_k=top_k,
                summary="A cheap worker reviewed the packet and requested another retrieval pass before drafting.",
                queries=revised_queries,
                selected_sources=summarize_retrieved_sources(retrieved, state.documents),
                source_coverage=source_coverage,
                next_step="Rebuild the evidence packet with the expanded sources.",
            )
            evidence_items = build_product_evidence_items(retrieved, queries=queries + revised_queries)
            state.extraction_records.append(
                {
                    "mode": "expanded_product_snippet_extraction",
                    "evidence_items": evidence_items,
                    "reviewer_queries": revised_queries,
                }
            )
            state.verification_records.append(
                {
                    "mode": "expanded_source_chunk_presence_check",
                    "accepted": [item["claim"] for item in evidence_items],
                    "rejected": [],
                    "weak": [],
                }
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
        "source_coverage": source_coverage,
        "verified_evidence": evidence_items,
        "packet_review": packet_review or {"status": "not_run"},
        "pinned_sources": build_pinned_source_records(
            state.documents,
            state.chunks,
            pinned_paths=[str(path) for path in pinned_corpus_paths],
            evidence_items=evidence_items,
        ),
        "unresolved": build_unresolved_items(
            state,
            source_coverage=source_coverage,
            packet_review=packet_review,
        ),
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


def plan_product_corpus_scope(
    *,
    objective: str,
    paths: list[Path],
    selected_paths: list[str] | None = None,
    plan_note: str | None = None,
    config: HarnessConfig | None = None,
    use_llm_planning: bool = False,
) -> CorpusScopeDecision:
    resolved_paths = [path.resolve() for path in paths]
    path_by_key = {str(path).lower(): path for path in resolved_paths}
    if selected_paths:
        approved: list[Path] = []
        for raw_path in selected_paths:
            key = str(Path(raw_path).expanduser().resolve()).lower()
            if key in path_by_key and path_by_key[key] not in approved:
                approved.append(path_by_key[key])
        if approved:
            return CorpusScopeDecision(
                selected_paths=approved,
                discovered_paths=resolved_paths,
                reason="User-approved first-read paths from the editable plan.",
                signals=build_objective_profile(objective, plan_note=plan_note),
                scored_paths=score_corpus_paths(objective, resolved_paths, plan_note=plan_note)[:40],
            )

    profile = build_objective_profile(objective, plan_note=plan_note)
    scored = score_corpus_paths(objective, resolved_paths, plan_note=plan_note, profile=profile)
    model_calls: list[ModelCallRecord] = []
    if not resolved_paths:
        selected = []
    elif profile["force_full_corpus"]:
        selected = resolved_paths
    elif use_llm_planning and config is not None:
        selected, planner_profile, planner_calls = plan_with_cheap_source_planner(
            objective=objective,
            plan_note=plan_note,
            paths=resolved_paths,
            scored_paths=scored,
            config=config,
        )
        profile["source_planner"] = planner_profile
        model_calls.extend(planner_calls)
    else:
        selected = select_first_read_paths(scored, total_paths=len(resolved_paths), profile=profile)
    if not selected:
        selected = resolved_paths
    planner_status = (profile.get("source_planner") or {}).get("status")
    planner_reason = str((profile.get("source_planner") or {}).get("reason") or "").strip()
    if len(selected) == len(resolved_paths):
        if planner_status == "used":
            reason = (
                "Worker source planner recommended reading the full selected corpus."
                + (f" {planner_reason}" if planner_reason else "")
            )
        else:
            reason = "No confident narrow first-read set was inferable from the objective and corpus structure."
    elif planner_status == "used":
        reason = (
            f"Worker source planner selected {len(selected)} of {len(resolved_paths)} discovered document(s)."
            + (f" {planner_reason}" if planner_reason else "")
        )
    else:
        reason = explain_scope_reason(profile, selected, len(resolved_paths))
    return CorpusScopeDecision(
        selected_paths=selected,
        discovered_paths=resolved_paths,
        reason=reason,
        signals=profile,
        scored_paths=scored[:80],
        model_calls=model_calls,
    )


def unique_paths(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        output.append(resolved)
        seen.add(key)
    return output


def resolve_pinned_paths(pinned_paths: list[str] | None, corpus_paths: list[Path]) -> list[Path]:
    if not pinned_paths:
        return []
    path_by_key = {str(path.resolve()).lower(): path.resolve() for path in corpus_paths}
    output: list[Path] = []
    seen: set[str] = set()
    for raw_path in pinned_paths:
        key = str(Path(raw_path).expanduser().resolve()).lower()
        pinned = path_by_key.get(key)
        if pinned is None or key in seen:
            continue
        output.append(pinned)
        seen.add(key)
    return output


def build_product_plan_preview(
    *,
    objective: str,
    paths: list[str],
    max_files: int | None = DEFAULT_PRODUCT_MAX_FILES,
    selected_paths: list[str] | None = None,
    plan_note: str | None = None,
    top_k: int = DEFAULT_PRODUCT_TOP_K,
    config: HarnessConfig | None = None,
    use_llm_planning: bool = False,
) -> dict[str, Any]:
    corpus_paths = discover_corpus_paths(paths, max_files=max_files)
    if not corpus_paths:
        raise ValueError("at least one supported document path is required")
    scope = plan_product_corpus_scope(
        objective=objective,
        paths=corpus_paths,
        selected_paths=selected_paths,
        plan_note=plan_note,
        config=config,
        use_llm_planning=use_llm_planning,
    )
    profile = scope.signals
    planner_needed = (profile.get("source_planner") or {}).get("needed_information") or []
    needed_information = unique_strings([*infer_needed_information(profile), *planner_needed])
    return {
        "objective": objective.strip(),
        "plan_note": plan_note or "",
        "interpreted_goal": objective.strip(),
        "document_strategy": scope.reason,
        "first_read_paths": [str(path) for path in scope.selected_paths],
        "first_read_count": len(scope.selected_paths),
        "discovered_count": len(scope.discovered_paths),
        "not_read_first_count": max(0, len(scope.discovered_paths) - len(scope.selected_paths)),
        "likely_document_families": profile.get("requested_families", []),
        "detected_years": profile.get("years", []),
        "needed_information": needed_information,
        "search_queries": build_path_level_queries(objective, profile),
        "top_candidates": scope.scored_paths[:20],
        "source_planner": profile.get("source_planner", {"status": "not_requested"}),
        "planner_metrics": {
            "estimated_cost": sum(call.estimated_cost for call in scope.model_calls),
            "total_tokens": sum(call.input_tokens + call.output_tokens for call in scope.model_calls),
            "model_calls": [call.to_dict() for call in scope.model_calls],
        },
        "retrieval": {"top_k": top_k},
    }


def plan_with_cheap_source_planner(
    *,
    objective: str,
    plan_note: str | None,
    paths: list[Path],
    scored_paths: list[dict[str, Any]],
    config: HarnessConfig,
) -> tuple[list[Path], dict[str, Any], list[ModelCallRecord]]:
    if not paths:
        return [], {"status": "not_run", "reason": "empty corpus"}, []
    if len(paths) > MAX_SOURCE_PLANNER_PATHS:
        return [], {
            "status": "skipped",
            "reason": f"corpus has {len(paths)} paths, above planner limit {MAX_SOURCE_PLANNER_PATHS}",
        }, []

    prompt = build_source_planner_prompt(
        objective=objective,
        plan_note=plan_note,
        paths=paths,
        scored_paths=scored_paths,
    )
    try:
        result = GeminiModelRouter(config).generate(
            module="product_source_planner",
            prompt=prompt,
            temperature=0.0,
            max_output_tokens=MAX_SOURCE_PLANNER_OUTPUT_TOKENS,
        )
        payload = parse_json_object(result.text)
        selected_paths = unique_paths(
            [
                *resolve_planner_selected_paths(payload, paths),
                *resolve_planner_selected_directories(payload, paths),
            ]
        )
        if payload.get("should_read_full_corpus") is True:
            selected_paths = paths
        status = "used" if selected_paths else "empty_selection"
        return selected_paths, {
            "status": status,
            "reason": str(payload.get("reason") or payload.get("document_strategy") or ""),
            "confidence": str(payload.get("confidence") or ""),
            "should_read_full_corpus": bool(payload.get("should_read_full_corpus")),
            "selected_count": len(selected_paths),
            "selected_paths": [str(path) for path in selected_paths],
            "selected_directories": [str(item) for item in payload.get("selected_directories", [])[:50]]
            if isinstance(payload.get("selected_directories"), list)
            else [],
            "rejected_paths": [str(item) for item in payload.get("rejected_paths", [])[:50]]
            if isinstance(payload.get("rejected_paths"), list)
            else [],
            "needed_information": [str(item) for item in payload.get("needed_information", [])[:20]]
            if isinstance(payload.get("needed_information"), list)
            else [],
            "raw_response": compact_snippet(result.text, max_chars=4000),
        }, [result.usage]
    except Exception as exc:  # noqa: BLE001 - product UI should fall back cleanly if planning fails.
        return [], {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }, []


def build_source_planner_prompt(
    *,
    objective: str,
    plan_note: str | None,
    paths: list[Path],
    scored_paths: list[dict[str, Any]],
) -> str:
    inventory = build_source_planner_inventory(paths=paths, scored_paths=scored_paths)
    return (
        "You are a cheap worker model helping route a user-defined document corpus before expensive reading.\n"
        "Decide which files should be read first for the user's objective. Use semantic judgment over path names, "
        "folder structure, file types, dates, and deterministic hints. Prefer high-quality work product over minimal token use.\n"
        "If the objective likely requires broad comparison or the candidate list is too ambiguous, set should_read_full_corpus true.\n"
        "Do not overfit to finance or legal filings; this may be legal, finance, biomedical, governance, email, research, or another document set.\n\n"
        f"User objective:\n{objective.strip()}\n\n"
        f"User plan correction:\n{(plan_note or '').strip() or '- None.'}\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "selected_paths": ["exact path strings from candidate_paths"],\n'
        '  "selected_directories": ["exact directory strings from directory_summary if a whole folder should be read"],\n'
        '  "rejected_paths": ["exact path strings from candidate_paths that look lower priority"],\n'
        '  "should_read_full_corpus": false,\n'
        '  "reason": "short practical explanation",\n'
        '  "needed_information": ["what the run must find"],\n'
        '  "confidence": "low|medium|high"\n'
        "}\n\n"
        "Compact corpus inventory JSON:\n"
        f"{json.dumps(inventory, indent=2)}"
    )


def build_source_planner_inventory(*, paths: list[Path], scored_paths: list[dict[str, Any]]) -> dict[str, Any]:
    score_by_path = {str(item.get("path") or ""): item for item in scored_paths}
    candidate_source_rows = scored_paths[:MAX_SOURCE_PLANNER_CANDIDATE_PATHS]
    seen_candidates = {str(row.get("path") or "") for row in candidate_source_rows}
    if len(candidate_source_rows) < min(len(paths), MAX_SOURCE_PLANNER_CANDIDATE_PATHS):
        for path in paths:
            key = str(path)
            if key in seen_candidates:
                continue
            candidate_source_rows.append(score_by_path.get(key) or {"path": key, "score": 0, "reasons": []})
            seen_candidates.add(key)
            if len(candidate_source_rows) >= MAX_SOURCE_PLANNER_CANDIDATE_PATHS:
                break

    candidate_rows = []
    score_by_path = {str(item.get("path") or ""): item for item in scored_paths}
    for index, row in enumerate(candidate_source_rows, 1):
        path = Path(str(row.get("path") or ""))
        scored = score_by_path.get(str(path), row)
        reasons = "; ".join(str(reason) for reason in scored.get("reasons", [])[:4])
        candidate_rows.append(
            {
                "index": index,
                "path": str(path),
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "deterministic_score": int(scored.get("score") or 0),
                "deterministic_reasons": compact_snippet(reasons, max_chars=280),
            }
        )

    directory_rows = summarize_corpus_directories(paths)
    extension_counts: dict[str, int] = {}
    for path in paths:
        extension_counts[path.suffix.lower() or "<none>"] = extension_counts.get(path.suffix.lower() or "<none>", 0) + 1
    return {
        "total_path_count": len(paths),
        "candidate_path_count": len(candidate_rows),
        "omitted_candidate_count": max(0, len(paths) - len(candidate_rows)),
        "extension_counts": dict(sorted(extension_counts.items())),
        "directory_summary": directory_rows,
        "candidate_paths": candidate_rows,
        "candidate_rule": (
            "candidate_paths contains the strongest deterministic path matches plus a bounded sample. "
            "Select exact candidate path strings or request full corpus if the answer likely needs omitted files."
        ),
    }


def summarize_corpus_directories(paths: list[Path]) -> list[dict[str, Any]]:
    by_parent: dict[str, dict[str, Any]] = {}
    for path in paths:
        key = str(path.parent)
        row = by_parent.setdefault(
            key,
            {
                "directory": key,
                "file_count": 0,
                "extensions": {},
                "sample_files": [],
            },
        )
        row["file_count"] += 1
        suffix = path.suffix.lower() or "<none>"
        row["extensions"][suffix] = row["extensions"].get(suffix, 0) + 1
        if len(row["sample_files"]) < 5:
            row["sample_files"].append(path.name)
    rows = sorted(by_parent.values(), key=lambda row: (-int(row["file_count"]), str(row["directory"]).lower()))
    return rows[:MAX_SOURCE_PLANNER_DIRECTORY_ROWS]


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("planner response must be a JSON object")
    return parsed


def resolve_planner_selected_paths(payload: dict[str, Any], paths: list[Path]) -> list[Path]:
    raw_items = payload.get("selected_paths", [])
    if not isinstance(raw_items, list):
        return []
    by_exact = {str(path).lower(): path for path in paths}
    by_name: dict[str, list[Path]] = {}
    for path in paths:
        by_name.setdefault(path.name.lower(), []).append(path)
    selected: list[Path] = []
    for raw_item in raw_items:
        raw = str(raw_item or "").strip()
        if not raw:
            continue
        match = by_exact.get(raw.lower())
        if match is None:
            name_matches = by_name.get(Path(raw).name.lower(), [])
            if len(name_matches) == 1:
                match = name_matches[0]
        if match is not None and match not in selected:
            selected.append(match)
    return selected


def resolve_planner_selected_directories(payload: dict[str, Any], paths: list[Path]) -> list[Path]:
    raw_items = payload.get("selected_directories", [])
    if not isinstance(raw_items, list):
        return []
    resolved_paths = [path.resolve() for path in paths]
    known_dirs = {str(path.parent.resolve()).lower(): path.parent.resolve() for path in resolved_paths}
    selected_dirs: list[Path] = []
    for raw_item in raw_items:
        raw = str(raw_item or "").strip()
        if not raw:
            continue
        try:
            candidate = Path(raw).expanduser().resolve()
        except OSError:
            continue
        key = str(candidate).lower()
        if key in known_dirs and candidate not in selected_dirs:
            selected_dirs.append(candidate)

    if not selected_dirs:
        return []

    selected_paths: list[Path] = []
    seen: set[str] = set()
    for path in resolved_paths:
        parent = path.parent.resolve()
        if not any(path_is_relative_to(parent, directory) for directory in selected_dirs):
            continue
        key = str(path).lower()
        if key not in seen:
            selected_paths.append(path)
            seen.add(key)
    return selected_paths


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def unique_strings(items: list[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def build_objective_profile(objective: str, *, plan_note: str | None = None) -> dict[str, Any]:
    combined = f"{objective}\n{plan_note or ''}"
    lowered = combined.lower()
    tokens = [token for token in tokenize(combined) if token not in PATH_PLANNING_STOPWORDS]
    years = sorted({int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", combined)})
    requested_families = infer_requested_document_families(lowered, tokens)
    return {
        "tokens": tokens,
        "years": years,
        "requested_families": requested_families,
        "issue_discovery": is_issue_discovery_objective(lowered, tokens),
        "force_full_corpus": any(
            phrase in lowered
            for phrase in [
                "all documents",
                "all files",
                "entire corpus",
                "everything",
                "read all",
                "whole folder",
            ]
        ),
        "has_focus_language": any(term in lowered for term in ["focus", "only", "first", "start with", "prioritize"]),
        "raw": combined,
    }


PATH_PLANNING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "here",
    "in",
    "is",
    "it",
    "main",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "which",
    "with",
}


def is_issue_discovery_objective(lowered: str, tokens: list[str]) -> bool:
    issue_phrases = [
        "main issue",
        "key issue",
        "what is going on",
        "what's going on",
        "what happened",
        "what is this about",
        "what's this about",
        "what is the problem",
        "identify the issue",
        "identify issues",
        "spot the issue",
        "issue spot",
        "summarize the matter",
        "understand the matter",
        "what are we dealing with",
    ]
    if any(phrase in lowered for phrase in issue_phrases):
        return True
    token_set = set(tokens)
    return bool({"issue", "issues", "problem", "dispute", "matter"} & token_set) and len(token_set) <= 8


def infer_requested_document_families(lowered: str, tokens: list[str]) -> list[str]:
    families: list[str] = []
    token_set = set(tokens)
    family_terms = [
        (
            "annual_report",
            {
                "10-k",
                "10k",
                "annual",
                "fiscal",
                "fy",
                "eps",
                "earnings",
                "revenue",
                "income",
                "profit",
                "loss",
                "shares",
                "stockholder",
                "stockholders",
            },
        ),
        ("quarterly_report", {"10-q", "10q", "quarter", "quarterly", "q1", "q2", "q3", "q4"}),
        ("current_report", {"8-k", "8k", "current", "event", "press", "release", "announced"}),
        (
            "agreement",
            {
                "agreement",
                "contract",
                "amendment",
                "clause",
                "msa",
                "lease",
                "credit",
                "loan",
                "indenture",
                "guaranty",
                "warranty",
                "termination",
                "indemnity",
                "indemnification",
                "covenant",
            },
        ),
        ("governance", {"board", "minutes", "consent", "resolution", "charter", "bylaws", "governance"}),
        ("table_or_workbook", {"schedule", "spreadsheet", "table", "xlsx", "workbook", "ledger", "list"}),
        (
            "research_paper",
            {
                "abstract",
                "article",
                "biomedical",
                "clinical",
                "cohort",
                "conclusion",
                "doi",
                "experiment",
                "finding",
                "journal",
                "literature",
                "method",
                "methods",
                "paper",
                "papers",
                "publication",
                "research",
                "result",
                "results",
                "study",
                "trial",
            },
        ),
        (
            "regulatory_or_policy",
            {
                "policy",
                "regulation",
                "regulatory",
                "rule",
                "guidance",
                "notice",
                "order",
                "statute",
                "section",
                "agency",
            },
        ),
        (
            "litigation",
            {
                "complaint",
                "motion",
                "brief",
                "deposition",
                "exhibit",
                "filing",
                "docket",
                "declaration",
                "affidavit",
                "transcript",
            },
        ),
        ("email", {"email", "eml", "message", "correspondence"}),
        ("presentation", {"presentation", "deck", "slides", "pptx"}),
    ]
    for family, hints in family_terms:
        if token_set & hints or any(hint in lowered for hint in hints if "-" in hint):
            families.append(family)
    return families


def infer_needed_information(profile: dict[str, Any]) -> list[str]:
    families = set(profile.get("requested_families", []))
    items = ["the user's requested answer", "the most likely source documents", "supporting source excerpts"]
    if "annual_report" in families or "quarterly_report" in families:
        items.extend(["the relevant reporting period", "the requested financial metric", "the source table or note"])
    if "agreement" in families:
        items.extend(["the controlling provision", "any exceptions or amendments", "the applicable party or obligation"])
    if "governance" in families:
        items.extend(["the relevant board action", "date and authority", "supporting minutes, consent, or charter text"])
    if profile.get("issue_discovery"):
        items.extend(
            [
                "case-specific narrative sources",
                "the event timeline",
                "the parties and disputed action",
                "background rules only after the practical issue is identified",
            ]
        )
    if "research_paper" in families:
        items.extend(["the relevant paper or study", "the method or population", "the finding or conclusion"])
    if "regulatory_or_policy" in families:
        items.extend(["the controlling rule or policy", "the applicable section", "exceptions or effective dates"])
    if "litigation" in families:
        items.extend(["the relevant filing or exhibit", "the procedural posture", "the specific factual or legal assertion"])
    return list(dict.fromkeys(items))


def build_path_level_queries(objective: str, profile: dict[str, Any]) -> list[str]:
    queries = [objective.strip(), " ".join(profile.get("tokens", []))]
    families = profile.get("requested_families", [])
    years = " ".join(str(year) for year in profile.get("years", []))
    if families:
        queries.append(" ".join(families + ([years] if years else [])))
    return [query for query in queries if query.strip()]


def score_corpus_paths(
    objective: str,
    paths: list[Path],
    *,
    plan_note: str | None = None,
    profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    profile = profile or build_objective_profile(objective, plan_note=plan_note)
    fiscal_year_by_accession = infer_sec_report_years_from_indices(paths)
    rows = [score_one_path(path, profile, fiscal_year_by_accession) for path in paths]
    return sorted(rows, key=lambda row: (-int(row["score"]), str(row["path"]).lower()))


def score_one_path(
    path: Path,
    profile: dict[str, Any],
    fiscal_year_by_accession: dict[str, int],
) -> dict[str, Any]:
    path_text = normalize_path_text(path)
    filename = path.name.lower()
    tokens = [token for token in profile.get("tokens", []) if len(token) >= 3]
    years = list(profile.get("years", []))
    families = list(profile.get("requested_families", []))
    score = 0
    reasons: list[str] = []

    matched_tokens = [token for token in tokens if token in path_text]
    if matched_tokens:
        score += min(12, len(set(matched_tokens)) * 2)
        reasons.append(f"path matched query terms: {', '.join(sorted(set(matched_tokens))[:6])}")

    source_role_score, source_role_reasons = score_path_source_role(path_text, profile)
    if source_role_score:
        score += source_role_score
        reasons.extend(source_role_reasons)

    family_score = score_path_family(path_text, families)
    if family_score:
        score += family_score
        reasons.append("path matched likely document family")

    if years:
        path_years = {int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", path_text)}
        matched_years = sorted(set(years) & path_years)
        if matched_years:
            score += 8 * len(matched_years)
            reasons.append(f"path matched year(s): {', '.join(str(year) for year in matched_years)}")
        accession = accession_from_path(path)
        fiscal_year = fiscal_year_by_accession.get(accession or "")
        if fiscal_year and fiscal_year in years:
            score += 22
            reasons.append(f"index maps accession to fiscal/report year {fiscal_year}")
        if "annual_report" in families and any(year + 1 in path_years for year in years):
            score += 7
            reasons.append("annual report filing date may follow the fiscal year")

    if filename in {"index.md", "index.csv"}:
        score += 3
        reasons.append("index file can explain the folder structure")
        if families and not any(family in {"table_or_workbook"} for family in families):
            score -= 2

    return {
        "path": str(path),
        "filename": path.name,
        "score": score,
        "reasons": reasons or ["no strong path-level match"],
    }


def score_path_source_role(path_text: str, profile: dict[str, Any]) -> tuple[int, list[str]]:
    if not profile.get("issue_discovery"):
        return 0, []
    score = 0
    reasons: list[str] = []
    case_specific_terms = [
        "email",
        "emails",
        "chain",
        "correspondence",
        "letter",
        "letters",
        "resignation",
        "notice",
        "memo",
        "notes",
        "minutes",
        "transcript",
        "complaint",
        "motion",
        "brief",
        "deposition",
        "messages",
        "thread",
        "chat",
    ]
    background_terms = [
        "act",
        "regulation",
        "regulations",
        "statute",
        "code",
        "rules",
        "policy",
        "manual",
        "handbook",
        "courtesy-copy",
        "published-by",
        "registrar",
    ]
    if any(term in path_text for term in case_specific_terms):
        score += 18
        reasons.append("issue-discovery source: likely case-specific narrative document")
    if any(term in path_text for term in background_terms):
        score -= 8
        reasons.append("issue-discovery source: likely background rule or generic reference")
    return score, reasons


def score_path_family(path_text: str, families: list[str]) -> int:
    score = 0
    family_path_terms = {
        "annual_report": ["10-k", "10_k", "10k", "annual-report", "annual-reports", "annual reports"],
        "quarterly_report": ["10-q", "10_q", "10q", "quarterly", "quarterly-results"],
        "current_report": ["8-k", "8_k", "8k", "news-release", "news-releases", "press-release"],
        "agreement": ["agreement", "agreements", "contract", "contracts", "amendment", "guaranty", "lease"],
        "governance": ["board", "minutes", "consent", "charter", "bylaws", "governance"],
        "table_or_workbook": ["schedule", "table", "workbook", "spreadsheet", ".xlsx", ".xlsm", ".csv"],
        "research_paper": [
            "paper",
            "papers",
            "research",
            "study",
            "studies",
            "journal",
            "article",
            "clinical",
            "trial",
            "biomedical",
            "abstract",
            "methods",
            "results",
        ],
        "regulatory_or_policy": ["policy", "regulation", "regulatory", "rule", "guidance", "notice", "order", "statute"],
        "litigation": ["complaint", "motion", "brief", "deposition", "exhibit", "filing", "docket", "transcript"],
        "email": [".eml", "email", "correspondence"],
        "presentation": [".pptx", "presentation", "deck", "slides"],
    }
    for family in families:
        terms = family_path_terms.get(family, [])
        if any(term in path_text for term in terms):
            score += 14
    return score


def select_first_read_paths(
    scored: list[dict[str, Any]],
    *,
    total_paths: int,
    profile: dict[str, Any],
) -> list[Path]:
    if not scored:
        return []
    top_score = int(scored[0]["score"])
    if top_score < 10:
        return []
    focused = bool(
        profile.get("has_focus_language")
        or profile.get("years")
        or profile.get("requested_families")
        or profile.get("issue_discovery")
    )
    if not focused and total_paths <= 12:
        return [Path(str(row["path"])) for row in scored]
    floor = max(8, top_score - 8)
    limit = 18 if focused else 30
    selected_rows = [row for row in scored if int(row["score"]) >= floor and int(row["score"]) > 0]
    if not selected_rows:
        selected_rows = [scored[0]]
    selected_rows = selected_rows[:limit]
    return [Path(str(row["path"])) for row in selected_rows]


def explain_scope_reason(profile: dict[str, Any], selected: list[Path], total_paths: int) -> str:
    parts = []
    families = profile.get("requested_families") or []
    years = profile.get("years") or []
    if families:
        parts.append("matched likely document family: " + ", ".join(families))
    if years:
        parts.append("matched requested year(s): " + ", ".join(str(year) for year in years))
    if not parts:
        parts.append("matched objective terms in the corpus structure")
    return f"{'; '.join(parts)}. First-read set is {len(selected)} of {total_paths} discovered document(s)."


def normalize_path_text(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    return re.sub(r"[_\s]+", "-", text)


def accession_from_path(path: Path) -> str | None:
    match = re.search(r"(?<!\d)\d{10}-\d{2}-\d{6}(?!\d)", path.name)
    return match.group(0) if match else None


def infer_sec_report_years_from_indices(paths: list[Path]) -> dict[str, int]:
    index_paths = [
        path
        for path in paths
        if path.name.lower() == "index.md" and any(segment.lower() in {"10-k", "10-q"} for segment in path.parts)
    ]
    report_years: dict[str, int] = {}
    for index_path in index_paths[:20]:
        try:
            text = index_path.read_text(encoding="utf-8", errors="replace")[:200_000]
        except OSError:
            continue
        for line in text.splitlines():
            accession_match = re.search(r"\b\d{10}-\d{2}-\d{6}\b", line)
            if not accession_match:
                continue
            date_match = re.search(r"(?:19|20)\d{2}(?:1231|0630|0930|0331)", line)
            if date_match:
                report_years[accession_match.group(0)] = int(date_match.group(0)[:4])
    return report_years


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
    profile = build_objective_profile(state.task.question, plan_note=state.task.metadata.get("plan_note"))
    needed_information = unique_strings(
        [
            "user objective",
            "active corpus inventory",
            "most relevant source chunks",
            "source-grounded facts and gaps",
            "clean answer or artifact draft",
            *infer_needed_information(profile),
        ]
    )
    return {
        "version": 1,
        "interpreted_goal": state.task.question,
        "required_output_format": "clean product answer or draft work product",
        "document_boundary": "user_defined_corpus",
        "needed_information": needed_information,
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
    profile = build_objective_profile(objective)
    objective_terms = " ".join(profile.get("tokens", []) or tokenize(objective))
    queries = [objective, objective_terms]
    families = set(profile.get("requested_families", []))
    if "annual_report" in families or "quarterly_report" in families:
        queries.extend(
            [
                "earnings per share diluted basic weighted average shares",
                "net income loss per share diluted basic fiscal year financial highlights",
            ]
        )
    if "current_report" in families:
        queries.append("press release announcement financial results business highlights priorities")
    if profile.get("issue_discovery"):
        queries.append("email correspondence timeline issue dispute problem parties events")
    if "agreement" in families:
        queries.append("agreement covenant obligation termination notice cure exception")
    if "regulatory_or_policy" in families:
        queries.append("rule policy regulation section requirement exception effective date")
    if "research_paper" in families:
        queries.append("study method result finding conclusion population trial")
    return unique_strings([query for query in queries if query.strip()])


def retrieve_product_chunks(
    chunks: list[dict[str, Any]],
    queries: list[str],
    documents: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[Any]:
    global_hits = retrieve_chunks(chunks, queries, top_k=max(top_k, min(top_k * 2, 120)))
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(str(chunk.get("doc_id")), []).append(chunk)

    per_doc_hits: list[Any] = []
    doc_count = len([doc for doc in documents if not doc.get("load_error")])
    if doc_count <= max(top_k, 80):
        per_doc_limit = 2 if top_k >= doc_count * 2 else 1
        for doc in documents:
            doc_id = str(doc.get("doc_id") or "")
            doc_chunks = chunks_by_doc.get(doc_id) or []
            if not doc_chunks:
                continue
            per_doc_hits.extend(retrieve_chunks(doc_chunks, queries, top_k=per_doc_limit))

    by_key: dict[tuple[str, str], RetrievedChunk] = {}
    for item in [*global_hits, *per_doc_hits]:
        boosted = boost_product_retrieval_score(item, queries)
        by_key[(boosted.doc_id, boosted.chunk_id)] = boosted

    if not by_key:
        return []

    best_by_doc: dict[str, Any] = {}
    for item in by_key.values():
        current = best_by_doc.get(item.doc_id)
        if current is None or item.score > current.score:
            best_by_doc[item.doc_id] = item

    selected: list[Any] = []
    seen: set[tuple[str, str]] = set()
    if doc_count <= max(top_k, 80):
        for item in sorted(best_by_doc.values(), key=lambda hit: hit.score, reverse=True):
            key = (item.doc_id, item.chunk_id)
            if key not in seen:
                selected.append(item)
                seen.add(key)
            if len(selected) >= top_k:
                return hydrate_retrieved_chunks_with_neighbors(selected, chunks)

    for item in sorted(by_key.values(), key=lambda hit: hit.score, reverse=True):
        key = (item.doc_id, item.chunk_id)
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= top_k:
            break
    return hydrate_retrieved_chunks_with_neighbors(selected, chunks)


def boost_product_retrieval_score(item: RetrievedChunk, queries: list[str]) -> RetrievedChunk:
    boost = product_retrieval_boost(item.text, queries)
    if boost <= 0:
        return item
    return RetrievedChunk(
        chunk_id=item.chunk_id,
        doc_id=item.doc_id,
        score=float(item.score) + boost,
        text=item.text,
    )


def product_retrieval_boost(text: str, queries: list[str]) -> float:
    query_text = " ".join(str(query or "") for query in queries).lower()
    text_lower = str(text or "").lower()
    boost = 0.0
    if any(term in query_text for term in ["eps", "earnings per share", "per share", "diluted", "basic"]):
        metric_phrases = [
            "net income per share",
            "net loss per share",
            "earnings per share",
            "net income (loss) per share",
            "income (loss) per share",
            "per share - basic",
            "per share - diluted",
            "per share attributable",
            "weighted-average shares",
            "weighted average shares",
        ]
        for phrase in metric_phrases:
            if phrase in text_lower:
                boost += 4.0
        if "basic" in text_lower and "diluted" in text_lower and "per share" in text_lower:
            boost += 6.0
        if "fiscal year" in text_lower or "year ended" in text_lower or "full year" in text_lower:
            boost += 2.0
    return boost


def hydrate_retrieved_chunks_with_neighbors(retrieved: list[Any], chunks: list[dict[str, Any]]) -> list[RetrievedChunk]:
    chunk_by_id = {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks}
    hydrated: list[RetrievedChunk] = []
    for item in retrieved:
        current = chunk_by_id.get(str(item.chunk_id)) or {}
        parts: list[str] = []
        prev_id = str(current.get("prev_chunk") or "")
        next_id = str(current.get("next_chunk") or "")
        if prev_id and prev_id in chunk_by_id:
            parts.append("[Previous chunk tail]\n" + tail_text(str(chunk_by_id[prev_id].get("text") or ""), 1800))
        parts.append("[Matched chunk]\n" + str(item.text or ""))
        if next_id and next_id in chunk_by_id:
            parts.append("[Next chunk head]\n" + head_text(str(chunk_by_id[next_id].get("text") or ""), 1800))
        hydrated.append(
            RetrievedChunk(
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                score=item.score,
                text="\n\n".join(parts),
            )
        )
    return hydrated


def build_source_coverage(documents: list[dict[str, Any]], retrieved: list[Any]) -> dict[str, Any]:
    represented_doc_ids = {str(item.doc_id) for item in retrieved}
    loaded_documents = [doc for doc in documents if not doc.get("load_error")]
    missing_documents = [doc for doc in loaded_documents if str(doc.get("doc_id")) not in represented_doc_ids]
    represented_documents = [doc for doc in loaded_documents if str(doc.get("doc_id")) in represented_doc_ids]
    return {
        "loaded_document_count": len(loaded_documents),
        "represented_document_count": len(represented_documents),
        "missing_document_count": len(missing_documents),
        "represented_documents": [
            {
                "doc_id": doc.get("doc_id"),
                "filename": doc.get("filename"),
                "path": doc.get("path"),
            }
            for doc in represented_documents[:40]
        ],
        "missing_documents": [
            {
                "doc_id": doc.get("doc_id"),
                "filename": doc.get("filename"),
                "path": doc.get("path"),
            }
            for doc in missing_documents[:40]
        ],
    }


def merge_retrieved_chunks(existing: list[Any], supplemental: list[Any], *, limit: int) -> list[Any]:
    by_key: dict[tuple[str, str], Any] = {}
    for item in [*supplemental, *existing]:
        key = (str(item.doc_id), str(item.chunk_id))
        current = by_key.get(key)
        if current is None or float(item.score) > float(current.score):
            by_key[key] = item
    if not by_key:
        return []
    best_by_doc: dict[str, Any] = {}
    for item in by_key.values():
        current = best_by_doc.get(item.doc_id)
        if current is None or float(item.score) > float(current.score):
            best_by_doc[item.doc_id] = item
    selected: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(best_by_doc.values(), key=lambda hit: float(hit.score), reverse=True):
        key = (str(item.doc_id), str(item.chunk_id))
        selected.append(item)
        seen.add(key)
        if len(selected) >= limit:
            return selected
    for item in sorted(by_key.values(), key=lambda hit: float(hit.score), reverse=True):
        key = (str(item.doc_id), str(item.chunk_id))
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def review_product_packet_with_cheap_worker(
    *,
    state: RunState,
    queries: list[str],
    evidence_items: list[dict[str, Any]],
    source_coverage: dict[str, Any],
    log: EventLogger,
    should_cancel: Callable[[], bool] | None,
) -> dict[str, Any]:
    check_product_cancel(should_cancel)
    log.emit(
        "ANALYZE",
        "reviewing packet before synthesis",
        summary="Asking a cheap worker to check whether the evidence packet is sufficient before drafting.",
        next_step="Expand retrieval if the worker finds missing source coverage or weak evidence.",
    )
    prompt = build_product_packet_review_prompt(
        state=state,
        queries=queries,
        evidence_items=evidence_items,
        source_coverage=source_coverage,
    )
    try:
        result = GeminiModelRouter(state.config).generate(
            module="product_packet_reviewer",
            prompt=prompt,
            temperature=0.0,
            max_output_tokens=8192,
        )
        state.metrics.add_call(result.usage)
        payload = parse_json_object(result.text)
        review = normalize_packet_review(payload)
        review.update(
            {
                "status": "used",
                "model": result.usage.model,
                "raw_response": compact_snippet(result.text, max_chars=6000),
            }
        )
        state.critic_records.append(
            {
                "mode": "cheap_worker_product_packet_review",
                **review,
            }
        )
        log.emit(
            "ANALYZE",
            "packet review complete",
            model=result.usage.model,
            summary=str(review.get("assessment") or "Packet review complete."),
            continue_retrieval=review.get("continue_retrieval"),
            revised_queries=review.get("revised_queries", []),
            missing_information=review.get("missing_information", []),
            coverage_risks=review.get("coverage_risks", []),
            relevant_source_ids=review.get("relevant_source_ids", []),
            low_value_source_ids=review.get("low_value_source_ids", []),
            next_step="Use the reviewed packet or run an additional retrieval pass.",
        )
        return review
    except Exception as exc:  # noqa: BLE001 - live packet review should not block final synthesis.
        review = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "sufficient": None,
            "continue_retrieval": False,
            "revised_queries": [],
            "missing_information": [],
            "assessment": "Packet review failed; proceeding with deterministic evidence packet.",
        }
        state.critic_records.append({"mode": "cheap_worker_product_packet_review", **review})
        log.emit(
            "ANALYZE",
            "packet review unavailable",
            summary=review["assessment"],
            error=review["error"],
            next_step="Proceed with the current evidence packet.",
        )
        return review


def build_product_packet_review_prompt(
    *,
    state: RunState,
    queries: list[str],
    evidence_items: list[dict[str, Any]],
    source_coverage: dict[str, Any],
) -> str:
    contract = state.answer_contract_versions[-1] if state.answer_contract_versions else {}
    return f"""Review a user-corpus evidence packet before final synthesis.

Use semantic judgment. This is a cheap worker step, so favor quality and missing-evidence detection over saving tokens.
Do not answer the user. Decide whether the current packet is enough for the final synthesizer, whether some loaded documents should be treated as low value, and whether another retrieval pass should run.

Return JSON only:
{{
  "sufficient": true,
  "continue_retrieval": false,
  "assessment": "short practical assessment",
  "missing_information": ["..."],
  "revised_queries": ["..."],
  "relevant_source_ids": ["doc_0001"],
  "low_value_source_ids": ["doc_0002"],
  "coverage_risks": ["..."]
}}

Rules:
- If the task asks for a year, period, entity, agreement, issue, or deliverable and the evidence does not directly cover it, set sufficient false.
- If selected documents exist but evidence is dominated by the wrong source family or stale period, set continue_retrieval true and provide better queries.
- If the answer would say information is unavailable, require strong coverage across likely source documents first.
- Queries should include synonyms and source-specific terms, not huge filename lists.
- Use the source IDs in the active corpus. Do not invent sources.

User objective:
{state.task.question}

Answer needs:
{format_list(list(contract.get("needed_information", []) or []))}

Current retrieval queries:
{format_list(queries)}

Active corpus:
{format_documents_for_prompt(state.documents)}

Source coverage:
{json.dumps(source_coverage, indent=2)}

Evidence packet:
{format_evidence_for_prompt(evidence_items)}
"""


def normalize_packet_review(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "sufficient": bool(payload.get("sufficient")) if payload.get("sufficient") is not None else None,
        "continue_retrieval": bool(payload.get("continue_retrieval")),
        "assessment": str(payload.get("assessment") or "").strip(),
        "missing_information": normalize_string_list(payload.get("missing_information"), limit=20),
        "revised_queries": normalize_string_list(payload.get("revised_queries"), limit=12),
        "relevant_source_ids": normalize_string_list(payload.get("relevant_source_ids"), limit=80),
        "low_value_source_ids": normalize_string_list(payload.get("low_value_source_ids"), limit=80),
        "coverage_risks": normalize_string_list(payload.get("coverage_risks"), limit=20),
    }


def select_supplemental_product_paths(
    *,
    objective: str,
    discovered_paths: list[Path],
    active_paths: list[Path],
    packet_review: dict[str, Any],
    config: HarnessConfig,
    use_llm_planning: bool,
    limit: int = MAX_SUPPLEMENTAL_FIRST_READ_DOCS,
) -> tuple[list[Path], dict[str, Any], list[ModelCallRecord]]:
    active_resolved = {path.resolve() for path in active_paths}
    held_back = [path.resolve() for path in discovered_paths if path.resolve() not in active_resolved]
    if not held_back or limit <= 0:
        return [], {"status": "no_held_back_documents", "reason": "No held-back documents were available."}, []

    missing = normalize_string_list(packet_review.get("missing_information"), limit=12)
    revised = normalize_string_list(packet_review.get("revised_queries"), limit=12)
    risks = normalize_string_list(packet_review.get("coverage_risks"), limit=8)
    supplemental_objective = "\n".join(
        part
        for part in [
            objective.strip(),
            "Supplemental source selection requested by packet reviewer.",
            "Missing information: " + "; ".join(missing) if missing else "",
            "Reviewer search targets: " + "; ".join(revised) if revised else "",
            "Coverage risks: " + "; ".join(risks) if risks else "",
        ]
        if part
    )

    if use_llm_planning:
        decision = plan_product_corpus_scope(
            objective=supplemental_objective,
            paths=held_back,
            config=config,
            use_llm_planning=True,
        )
        selected = decision.selected_paths
        if len(selected) > limit and len(held_back) > limit:
            selected = selected[:limit]
        profile = {
            "status": "used",
            "reason": decision.reason,
            "source_planner": decision.signals.get("source_planner", {}),
            "selected_count": len(selected),
            "held_back_count": len(held_back),
            "missing_information": missing,
            "coverage_risks": risks,
        }
        return selected, profile, decision.model_calls

    scored = score_corpus_paths(supplemental_objective, held_back)
    selected = [Path(row["path"]).resolve() for row in scored[:limit]]
    profile = {
        "status": "path_scoring",
        "reason": "Selected held-back documents against the packet reviewer's missing information and revised queries.",
        "selected_count": len(selected),
        "held_back_count": len(held_back),
        "missing_information": missing,
        "coverage_risks": risks,
    }
    return selected, profile, []


def normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return unique_strings([str(item).strip() for item in value if str(item).strip()])[:limit]


def build_product_evidence_items(retrieved: list[Any], *, queries: list[str] | None = None) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(retrieved, 1):
        snippet = compact_relevant_snippet(item.text, queries=queries or [], max_chars=2400)
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


def compact_relevant_snippet(text: str, *, queries: list[str], max_chars: int = 1600) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    lower = cleaned.lower()
    query_text = " ".join(str(query or "") for query in queries)
    query_terms = [term for term in tokenize(query_text) if len(term) >= 4 or term == "eps"]
    phrases = []
    if any(term in {"eps", "share", "shares", "income", "loss", "diluted", "basic"} for term in query_terms):
        phrases.extend(
            [
                "net income per share after",
                "net loss per share after",
                "earnings per share after",
                "net income per share",
                "net loss per share",
                "earnings per share",
                "per share",
                "diluted",
                "basic",
            ]
        )
    phrases.extend(sorted(set(query_terms), key=len, reverse=True))
    positions = [lower.find(phrase) for phrase in phrases if phrase and lower.find(phrase) >= 0]
    if not positions:
        return compact_snippet(cleaned, max_chars=max_chars)
    center = min(positions)
    query_lower = query_text.lower()
    if "before" not in query_lower and has_before_after_adjustment_per_share(lower):
        after_match = re.search(
            r"\b(?:net\s+income|net\s+loss|earnings)?\s*per\s+share\s+after\b.{0,160}?(?:tax|adjustment|adjustments)",
            lower,
        )
        if after_match:
            center = after_match.start()
    start = max(0, center - max_chars // 3)
    end = min(len(cleaned), start + max_chars)
    start = max(0, end - max_chars)
    snippet = cleaned[start:end].strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(cleaned):
        snippet += " ..."
    return snippet


def has_before_after_adjustment_per_share(text: str) -> bool:
    normalized_words = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    has_before_adjustment = bool(
        re.search(r"\bbefore\b(?:\s+\w+){0,12}\s+(?:tax|adjustment|adjustments)", normalized_words)
    )
    has_after_adjustment = bool(
        re.search(r"\bafter\b(?:\s+\w+){0,12}\s+(?:tax|adjustment|adjustments)", normalized_words)
    )
    return has_before_adjustment and has_after_adjustment and ("per share" in normalized_words or "eps" in normalized_words)


def head_text(text: str, max_chars: int) -> str:
    cleaned = str(text or "")
    return cleaned[:max_chars]


def tail_text(text: str, max_chars: int) -> str:
    cleaned = str(text or "")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[-max_chars:]


def build_unresolved_items(
    state: RunState,
    *,
    source_coverage: dict[str, Any] | None = None,
    packet_review: dict[str, Any] | None = None,
) -> list[str]:
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
    coverage = source_coverage or {}
    loaded_count = int(coverage.get("loaded_document_count") or 0)
    missing_count = int(coverage.get("missing_document_count") or 0)
    if loaded_count > 1 and missing_count:
        items.append(
            f"{missing_count} loaded source document(s) did not contribute retrieved evidence; review source coverage before making absence claims."
        )
    if packet_review:
        if packet_review.get("sufficient") is False:
            assessment = str(packet_review.get("assessment") or "Cheap worker packet review found the evidence insufficient.")
            items.append(assessment)
        for missing in packet_review.get("missing_information", [])[:6]:
            items.append(f"Packet review missing information: {missing}")
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
    pinned_sources = list(packet.get("pinned_sources", []) or [])
    source_coverage = dict(packet.get("source_coverage") or {})
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
        "pinned_source_count": len(pinned_sources),
        "source_coverage": source_coverage,
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


def build_pinned_source_records(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    pinned_paths: list[str],
    evidence_items: list[dict[str, Any]],
    max_chunks_per_doc: int = 6,
    max_chars_per_doc: int = 16_000,
) -> list[dict[str, Any]]:
    pinned_keys = {str(Path(path).expanduser().resolve()).lower() for path in pinned_paths}
    if not pinned_keys:
        return []
    docs_by_id = {
        str(doc.get("doc_id")): doc
        for doc in documents
        if str(doc.get("path") or "").strip()
        and str(Path(str(doc.get("path"))).expanduser().resolve()).lower() in pinned_keys
    }
    if not docs_by_id:
        return []
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
    chunks_by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or "")
        chunks_by_doc.setdefault(doc_id, []).append(chunk)
        chunks_by_id[str(chunk.get("chunk_id") or "")] = chunk
    pinned_records: list[dict[str, Any]] = []
    for doc_id, doc in docs_by_id.items():
        preferred_chunk_ids: list[str] = []
        for item in evidence_items:
            source = item.get("source") or {}
            if str(source.get("doc_id") or "") == doc_id and source.get("chunk_id"):
                preferred_chunk_ids.append(str(source.get("chunk_id")))
        selected_chunks: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()
        for chunk_id in preferred_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            for neighbor_id in [chunk.get("prev_chunk"), chunk.get("chunk_id"), chunk.get("next_chunk")]:
                neighbor = chunks_by_id.get(str(neighbor_id or ""))
                if not neighbor:
                    continue
                key = str(neighbor.get("chunk_id") or "")
                if key and key not in seen_chunks:
                    selected_chunks.append(neighbor)
                    seen_chunks.add(key)
        for chunk in chunks_by_doc.get(doc_id, []):
            if len(selected_chunks) >= max_chunks_per_doc:
                break
            key = str(chunk.get("chunk_id") or "")
            if key and key not in seen_chunks:
                selected_chunks.append(chunk)
                seen_chunks.add(key)
        excerpts: list[dict[str, Any]] = []
        used_chars = 0
        for chunk in selected_chunks:
            if len(excerpts) >= max_chunks_per_doc or used_chars >= max_chars_per_doc:
                break
            text = str(chunk.get("text") or "")
            remaining = max_chars_per_doc - used_chars
            if len(text) > remaining:
                text = text[:remaining]
            used_chars += len(text)
            excerpts.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "index": chunk.get("index"),
                    "text": text,
                }
            )
        pinned_records.append(
            {
                "doc_id": doc_id,
                "filename": doc.get("filename"),
                "path": doc.get("path"),
                "text_chars": doc.get("text_chars"),
                "excerpts": excerpts,
            }
        )
    return pinned_records


def build_product_synthesis_prompt(state: RunState) -> str:
    packet = state.final_packet or {}
    metric_notes = build_metric_selection_notes(packet.get("verified_evidence", []))
    return f"""Produce the requested legal work product or answer using only the user-provided corpus.

Conversation history:
{format_conversation_history(packet.get("conversation_history", []))}

Use conversation history for continuity only. Do not treat prior answers as source evidence unless the current retrieved corpus also supports them.

User objective:
{state.task.question}

Active corpus:
{format_documents_for_prompt(state.documents)}

Pinned sources:
{format_pinned_sources_for_prompt(packet.get("pinned_sources", []))}

Evidence packet:
{format_evidence_for_prompt(packet.get("verified_evidence", []))}

Worker analysis:
{packet.get("worker_analysis") or "- Not run."}

Metric selection notes:
{format_list(metric_notes)}

Unresolved gaps:
{format_list(packet.get("unresolved", []))}

Numeric and table evidence: preserve row labels, column labels, periods, classes, and basic/diluted distinctions. Do not collapse nearby values into one figure unless the source explicitly says they are identical. When a table gives paired values for the same requested metric, report the pair unless the user asked for only one side.
For every numeric fact, preserve the source period exactly. Do not label a three-month, quarterly, Q4, or interim value as fiscal-year or annual. Only call a value fiscal-year, full-year, or annual when the source row or heading says fiscal year, year ended, full year, or equivalent.
If a reconciliation table includes both before-adjustment and after-adjustment versions of a metric, treat before-adjustment rows as intermediate reconciliation rows. Use the final or after-adjustment row for a general adjusted/non-GAAP answer unless the user explicitly asks for the before-adjustment figure.
If the metric selection notes identify final/after-adjustment evidence, that evidence controls over any worker analysis sentence that only mentions before-adjustment values.

Write a clean answer. Cite source doc IDs and chunk IDs when relying on evidence. If the corpus does not contain enough support, state the gap plainly."""


def build_product_worker_analysis_prompt(state: RunState) -> str:
    packet = state.final_packet or {}
    metric_notes = build_metric_selection_notes(packet.get("verified_evidence", []))
    return f"""Analyze the retrieved user-corpus evidence for the product objective.

Return compact structured notes for the final drafter. Use only the provided evidence.

Conversation history:
{format_conversation_history(packet.get("conversation_history", []))}

Use conversation history for continuity only. Do not treat prior answers as source evidence unless the current evidence packet supports them.

User objective:
{state.task.question}

Evidence packet:
{format_evidence_for_prompt(packet.get("verified_evidence", []))}

Pinned sources:
{format_pinned_sources_for_prompt(packet.get("pinned_sources", []))}

Metric selection notes:
{format_list(metric_notes)}

Return:
- ANSWER_TARGET: the exact question or artifact to satisfy.
- MATERIAL_FACTS: concise source-grounded facts with doc/chunk IDs.
- CONDITIONS_OR_REQUIREMENTS: notice, timing, approval, threshold, document, or procedural requirements.
- GAPS: facts missing from the corpus that the drafter should not invent.
- DRAFTING_NOTES: constraints for a clean user-facing answer.

For numeric or table evidence, preserve row labels, column labels, periods, classes, and basic/diluted distinctions. Do not collapse adjacent rows or columns into one value unless the source explicitly says they are identical. When a table gives paired values for the same requested metric, keep the pair unless the user asked for only one side.
For every numeric fact, preserve the source period exactly. Do not label a three-month, quarterly, Q4, or interim value as fiscal-year or annual. Only call a value fiscal-year, full-year, or annual when the source row or heading says fiscal year, year ended, full year, or equivalent.
If a reconciliation table includes both before-adjustment and after-adjustment versions of a metric, treat before-adjustment rows as intermediate reconciliation rows. Use the final or after-adjustment row for a general adjusted/non-GAAP answer unless the user explicitly asks for the before-adjustment figure."""


def build_metric_selection_notes(evidence_items: list[dict[str, Any]]) -> list[str]:
    evidence_text = "\n".join(str(item.get("raw_support") or "") for item in evidence_items).lower()
    notes: list[str] = []
    if has_before_after_adjustment_per_share(evidence_text):
        notes.append(
            "This packet contains both before-adjustment and after-adjustment per-share rows. "
            "Before-adjustment rows are intermediate reconciliation rows; use the final/after-adjustment per-share row for a general adjusted or non-GAAP EPS answer unless the user asked for before-adjustment EPS."
        )
        notes.extend(extract_adjusted_metric_windows(evidence_items, limit=2))
    return notes


def extract_adjusted_metric_windows(evidence_items: list[dict[str, Any]], *, limit: int) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for item in evidence_items:
        display_text = " ".join(str(item.get("raw_support") or "").replace("\xa0", " ").split())
        lower_display = display_text.lower()
        row_matches = list(
            re.finditer(
                r"\bper\s+share\s+after\b.{0,120}?(?:tax|adjustment|adjustments)",
                lower_display,
            )
        )
        if not row_matches:
            row_matches = list(re.finditer(r"\bafter\b.{0,120}?(?:tax|adjustment|adjustments)", lower_display))
        source = item.get("source") or {}
        source_ref = " / ".join(str(part) for part in [source.get("doc_id"), source.get("chunk_id")] if part)
        period_score = 0 if any(term in lower_display for term in ["year ended", "fiscal year", "full year"]) else 1
        for match in row_matches:
            start = max(0, match.start() - 80)
            end = min(len(display_text), match.end() + 360)
            window = display_text[start:end].strip()
            if "per share" not in window.lower():
                continue
            prefix = f"{source_ref}: " if source_ref else ""
            candidates.append((period_score, f"Detected final/after-adjustment per-share evidence: {prefix}{window}"))
    return unique_strings([item for _, item in sorted(candidates, key=lambda pair: pair[0])])[:limit]


def write_product_answer_artifact(state: RunState, *, diagnostic: bool) -> dict[str, Any]:
    if not state.output_dir:
        raise ValueError("state.output_dir is required for product artifact writing")
    output_dir = Path(state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"answer-{state.run_id}.md"
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
    pinned = packet.get("pinned_sources", [])
    if pinned:
        lines.extend(["", "## Pinned Sources"])
        for source in pinned:
            lines.append(f"- {source.get('doc_id')}: {source.get('filename')} ({len(source.get('excerpts') or [])} excerpt(s) pinned)")
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


def format_pinned_sources_for_prompt(pinned_sources: list[dict[str, Any]]) -> str:
    if not pinned_sources:
        return "- None."
    parts: list[str] = []
    for source in pinned_sources:
        parts.append(f"- Pinned document: {source.get('doc_id')} {source.get('filename')} path={source.get('path')}")
        excerpts = source.get("excerpts") or []
        if not excerpts:
            parts.append("  - No extracted excerpts were available.")
            continue
        for excerpt in excerpts:
            parts.append(f"  - {excerpt.get('chunk_id')} index={excerpt.get('index')}: {excerpt.get('text')}")
    return "\n".join(parts)


def format_list(items: list[str]) -> str:
    if not items:
        return "- None."
    return "\n".join(f"- {item}" for item in items)
