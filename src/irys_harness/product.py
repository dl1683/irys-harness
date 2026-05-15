from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Callable

from .config import HarnessConfig
from .events import EventLogger
from .indexing import load_documents, retrieve_chunks, tokenize
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
MAX_SOURCE_PLANNER_PATHS = 4000
MAX_SOURCE_PLANNER_OUTPUT_TOKENS = 4096


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
    top_k: int = 12,
    max_files: int | None = DEFAULT_PRODUCT_MAX_FILES,
    max_chars_per_doc: int | None = None,
    verbose: bool = True,
    parent_trace_path: str | None = None,
    user_nudge: str | None = None,
    selected_paths: list[str] | None = None,
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
    scope_decision = plan_product_corpus_scope(
        objective=objective,
        paths=corpus_paths,
        selected_paths=selected_paths,
        plan_note=plan_note,
        config=config,
        use_llm_planning=use_llm_planning,
    )
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
    active_corpus_paths = scope_decision.selected_paths
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


def build_product_plan_preview(
    *,
    objective: str,
    paths: list[str],
    max_files: int | None = DEFAULT_PRODUCT_MAX_FILES,
    selected_paths: list[str] | None = None,
    plan_note: str | None = None,
    top_k: int = 12,
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
        selected_paths = resolve_planner_selected_paths(payload, paths)
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
    inventory_rows = []
    score_by_path = {str(item.get("path") or ""): item for item in scored_paths}
    for index, path in enumerate(paths, 1):
        scored = score_by_path.get(str(path), {})
        reasons = "; ".join(str(reason) for reason in scored.get("reasons", [])[:4])
        inventory_rows.append(
            {
                "index": index,
                "path": str(path),
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "deterministic_score": int(scored.get("score") or 0),
                "deterministic_reasons": reasons,
            }
        )
    return (
        "You are a cheap worker model helping route a user-defined document corpus before expensive reading.\n"
        "Decide which files should be read first for the user's objective. Use semantic judgment over path names, "
        "folder structure, file types, dates, and deterministic hints. Prefer high-quality work product over minimal token use.\n"
        "If the objective likely requires broad comparison or the path inventory is too ambiguous, set should_read_full_corpus true.\n"
        "Do not overfit to finance or legal filings; this may be legal, finance, biomedical, governance, email, research, or another document set.\n\n"
        f"User objective:\n{objective.strip()}\n\n"
        f"User plan correction:\n{(plan_note or '').strip() or '- None.'}\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "selected_paths": ["exact path strings from the inventory"],\n'
        '  "rejected_paths": ["exact path strings that look lower priority"],\n'
        '  "should_read_full_corpus": false,\n'
        '  "reason": "short practical explanation",\n'
        '  "needed_information": ["what the run must find"],\n'
        '  "confidence": "low|medium|high"\n'
        "}\n\n"
        "Path inventory JSON:\n"
        f"{json.dumps(inventory_rows, indent=2)}"
    )


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
    match = re.search(r"\b\d{10}-\d{2}-\d{6}\b", path.name)
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
