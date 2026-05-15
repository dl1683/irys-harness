from __future__ import annotations

from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .config import HarnessConfig
from .product import (
    ProductRunCancelled,
    build_product_plan_preview,
    compare_product_traces,
    run_product_matter,
    sanitize_matter_id,
)
from .trace import load_trace, trace_summary


RUN_JOBS: dict[str, dict[str, Any]] = {}
RUN_JOBS_LOCK = threading.Lock()


def serve_product_ui(
    *,
    host: str,
    port: int,
    config: HarnessConfig,
    trace_dir: str | Path,
    output_dir: str | Path,
) -> None:
    handler = build_handler(config=config, trace_dir=trace_dir, output_dir=output_dir)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"[UI] product UI serving at http://{host}:{port}")
    server.serve_forever()


def build_handler(
    *,
    config: HarnessConfig,
    trace_dir: str | Path,
    output_dir: str | Path,
) -> type[BaseHTTPRequestHandler]:
    class ProductUIHandler(BaseHTTPRequestHandler):
        server_version = "IrysProductUI/0.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self.send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/health":
                self.send_json({"ok": True})
                return
            if parsed.path == "/api/run-status":
                query = parse_qs(parsed.query)
                job_id = query.get("job_id", [""])[0]
                try:
                    self.send_json(snapshot_run_job(job_id))
                except Exception as exc:  # noqa: BLE001 - UI endpoint should return structured error.
                    self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/trace":
                query = parse_qs(parsed.query)
                trace_path = query.get("path", [""])[0]
                try:
                    trace = load_trace(resolve_trace_path(trace_path, trace_dir))
                    response = {
                        "summary": trace_summary(trace),
                        "trace": compact_trace_for_ui(trace),
                    }
                    comparison = comparison_from_parent(trace, trace_dir=trace_dir)
                    if comparison:
                        response["comparison"] = comparison
                    self.send_json(response)
                except Exception as exc:  # noqa: BLE001 - UI endpoint should return structured error.
                    self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/traces":
                query = parse_qs(parsed.query)
                try:
                    matter_id = query.get("matter_id", [""])[0]
                    chat_id = query.get("chat_id", [""])[0]
                    limit = int(query.get("limit", ["100"])[0] or "100")
                    traces = list_product_traces(
                        trace_dir,
                        matter_id=matter_id,
                        chat_id=chat_id,
                        limit=limit,
                    )
                    matter_traces = list_product_traces(
                        trace_dir,
                        matter_id=matter_id,
                        limit=10_000,
                    ) if matter_id else traces
                    self.send_json(
                        {
                            "traces": traces,
                            "summary": summarize_trace_rows(traces),
                            "matter_summary": summarize_trace_rows(matter_traces),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - UI endpoint should return structured error.
                    self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            parsed = urlparse(self.path)
            if parsed.path not in {
                "/api/plan",
                "/api/rerun-plan",
                "/api/run",
                "/api/run-async",
                "/api/rerun",
                "/api/rerun-async",
                "/api/cancel-run",
                "/api/pick-path",
            }:
                self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self.read_json()
                if parsed.path == "/api/cancel-run":
                    self.send_json(cancel_run_job(str(payload.get("job_id") or "")))
                    return
                if parsed.path == "/api/pick-path":
                    self.send_json(
                        {
                            "paths": pick_local_paths(
                                mode=str(payload.get("mode") or "folder"),
                                initial_dir=str(payload.get("initial_dir") or ""),
                            )
                        }
                    )
                    return
                if parsed.path == "/api/plan":
                    self.send_json(
                        build_product_plan_preview(
                            objective=str(payload.get("objective", "")),
                            paths=parse_paths(payload.get("paths", [])),
                            max_files=parse_optional_int(payload.get("max_files")),
                            selected_paths=parse_paths(payload.get("selected_paths", [])) or None,
                            plan_note=str(payload.get("plan_note") or ""),
                            top_k=int(payload.get("top_k", 36) or 36),
                            config=config,
                            use_llm_planning=bool(payload.get("use_llm_planning", False)),
                        )
                    )
                    return
                if parsed.path == "/api/rerun-plan":
                    self.send_json(
                        rerun_plan_from_trace(
                            payload,
                            config=config,
                            trace_dir=trace_dir,
                        )
                    )
                    return
                if parsed.path == "/api/run-async":
                    self.send_json(
                        start_product_run_job(
                            payload,
                            mode="run",
                            config=config,
                            trace_dir=trace_dir,
                            output_dir=output_dir,
                        )
                    )
                    return
                if parsed.path == "/api/rerun-async":
                    self.send_json(
                        start_product_run_job(
                            payload,
                            mode="rerun",
                            config=config,
                            trace_dir=trace_dir,
                            output_dir=output_dir,
                        )
                    )
                    return
                if parsed.path == "/api/rerun":
                    response = rerun_from_trace(
                        payload,
                        config=config,
                        trace_dir=trace_dir,
                        output_dir=output_dir,
                    )
                    self.send_json(response)
                    return
                paths = parse_paths(payload.get("paths", []))
                matter_id = str(payload.get("matter_id", "matter"))
                result = run_product_matter(
                    objective=str(payload.get("objective", "")),
                    paths=paths,
                    matter_id=matter_id,
                    chat_id=str(payload.get("chat_id") or "main"),
                    conversation_history=payload.get("conversation_history"),
                    config=config,
                    trace_dir=trace_dir,
                    output_dir=output_dir,
                    live_synthesis=bool(payload.get("live_synthesis", False)),
                    top_k=int(payload.get("top_k", 36) or 36),
                    max_files=parse_optional_int(payload.get("max_files")),
                    selected_paths=parse_paths(payload.get("selected_paths", [])) or None,
                    pinned_paths=parse_paths(payload.get("pinned_paths", [])) or None,
                    plan_note=str(payload.get("plan_note") or ""),
                    use_llm_planning=bool(payload.get("use_llm_planning", False)),
                    verbose=False,
                )
                response = result.to_dict()
                full_trace = result.state.to_trace()
                response["summary"] = trace_summary(full_trace)
                response["trace"] = compact_trace_for_ui(full_trace)
                self.send_json(response)
            except Exception as exc:  # noqa: BLE001 - return actionable UI errors.
                self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            if not raw.strip():
                return {}
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("JSON request body must be an object")
            return parsed

        def send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_text(self, body: str, *, content_type: str) -> None:
            raw = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature.
            print("[UI]", format % args)

    return ProductUIHandler


def start_product_run_job(
    payload: dict[str, Any],
    *,
    mode: str,
    config: HarnessConfig,
    trace_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    job_id = f"job_{uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "events": [],
        "result": None,
        "error": None,
        "cancel_requested": False,
    }
    with RUN_JOBS_LOCK:
        RUN_JOBS[job_id] = job
        prune_run_jobs_locked()
    append_run_job_event(
        job_id,
        "RUN",
        "queued product matter run",
        mode=mode,
        summary="Queued your request.",
        next_step="Check the selected corpus.",
    )

    def event_callback(event: dict[str, Any]) -> None:
        append_run_job_event(job_id, event=event)

    def should_cancel() -> bool:
        with RUN_JOBS_LOCK:
            return bool((RUN_JOBS.get(job_id) or {}).get("cancel_requested"))

    def worker() -> None:
        try:
            if mode == "rerun":
                response = rerun_from_trace(
                    payload,
                    config=config,
                    trace_dir=trace_dir,
                    output_dir=output_dir,
                    event_callback=event_callback,
                    should_cancel=should_cancel,
                )
            else:
                paths = parse_paths(payload.get("paths", []))
                matter_id = str(payload.get("matter_id", "matter"))
                result = run_product_matter(
                    objective=str(payload.get("objective", "")),
                    paths=paths,
                    matter_id=matter_id,
                    chat_id=str(payload.get("chat_id") or "main"),
                    conversation_history=payload.get("conversation_history"),
                    config=config,
                    trace_dir=trace_dir,
                    output_dir=output_dir,
                    live_synthesis=bool(payload.get("live_synthesis", False)),
                    top_k=int(payload.get("top_k", 36) or 36),
                    max_files=parse_optional_int(payload.get("max_files")),
                    verbose=False,
                    event_callback=event_callback,
                    should_cancel=should_cancel,
                    selected_paths=parse_paths(payload.get("selected_paths", [])) or None,
                    pinned_paths=parse_paths(payload.get("pinned_paths", [])) or None,
                    plan_note=str(payload.get("plan_note") or ""),
                    use_llm_planning=bool(payload.get("use_llm_planning", False)),
                )
                response = result.to_dict()
                full_trace = result.state.to_trace()
                response["summary"] = trace_summary(full_trace)
                response["trace"] = compact_trace_for_ui(full_trace)
            append_run_job_event(job_id, "DONE", "product matter run completed")
            with RUN_JOBS_LOCK:
                RUN_JOBS[job_id]["status"] = "completed"
                RUN_JOBS[job_id]["result"] = response
        except ProductRunCancelled as exc:
            append_run_job_event(
                job_id,
                "STOP",
                "run stopped",
                summary="Stopped the run.",
                detail=str(exc),
            )
            with RUN_JOBS_LOCK:
                RUN_JOBS[job_id]["status"] = "canceled"
                RUN_JOBS[job_id]["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - async endpoint reports the normalized failure.
            error = f"{type(exc).__name__}: {exc}"
            with RUN_JOBS_LOCK:
                canceled = bool((RUN_JOBS.get(job_id) or {}).get("cancel_requested"))
            if canceled and "canceled" in str(exc).lower():
                append_run_job_event(
                    job_id,
                    "STOP",
                    "run stopped",
                    summary="Stopped the run.",
                    detail=str(exc),
                )
                with RUN_JOBS_LOCK:
                    RUN_JOBS[job_id]["status"] = "canceled"
                    RUN_JOBS[job_id]["error"] = str(exc)
                return
            append_run_job_event(
                job_id,
                "ERROR",
                "product matter run failed",
                error=error,
                summary="The run failed before completion.",
                next_step="Review the error, adjust the corpus or objective, then run again.",
            )
            with RUN_JOBS_LOCK:
                RUN_JOBS[job_id]["status"] = "failed"
                RUN_JOBS[job_id]["error"] = error

    threading.Thread(target=worker, name=f"irys-product-run-{job_id}", daemon=True).start()
    return snapshot_run_job(job_id)


def append_run_job_event(
    job_id: str,
    label: str | None = None,
    message: str | None = None,
    *,
    event: dict[str, Any] | None = None,
    **fields: Any,
) -> None:
    row = dict(event or {})
    if not row:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "label": label or "EVENT",
            "message": message or "",
            "fields": fields,
        }
    row.setdefault("ts", datetime.now(UTC).isoformat())
    row.setdefault("label", label or "EVENT")
    row.setdefault("message", message or "")
    row.setdefault("fields", fields)
    with RUN_JOBS_LOCK:
        job = RUN_JOBS.get(job_id)
        if job is not None:
            job.setdefault("events", []).append(row)


def snapshot_run_job(job_id: str) -> dict[str, Any]:
    with RUN_JOBS_LOCK:
        job = RUN_JOBS.get(job_id)
        if job is None:
            raise ValueError("run job not found")
        return {
            "job_id": job["job_id"],
            "mode": job["mode"],
            "status": job["status"],
            "started_at": job["started_at"],
            "events": list(job.get("events", [])),
            "result": job.get("result"),
            "error": job.get("error"),
            "cancel_requested": bool(job.get("cancel_requested")),
        }


def cancel_run_job(job_id: str) -> dict[str, Any]:
    if not job_id:
        raise ValueError("job_id is required")
    with RUN_JOBS_LOCK:
        job = RUN_JOBS.get(job_id)
        if job is None:
            raise ValueError("run job not found")
        if job.get("status") != "running":
            return {
                "job_id": job["job_id"],
                "mode": job["mode"],
                "status": job["status"],
                "started_at": job["started_at"],
                "events": list(job.get("events", [])),
                "result": job.get("result"),
                "error": job.get("error"),
                "cancel_requested": bool(job.get("cancel_requested")),
            }
        job["cancel_requested"] = True
    append_run_job_event(
        job_id,
        "STOP",
        "stop requested",
        summary="Stop requested. The current document or model call may need to finish first.",
    )
    return snapshot_run_job(job_id)


def prune_run_jobs_locked(*, keep: int = 50) -> None:
    if len(RUN_JOBS) <= keep:
        return
    sorted_jobs = sorted(RUN_JOBS.items(), key=lambda item: str(item[1].get("started_at") or ""))
    for job_id, _ in sorted_jobs[: max(0, len(sorted_jobs) - keep)]:
        RUN_JOBS.pop(job_id, None)


def rerun_from_trace(
    payload: dict[str, Any],
    *,
    config: HarnessConfig,
    trace_dir: str | Path,
    output_dir: str | Path,
    event_callback: Any = None,
    should_cancel: Any = None,
) -> dict[str, Any]:
    parent_path = resolve_trace_path(str(payload.get("trace_path") or ""), trace_dir)
    parent = load_trace(parent_path)
    task = parent.get("task") or {}
    metadata = task.get("metadata") or {}
    original_objective = str(task.get("question") or "")
    nudge = str(payload.get("nudge") or "").strip()
    if not nudge:
        raise ValueError("nudge is required")
    paths = rerun_context_paths(parent, payload)
    if not paths:
        raise ValueError("parent trace does not contain context_files")
    base_matter_id = str(metadata.get("matter_id") or task.get("task_id") or "matter")
    chat_id = str(payload.get("chat_id") or metadata.get("chat_id") or "main")
    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    matter_id = sanitize_matter_id(f"{base_matter_id}-nudge-{suffix}")
    objective = f"{original_objective}\n\nUser steering note: {nudge}"
    plan_note = build_rerun_plan_note(payload.get("plan_note"), nudge)
    result = run_product_matter(
        objective=objective,
        paths=paths,
        matter_id=matter_id,
        chat_id=chat_id,
        conversation_history=conversation_history_for_rerun(parent),
        config=config,
        trace_dir=trace_dir,
        output_dir=output_dir,
        live_synthesis=bool(payload.get("live_synthesis", False)),
        top_k=int(payload.get("top_k", 36) or 36),
        max_files=parse_optional_int(payload.get("max_files")),
        verbose=False,
        parent_trace_path=str(parent_path),
        user_nudge=nudge,
        selected_paths=selected_paths_for_rerun_payload(payload),
        pinned_paths=parse_paths(payload.get("pinned_paths", [])) or None,
        plan_note=plan_note,
        use_llm_planning=bool(payload.get("use_llm_planning", False)),
        event_callback=event_callback,
        should_cancel=should_cancel,
    )
    child_trace = result.state.to_trace()
    response = result.to_dict()
    response["summary"] = trace_summary(child_trace)
    response["trace"] = compact_trace_for_ui(child_trace)
    response["comparison"] = compare_product_traces(parent, child_trace)
    return response


def rerun_plan_from_trace(
    payload: dict[str, Any],
    *,
    config: HarnessConfig,
    trace_dir: str | Path,
) -> dict[str, Any]:
    parent_path = resolve_trace_path(str(payload.get("trace_path") or ""), trace_dir)
    parent = load_trace(parent_path)
    task = parent.get("task") or {}
    metadata = task.get("metadata") or {}
    original_objective = str(task.get("question") or "")
    nudge = str(payload.get("nudge") or "").strip()
    if not nudge:
        raise ValueError("nudge is required")
    paths = rerun_context_paths(parent, payload)
    if not paths:
        raise ValueError("parent trace does not contain context_files")
    objective = f"{original_objective}\n\nUser steering note: {nudge}"
    return build_product_plan_preview(
        objective=objective,
        paths=paths,
        max_files=parse_optional_int(payload.get("max_files")),
        selected_paths=selected_paths_for_rerun_payload(payload),
        plan_note=build_rerun_plan_note(payload.get("plan_note"), nudge),
        top_k=int(payload.get("top_k", 36) or 36),
        config=config,
        use_llm_planning=bool(payload.get("use_llm_planning", False)),
    )


def selected_paths_for_rerun_payload(payload: dict[str, Any]) -> list[str] | None:
    selected_paths = parse_paths(payload.get("selected_paths", []))
    if selected_paths and bool(payload.get("selected_paths_locked", False)):
        return filter_excluded_paths(selected_paths, payload)
    return None


def rerun_context_paths(parent: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    task = parent.get("task") or {}
    metadata = task.get("metadata") or {}
    paths = [
        str(item)
        for item in (metadata.get("discovered_context_files") or task.get("context_files", []))
        if str(item).strip()
    ]
    paths.extend(parse_paths(payload.get("paths", [])))
    return filter_excluded_paths(paths, payload)


def filter_excluded_paths(paths: list[str], payload: dict[str, Any]) -> list[str]:
    excluded = {normalized_path_key(path) for path in parse_paths(payload.get("excluded_paths", []))}
    output: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        key = normalized_path_key(raw_path)
        if not key or key in excluded or key in seen:
            continue
        output.append(raw_path)
        seen.add(key)
    return output


def normalized_path_key(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    return str(Path(raw).expanduser().resolve()).lower()


def build_rerun_plan_note(raw_plan_note: Any, nudge: str) -> str:
    parts = []
    plan_note = str(raw_plan_note or "").strip()
    if plan_note:
        parts.append(plan_note)
    steering_note = f"Current steering note: {nudge.strip()}"
    if steering_note not in parts:
        parts.append(steering_note)
    return "\n\n".join(parts)


def parse_paths(raw_paths: Any) -> list[str]:
    if isinstance(raw_paths, str):
        return [line.strip() for line in raw_paths.splitlines() if line.strip()]
    return [str(item).strip() for item in raw_paths or [] if str(item).strip()]


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def pick_local_paths(*, mode: str = "folder", initial_dir: str = "") -> list[str]:
    normalized = mode.strip().lower()
    if normalized not in {"file", "files", "folder"}:
        raise ValueError("mode must be file, files, or folder")

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        options: dict[str, Any] = {"title": "Select corpus path"}
        initial = Path(initial_dir).expanduser() if initial_dir.strip() else None
        if initial and initial.exists():
            options["initialdir"] = str(initial if initial.is_dir() else initial.parent)

        if normalized == "folder":
            selected = filedialog.askdirectory(**options)
            return [str(Path(selected).resolve())] if selected else []
        if normalized == "files":
            selected_files = filedialog.askopenfilenames(**options)
            return [str(Path(path).resolve()) for path in selected_files]
        selected_file = filedialog.askopenfilename(**options)
        return [str(Path(selected_file).resolve())] if selected_file else []
    finally:
        root.destroy()


def conversation_history_for_rerun(parent: dict[str, Any]) -> list[dict[str, str]]:
    packet = parent.get("final_packet") or {}
    history = [
        {"user": str(item.get("user") or ""), "assistant": str(item.get("assistant") or "")}
        for item in packet.get("conversation_history", []) or []
        if isinstance(item, dict)
    ]
    task = parent.get("task") or {}
    question = str(task.get("question") or "").strip()
    answer = str(parent.get("rendered_answer") or "").strip()
    if question or answer:
        history.append({"user": question, "assistant": answer})
    return history


def comparison_from_parent(trace: dict[str, Any], *, trace_dir: str | Path) -> dict[str, Any] | None:
    metadata = ((trace.get("task") or {}).get("metadata") or {})
    parent_trace_path = str(metadata.get("parent_trace_path") or "").strip()
    if not parent_trace_path:
        return None
    try:
        parent = load_trace(resolve_trace_path(parent_trace_path, trace_dir))
        return compare_product_traces(parent, trace)
    except Exception as exc:  # noqa: BLE001 - trace loading should stay usable if comparison fails.
        return {
            "mode": "product_trace_comparison",
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }


def list_product_traces(
    trace_dir: str | Path,
    *,
    matter_id: str = "",
    chat_id: str = "",
    limit: int = 100,
) -> list[dict[str, Any]]:
    root = Path(trace_dir).resolve()
    product_dir = root / "product_matter"
    if not product_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(product_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            trace = load_trace(path)
        except Exception:  # noqa: BLE001 - ignore broken trace files in the browser list.
            continue
        task = trace.get("task") or {}
        metadata = task.get("metadata") or {}
        if matter_id and str(metadata.get("matter_id") or "") != matter_id:
            continue
        if chat_id and str(metadata.get("chat_id") or "") != chat_id:
            continue
        rows.append(
            {
                "path": str(path),
                "run_id": trace.get("run_id"),
                "task_id": trace.get("task_id"),
                "matter_id": metadata.get("matter_id"),
                "chat_id": metadata.get("chat_id"),
                "started_at": trace.get("started_at"),
                "question": task.get("question"),
                "estimated_cost": (trace.get("metrics") or {}).get("estimated_cost"),
                "total_tokens": (trace.get("metrics") or {}).get("total_tokens"),
            }
        )
        if len(rows) >= max(1, limit):
            break
    return rows


def summarize_trace_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chat_ids = {str(row.get("chat_id") or "main") for row in rows}
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    estimated_cost = sum(float(row.get("estimated_cost") or 0.0) for row in rows)
    return {
        "messages": len(rows),
        "chats": len(chat_ids),
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "average_cost_per_message": estimated_cost / len(rows) if rows else 0.0,
    }


def compact_trace_for_ui(trace: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: compact_value_for_ui(value)
        for key, value in trace.items()
        if key not in {"chunks", "final_packet", "extraction_records", "critic_records", "verification_records"}
    }
    chunks = trace.get("chunks") if isinstance(trace.get("chunks"), list) else []
    compact["chunks"] = [compact_chunk_for_ui(chunk) for chunk in chunks]
    if "final_packet" in trace:
        compact["final_packet"] = compact_value_for_ui(trace.get("final_packet"), max_str=5000, max_list=260)
    for key in ["extraction_records", "critic_records", "verification_records"]:
        if key in trace:
            compact[key] = compact_value_for_ui(trace.get(key), max_str=5000, max_list=120)
    compact["_ui_compaction"] = {
        "compacted": True,
        "original_chunk_count": len(chunks),
        "policy": "Full traces remain on disk; browser responses keep metadata and trim large text fields.",
    }
    return compact


def compact_chunk_for_ui(chunk: Any) -> dict[str, Any]:
    if not isinstance(chunk, dict):
        return {"value": compact_value_for_ui(chunk, max_str=700)}
    row = {key: compact_value_for_ui(value, max_str=700) for key, value in chunk.items() if key != "text"}
    text = str(chunk.get("text") or "")
    if text:
        row["text_preview"] = compact_text_for_ui(text, max_chars=700)
        row["text_chars"] = len(text)
    return row


def compact_value_for_ui(value: Any, *, max_str: int = 4000, max_list: int = 500, depth: int = 0) -> Any:
    if isinstance(value, str):
        return compact_text_for_ui(value, max_chars=max_str)
    if isinstance(value, list):
        items = [
            compact_value_for_ui(item, max_str=max_str, max_list=max_list, depth=depth + 1)
            for item in value[:max_list]
        ]
        if len(value) > max_list:
            items.append({"_ui_omitted_items": len(value) - max_list})
        return items
    if isinstance(value, dict):
        if depth >= 8:
            return compact_text_for_ui(json.dumps(value, sort_keys=True, default=str), max_chars=max_str)
        return {
            str(key): compact_value_for_ui(item, max_str=max_str, max_list=max_list, depth=depth + 1)
            for key, item in value.items()
        }
    return value


def compact_text_for_ui(value: str, *, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + f"\n\n[UI truncated {len(text) - (max_chars - 80)} character(s); full text is saved in the trace file.]"


def resolve_trace_path(raw_path: str, trace_dir: str | Path) -> Path:
    root = Path(trace_dir).resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("trace path must be under the configured trace directory")
    return resolved


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Irys Matter Runner</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #5e6b78;
      --line: #d8dee6;
      --panel: #f7f8fb;
      --accent: #126b5f;
      --accent-2: #8d3f1f;
      --bg: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding: 0 18px;
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 20;
    }
    h1 {
      font-size: 18px;
      font-weight: 700;
      margin: 0;
      letter-spacing: 0;
    }
    .command-bar {
      min-height: 56px;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 14px;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding: 8px 16px;
      background: #f8fafb;
      position: sticky;
      top: 56px;
      z-index: 18;
    }
    .command-bar strong {
      display: block;
      font-size: 13px;
      margin-bottom: 2px;
    }
    .command-bar small {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .action-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }
    .action-pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: #2f3b46;
      padding: 3px 8px;
      font-size: 12px;
      line-height: 1.2;
    }
    .action-pill.important {
      border-color: #9fc2e8;
      background: #f2f8ff;
    }
    .action-pill.warn {
      border-color: #d8c1ad;
      background: #fff8f0;
    }
    .command-actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }
    .command-actions button {
      padding: 8px 10px;
      font-size: 12px;
    }
    .control-dock {
      display: grid;
      grid-template-columns: minmax(220px, 0.8fr) minmax(360px, 1.2fr) auto;
      gap: 12px;
      align-items: start;
      border-bottom: 1px solid var(--line);
      padding: 10px 16px;
      background: #ffffff;
      position: sticky;
      top: 113px;
      z-index: 17;
      box-shadow: 0 2px 12px rgba(23, 32, 42, 0.06);
    }
    .control-dock h2 {
      margin-bottom: 6px;
    }
    .control-dock small {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .control-dock textarea {
      min-height: 58px;
      max-height: 132px;
      resize: vertical;
    }
    .control-dock-actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(130px, 1fr));
      gap: 8px;
      min-width: 300px;
    }
    .control-dock-actions button {
      padding: 8px 10px;
      font-size: 12px;
    }
    .dock-summary {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafb;
      padding: 8px;
      min-height: 58px;
    }
    .dock-summary strong {
      display: block;
      font-size: 13px;
      margin-bottom: 3px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(420px, 1fr) minmax(360px, 0.85fr);
      min-height: calc(100vh - 112px);
    }
    section {
      min-width: 0;
      border-right: 1px solid var(--line);
      padding: 16px;
      overflow: auto;
    }
    section:last-child { border-right: 0; }
    h2 {
      margin: 0 0 12px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      font-size: 12px;
      font-weight: 700;
      color: #2f3b46;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    textarea { min-height: 116px; resize: vertical; }
    .objective { min-height: 190px; }
    .objective.compact { min-height: 74px; }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    .row > * { flex: 1 1 140px; }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #eef2f5;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .toggle input { width: auto; }
    .candidate {
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      margin: 0;
      font-size: 13px;
      font-weight: 400;
      color: var(--ink);
    }
    .candidate input { width: auto; margin-top: 2px; }
    .candidate strong {
      display: block;
      font-size: 13px;
      margin-bottom: 3px;
      overflow-wrap: anywhere;
    }
    .candidate small {
      display: block;
      color: var(--muted);
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .status {
      color: var(--accent-2);
      font-size: 13px;
      min-height: 18px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 58px;
    }
    .metric b { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .metric span { font-size: 18px; font-weight: 700; }
    .list {
      display: grid;
      gap: 8px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 10px;
      overflow-wrap: anywhere;
    }
    .item small {
      display: block;
      color: var(--muted);
      margin-top: 4px;
    }
    .brief-card small {
      color: #2f3b46;
      font-size: 13px;
      line-height: 1.5;
    }
    .audit-board {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .audit-card {
      border: 1px solid var(--line);
      border-left: 4px solid #9aa6b2;
      border-radius: 6px;
      background: #fff;
      padding: 10px;
      min-height: 112px;
      overflow-wrap: anywhere;
    }
    .audit-card.good { border-left-color: #2f8f5b; background: #f6fbf7; }
    .audit-card.warn { border-left-color: #b36b2c; background: #fff8f0; }
    .audit-card.bad { border-left-color: #bf3f3f; background: #fff6f6; }
    .audit-card b {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .audit-card strong {
      display: block;
      font-size: 15px;
      margin-bottom: 6px;
    }
    .audit-card small {
      display: block;
      color: #2f3b46;
      font-size: 12px;
      line-height: 1.45;
    }
    .investigation-map {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .map-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 10px;
      min-height: 142px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      overflow-wrap: anywhere;
    }
    .map-card b {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .map-card strong {
      display: block;
      font-size: 15px;
      line-height: 1.25;
    }
    .map-card small {
      color: #2f3b46;
      font-size: 12px;
      line-height: 1.45;
    }
    .map-card button {
      margin-top: auto;
      width: 100%;
      padding: 8px 9px;
      font-size: 12px;
    }
    .action-board {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0 14px;
    }
    .action-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 10px;
      min-height: 138px;
      display: flex;
      flex-direction: column;
      gap: 7px;
      overflow-wrap: anywhere;
    }
    .action-card.ready {
      border-color: #b8d8c2;
      background: #f6fbf7;
    }
    .action-card.next {
      border-color: #9fc2e8;
      background: #f2f8ff;
    }
    .action-card.warn {
      border-color: #d8c1ad;
      background: #fff8f0;
    }
    .action-card b {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .action-card strong {
      font-size: 15px;
      line-height: 1.25;
    }
    .action-card small {
      color: #2f3b46;
      font-size: 12px;
      line-height: 1.4;
    }
    .action-card button {
      margin-top: auto;
      width: 100%;
      padding: 8px 9px;
      font-size: 12px;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      margin-top: 6px;
    }
    .guide-steps {
      margin: 8px 0 0 20px;
      padding: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .guide-steps li { margin: 4px 0; }
    .timeline {
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }
    .timeline-step {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
    }
    .timeline-step b {
      display: block;
      color: var(--ink);
      font-size: 13px;
      margin-bottom: 2px;
    }
    .timeline-step.done {
      border-color: #b8d8c2;
      background: #f3fbf5;
    }
    .timeline-step.active {
      border-color: #9fc2e8;
      background: #f2f8ff;
    }
    .timeline-step.error {
      border-color: #e1a1a1;
      background: #fff5f5;
    }
    .control-help {
      margin-top: 10px;
      background: #f7f9fb;
    }
    .workspace-details {
      margin-top: 14px;
    }
    .workspace-details[open] summary {
      margin-bottom: 10px;
    }
    .steering-panel {
      position: sticky;
      top: 0;
      z-index: 4;
      margin-bottom: 12px;
      border: 1px solid #bfd4de;
      border-radius: 6px;
      padding: 10px;
      background: #f8fbfd;
      box-shadow: 0 2px 10px rgba(23, 32, 42, 0.08);
    }
    .steering-panel label:first-child { margin-top: 0; }
    .next-pass-setup {
      margin-top: 10px;
      background: #fff;
    }
    .next-pass-setup .row {
      margin-top: 8px;
    }
    .empty {
      color: var(--muted);
      font-size: 13px;
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }
    summary {
      cursor: pointer;
      font-weight: 700;
      color: #2f3b46;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: Consolas, Monaco, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .answer {
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 320px;
      padding: 14px;
      background: var(--panel);
    }
    .answer h1, .answer h2, .answer h3 {
      margin: 14px 0 8px;
      color: var(--ink);
      text-transform: none;
    }
    .answer h1 { font-size: 20px; }
    .answer h2 { font-size: 17px; }
    .answer h3 { font-size: 15px; }
    .answer p { margin: 0 0 10px; line-height: 1.55; }
    .answer ul, .answer ol { margin: 0 0 12px 22px; padding: 0; }
    .answer li { margin: 5px 0; line-height: 1.5; }
    .answer code {
      background: #e9edf2;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
      font-family: Consolas, Monaco, monospace;
      font-size: 12px;
    }
    @media (max-width: 1100px) {
      .command-bar { grid-template-columns: 1fr; position: static; }
      .command-actions { justify-content: stretch; }
      .command-actions button { flex: 1 1 140px; }
      .control-dock { grid-template-columns: 1fr; position: static; }
      .control-dock-actions { grid-template-columns: 1fr; min-width: 0; }
      main { grid-template-columns: 1fr; }
      section { border-right: 0; border-bottom: 1px solid var(--line); }
      .action-board { grid-template-columns: 1fr; }
      .investigation-map { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Irys Matter Runner</h1>
    <div class="status" id="status"></div>
  </header>
  <div class="command-bar">
    <div>
      <strong id="commandStepTitle">Ready</strong>
      <small id="commandStepDetail">Choose a corpus, ask a question, review the source plan, then run.</small>
      <div class="action-summary" id="nextPassSummary"></div>
    </div>
    <div class="command-actions">
      <button class="secondary" id="topPlan">Review Source Plan</button>
      <button id="topRun">Run Approved Plan</button>
      <button class="secondary" id="topStop" disabled>Stop</button>
      <button class="secondary" id="topPreviewNudge">Preview Steering</button>
      <button class="secondary" id="topApplyNudge">Run Corrected Pass</button>
    </div>
  </div>
  <div class="control-dock" id="controlDock">
    <div class="dock-summary" id="quickDockSummary">
      <strong>Ready</strong>
      <small>Choose a corpus, ask a question, review the plan, then run.</small>
    </div>
    <div>
      <h2>Correction Or Steering</h2>
      <textarea id="quickInstruction" placeholder="Write the correction here. Before a run, use it to fix the source plan. After a run, use it to steer the next pass."></textarea>
      <small>Use this dock when the plan is wrong, a source should be pinned or ignored, or Irys needs to read a document more deeply.</small>
    </div>
    <div class="control-dock-actions">
      <button class="secondary" id="quickApplyPlan">Apply To Source Plan</button>
      <button class="secondary" id="quickPreviewNudge">Preview Next Pass</button>
      <button id="quickRunNudge">Run Corrected Pass</button>
      <button class="secondary" id="quickReviewSources">Review Sources</button>
      <button class="secondary" id="quickReviewAnswer">Review Answer</button>
      <button class="secondary" id="quickStop" disabled>Stop</button>
    </div>
  </div>
  <main>
    <section>
      <h2>Matter</h2>
      <div class="item">
        <strong>Workflow</strong>
        <ol class="guide-steps">
          <li>Choose a folder or files for this matter.</li>
          <li>Ask the question or describe the deliverable.</li>
          <li>Review the proposed first-read documents.</li>
          <li>Run, audit the sources, then steer the next pass if needed.</li>
        </ol>
      </div>
      <details class="control-help">
        <summary>What the controls mean</summary>
        <small>
          Folders are read recursively. No artificial file-count or per-document character cap is applied to local corpus paths. Matter ID groups saved runs and costs. Chat ID keeps separate conversations inside the same matter. Smart source planning reviews the file inventory before the first read and falls back to path scoring if model planning is unavailable. Draft final answer is on by default and calls the configured drafting model; turn it off only for a cheap dry run. Evidence passages controls how many retrieved source passages are used for the answer packet; the default favors source coverage over minimal token use. Source actions let you read a document more deeply, pin it into synthesis, or hold it back on the next pass. Current Run Cost is the loaded run; Matter Cost totals saved traces for the matter.
        </small>
      </details>
      <label for="matter">Matter ID</label>
      <input id="matter" value="local-matter" />
      <label for="chat">Chat ID</label>
      <input id="chat" value="main" />
      <label for="paths">Corpus Paths</label>
      <textarea id="paths" spellcheck="false"></textarea>
      <div class="hint">Paste local file or folder paths one per line, or use the picker buttons.</div>
      <div class="row">
        <button class="secondary" id="chooseFolder">Choose Folder</button>
        <button class="secondary" id="chooseFile">Choose File</button>
        <button class="secondary" id="chooseFiles">Choose Files</button>
      </div>
      <div class="row">
        <label class="toggle"><input id="usePlanner" type="checkbox" checked /> Smart source planning</label>
        <label class="toggle"><input id="live" type="checkbox" checked /> Draft final answer</label>
        <label for="topk">Evidence passages</label>
        <input id="topk" type="number" min="1" max="200" value="36" />
      </div>
      <div class="row">
        <button id="run">Run Approved Plan</button>
        <button class="secondary" id="stopRun" disabled>Stop Run</button>
        <button class="secondary" id="clear">Clear</button>
      </div>
    </section>
    <section>
      <h2>Question Or Work Product</h2>
      <textarea id="objective" class="objective" placeholder="Ask the question or describe the work product you want from this matter."></textarea>
      <div class="hint">For a follow-up, keep the same Matter ID and Chat ID, replace the question, and run again. Prior questions and final answers are used for continuity; intermediate trace details are not used as chat history.</div>
      <div class="row">
        <button class="secondary" id="planRun">Review Source Plan</button>
      </div>
      <h2 style="margin-top:16px">Next Best Action</h2>
      <div class="action-board" id="actionBoard">
        <div class="empty">Choose a corpus and objective. Irys will show the next useful action here.</div>
      </div>
      <h2 style="margin-top:16px">Run Brief</h2>
      <div class="list" id="runBrief">
        <div class="empty">The brief will explain the current plan, what Irys is doing, and what you can change next.</div>
      </div>
      <h2 style="margin-top:16px">Review Checklist</h2>
      <div class="audit-board" id="reviewChecklist">
        <div class="empty">After a plan or run, this checklist will show whether the answer has enough source support, what still needs review, and what to do next.</div>
      </div>
      <h2 style="margin-top:16px">Investigation Map</h2>
      <div class="investigation-map" id="investigationMap">
        <div class="empty">After planning or running, this will show the question, source choices, evidence coverage, worker review, held-back documents, and next correction path.</div>
      </div>
      <details class="workspace-details" id="sourcePlanDetails" open>
        <summary>Review And Edit Source Plan</summary>
        <div class="hint">This is where you correct Irys before it spends time reading. Uncheck irrelevant documents, add missing first-read paths, or write a plan correction and preview again.</div>
        <h2 style="margin-top:16px">First-Read Review</h2>
        <div class="list" id="candidateReview">
          <div class="empty">Review Source Plan will show the highest-ranked documents here. Check or uncheck what Irys should read first.</div>
        </div>
        <div class="row">
          <button class="secondary" id="applyCheckedCandidates">Apply Checked Documents</button>
          <button class="secondary" id="selectAllCandidates">Select All Shown</button>
          <button class="secondary" id="restoreRecommendedCandidates">Restore Recommended</button>
        </div>
        <label for="firstReadPaths">First-Read Documents</label>
        <textarea id="firstReadPaths" placeholder="Review Source Plan fills this with the files Irys intends to read first. Edit it before running if the plan is wrong."></textarea>
        <label for="planNote">Plan Correction</label>
        <textarea id="planNote" placeholder="Correct the plan here: focus on 10-Ks, start with the latest amendment, ignore draft folders, include quarterly reports, compare all agreements, or answer only the narrow question."></textarea>
        <div class="hint">Preview Corrected Plan re-ranks the corpus using this correction. Run Approved Plan uses the current first-read documents and correction.</div>
        <div class="row">
          <button class="secondary" id="applyPlanCorrection">Preview Corrected Plan</button>
        </div>
        <h2 style="margin-top:16px">Detailed Plan</h2>
        <div class="list" id="planPreview">
          <div class="empty">Choose a corpus and objective, then review the source plan before running.</div>
        </div>
      </details>
      <div class="metric-grid">
        <div class="metric"><b>Documents</b><span id="docCount">0</span></div>
        <div class="metric"><b>Source Passages</b><span id="chunkCount">0</span></div>
        <div class="metric"><b>Tokens</b><span id="tokens">0</span></div>
        <div class="metric"><b>Current Run Cost</b><span id="cost">$0.00</span></div>
        <div class="metric"><b>Matter Cost</b><span id="matterCost">$0.00</span></div>
        <div class="metric"><b>Matter Runs</b><span id="matterMessages">0</span></div>
      </div>
      <h2>Answer</h2>
      <div class="answer" id="answer"></div>
      <h2 style="margin-top:16px">Source Review</h2>
      <div class="list" id="sourceSummary"></div>
      <h2 style="margin-top:16px">Pinned For Next Pass</h2>
      <div class="list" id="pinnedSources"></div>
      <h2 style="margin-top:16px">Documents Held Back</h2>
      <div class="list" id="heldBackSources"></div>
      <h2 style="margin-top:16px">Sources Used</h2>
      <div class="list" id="sourcesUsed"></div>
      <h2 style="margin-top:16px">Open Questions</h2>
      <div class="list" id="openQuestions"></div>
      <h2 style="margin-top:16px">Conversation History</h2>
      <div class="list" id="chatHistory"></div>
    </section>
    <section>
      <h2>Steering</h2>
      <div class="steering-panel">
        <label for="nudge">Correct Or Steer The Next Pass</label>
        <textarea id="nudge" placeholder="Example: focus on the 2024 10-K only, ignore Form 4s, compare against the latest amendment, read the guaranty more closely, or answer only the EPS question."></textarea>
        <div class="hint">This stays available while a run is working. Preview shows how the next pass re-plans which files to read; Run Corrected Pass reruns the same matter with your correction.</div>
        <label for="rerunPaths">Additional Corpus Paths</label>
        <textarea id="rerunPaths" placeholder="Optional: add another local file or folder path for the next pass."></textarea>
        <div class="row">
          <button class="secondary" id="chooseRerunFolder">Choose Folder</button>
          <button class="secondary" id="chooseRerunFile">Choose File</button>
          <button class="secondary" id="chooseRerunFiles">Choose Files</button>
        </div>
        <div class="row">
          <button class="secondary" id="previewNudgePlan">Preview Steering Plan</button>
          <button class="secondary" id="rerunTrace">Run Corrected Pass</button>
        </div>
        <div class="item next-pass-setup" id="nextPassSetup">
          <strong>Next Pass Setup</strong>
          <small>No corrections selected yet.</small>
        </div>
      </div>
      <h2 style="margin-top:16px">Current Work</h2>
      <div class="item" id="currentStep"><strong>Idle</strong><small>No run has started.</small></div>
      <div class="timeline" id="runTimeline"></div>
      <h2 style="margin-top:16px">What Irys Is Doing</h2>
      <div class="list" id="liveEvents"></div>
      <h2 style="margin-top:16px">Run Health</h2>
      <div class="list" id="diagnosis"></div>
      <h2 style="margin-top:16px">Saved Runs</h2>
      <div class="list" id="traceList"></div>
      <details style="margin-top:16px">
        <summary>Advanced run details</summary>
        <label for="tracepath">Saved Run Path</label>
        <input id="tracepath" />
        <div class="row">
          <button class="secondary" id="loadTrace">Load Saved Run</button>
          <button class="secondary" id="listTraces">List Saved Runs</button>
        </div>
      </details>
      <h2 style="margin-top:16px">Run Comparison</h2>
      <div class="list" id="comparison"></div>
      <details style="margin-top:16px">
        <summary>Advanced diagnostic data</summary>
        <h2 style="margin-top:16px">Raw Events</h2>
        <div class="list" id="events"></div>
        <h2 style="margin-top:16px">Document Inventory</h2>
        <div class="list" id="documents"></div>
        <h2 style="margin-top:16px">Artifacts</h2>
        <div class="list" id="artifacts"></div>
        <h2 style="margin-top:16px">Evidence Records</h2>
        <div class="list" id="evidence"></div>
        <h2 style="margin-top:16px">Answer Source Map</h2>
        <div class="list" id="answerSources"></div>
      </details>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const status = $("status");
    const run = $("run");
    const planRun = $("planRun");
    const applyPlanCorrection = $("applyPlanCorrection");
    const stopRun = $("stopRun");
    const topPlan = $("topPlan");
    const topRun = $("topRun");
    const topStop = $("topStop");
    const topPreviewNudge = $("topPreviewNudge");
    const topApplyNudge = $("topApplyNudge");
    const quickInstruction = $("quickInstruction");
    const quickApplyPlan = $("quickApplyPlan");
    const quickPreviewNudge = $("quickPreviewNudge");
    const quickRunNudge = $("quickRunNudge");
    const quickReviewSources = $("quickReviewSources");
    const quickReviewAnswer = $("quickReviewAnswer");
    const quickStop = $("quickStop");
    const loadTrace = $("loadTrace");
    const rerunTrace = $("rerunTrace");
    const previewNudgePlan = $("previewNudgePlan");
    const listTraces = $("listTraces");
    const chooseFolder = $("chooseFolder");
    const chooseFile = $("chooseFile");
    const chooseFiles = $("chooseFiles");
    const chooseRerunFolder = $("chooseRerunFolder");
    const chooseRerunFile = $("chooseRerunFile");
    const chooseRerunFiles = $("chooseRerunFiles");
    const applyCheckedCandidates = $("applyCheckedCandidates");
    const selectAllCandidates = $("selectAllCandidates");
    const restoreRecommendedCandidates = $("restoreRecommendedCandidates");
    let conversationByChat = {};
    let activeJobPoll = null;
    let activeJobId = "";
    let currentPlan = null;
    let currentPlanNote = "";
    let currentPlanObjective = "";
    let currentPlanMode = "";
    let currentPlanPathKey = "";
    let firstReadPathsDirty = false;
    let suppressFirstReadDirty = false;
    let excludedSourcePaths = [];
    let pinnedSourcePaths = [];
    let lastLiveEvents = [];
    let lastRenderedTrace = null;
    topPlan.addEventListener("click", () => planRun.click());
    topRun.addEventListener("click", () => run.click());
    topStop.addEventListener("click", () => stopRun.click());
    topPreviewNudge.addEventListener("click", () => previewNudgePlan.click());
    topApplyNudge.addEventListener("click", () => rerunTrace.click());
    quickApplyPlan.addEventListener("click", () => {
      const text = quickInstruction.value.trim();
      if (!text) {
        quickInstruction.focus();
        status.textContent = "Write the plan correction first";
        return;
      }
      appendInstructionToTextarea("planNote", text);
      setSourcePlanOpen(true);
      $("sourcePlanDetails").scrollIntoView({behavior: "smooth", block: "start"});
      applyPlanCorrection.click();
    });
    quickPreviewNudge.addEventListener("click", () => {
      const text = quickInstruction.value.trim();
      if (text) appendInstructionToTextarea("nudge", text);
      if (!hasLoadedRunForCorrection()) {
        status.textContent = "Load or finish a run before previewing a corrected next pass";
        $("sourcePlanDetails").scrollIntoView({behavior: "smooth", block: "start"});
        renderNextPassSetup();
        return;
      }
      previewNudgePlan.click();
    });
    quickRunNudge.addEventListener("click", () => {
      const text = quickInstruction.value.trim();
      if (text) appendInstructionToTextarea("nudge", text);
      if (!hasLoadedRunForCorrection()) {
        status.textContent = "Load or finish a run before running a corrected pass";
        renderNextPassSetup();
        return;
      }
      rerunTrace.click();
    });
    quickReviewSources.addEventListener("click", () => $("sourceSummary").scrollIntoView({behavior: "smooth", block: "start"}));
    quickReviewAnswer.addEventListener("click", () => $("answer").scrollIntoView({behavior: "smooth", block: "start"}));
    quickStop.addEventListener("click", () => stopRun.click());
    syncCommandButtons();
    updateQuickDock();
    renderNextPassSetup();
    renderActionBoard();
    $("clear").addEventListener("click", () => {
      $("objective").value = "";
      $("objective").classList.remove("compact");
      $("answer").innerHTML = "";
      $("chatHistory").innerHTML = "";
      $("tracepath").value = "";
      $("nudge").value = "";
      quickInstruction.value = "";
      $("rerunPaths").value = "";
      setFirstReadPaths([], {dirty: false});
      $("planNote").value = "";
      $("planPreview").innerHTML = emptyState("Choose a corpus and objective, then review the plan before running.");
      $("runBrief").innerHTML = emptyState("The brief will explain the current plan, what Irys is doing, and what you can change next.");
      $("reviewChecklist").innerHTML = emptyState("After a plan or run, this checklist will show whether the answer has enough source support, what still needs review, and what to do next.");
      $("investigationMap").innerHTML = emptyState("After planning or running, this will show the question, source choices, evidence coverage, worker review, held-back documents, and next correction path.");
      $("candidateReview").innerHTML = emptyState("Review Plan will show the highest-ranked documents here. Check or uncheck what Irys should read first.");
      setSourcePlanOpen(true);
      currentPlan = null;
      currentPlanNote = "";
      currentPlanObjective = "";
      currentPlanMode = "";
      currentPlanPathKey = "";
      excludedSourcePaths = [];
      pinnedSourcePaths = [];
      $("diagnosis").innerHTML = "";
      $("currentStep").innerHTML = "<strong>Idle</strong><small>No run has started.</small>";
      $("runTimeline").innerHTML = "";
      $("liveEvents").innerHTML = "";
      $("sourceSummary").innerHTML = "";
      $("pinnedSources").innerHTML = "";
      $("heldBackSources").innerHTML = "";
      $("sourcesUsed").innerHTML = "";
      $("openQuestions").innerHTML = "";
      $("traceList").innerHTML = "";
      $("comparison").innerHTML = "";
      $("events").innerHTML = "";
      $("documents").innerHTML = "";
      $("artifacts").innerHTML = "";
      $("evidence").innerHTML = "";
      $("answerSources").innerHTML = "";
      lastLiveEvents = [];
      lastRenderedTrace = null;
      updateCommandStep("Ready", "Choose a corpus, ask a question, review the source plan, then run.");
      syncCommandButtons();
      updateQuickDock();
      renderNextPassSetup();
      renderActionBoard();
      status.textContent = "";
    });
    $("firstReadPaths").addEventListener("input", () => {
      if (!suppressFirstReadDirty) firstReadPathsDirty = true;
      renderNextPassSetup();
    });
    $("nudge").addEventListener("input", renderNextPassSetup);
    $("rerunPaths").addEventListener("input", renderNextPassSetup);
    $("tracepath").addEventListener("input", renderNextPassSetup);
    $("paths").addEventListener("input", renderActionBoard);
    $("objective").addEventListener("input", renderActionBoard);
    stopRun.addEventListener("click", async () => {
      if (!activeJobId) return;
      stopRun.disabled = true;
      syncCommandButtons();
      status.textContent = "Stopping";
      try {
        const response = await fetch("/api/cancel-run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({job_id: activeJobId})
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Stop failed");
        renderLiveEvents(data.events || []);
        status.textContent = data.status === "canceled" ? "Stopped" : "Stop requested";
      } catch (error) {
        status.textContent = error.message;
      }
    });
    chooseFolder.addEventListener("click", () => choosePath("folder", "paths"));
    chooseFile.addEventListener("click", () => choosePath("file", "paths"));
    chooseFiles.addEventListener("click", () => choosePath("files", "paths"));
    chooseRerunFolder.addEventListener("click", () => choosePath("folder", "rerunPaths"));
    chooseRerunFile.addEventListener("click", () => choosePath("file", "rerunPaths"));
    chooseRerunFiles.addEventListener("click", () => choosePath("files", "rerunPaths"));
    applyCheckedCandidates.addEventListener("click", () => applyCheckedCandidatePaths({silent: false}));
    selectAllCandidates.addEventListener("click", () => {
      document.querySelectorAll(".candidate-check").forEach(input => input.checked = true);
      applyCheckedCandidatePaths({silent: false});
    });
    restoreRecommendedCandidates.addEventListener("click", () => {
      const recommended = currentPlan && Array.isArray(currentPlan.first_read_paths) ? currentPlan.first_read_paths : [];
      if (!recommended.length) {
        status.textContent = "No recommended first-read set is loaded";
        return;
      }
      setFirstReadPaths(recommended, {dirty: true});
      renderCandidateReview(currentPlan.top_candidates || [], recommended);
      status.textContent = `Restored ${recommended.length} recommended document(s)`;
    });
    planRun.addEventListener("click", async () => {
      planRun.disabled = true;
      syncCommandButtons();
      status.textContent = "Planning";
      try {
        const plan = await requestPlan({paths: pathPayload($("paths").value)});
        currentPlan = plan;
        renderPlan(plan, {mode: "initial", pathKey: initialPlanPathKey()});
        status.textContent = `Plan ready: ${plan.first_read_count || 0} first-read document(s)`;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        planRun.disabled = false;
        syncCommandButtons();
      }
    });
    applyPlanCorrection.addEventListener("click", () => {
      planRun.click();
    });
    run.addEventListener("click", async () => {
      run.disabled = true;
      syncCommandButtons();
      status.textContent = "Running";
      try {
        if (planNeedsRefresh()) {
          const plan = await requestPlan({paths: pathPayload($("paths").value)});
          currentPlan = plan;
          renderPlan(plan, {mode: "initial", pathKey: initialPlanPathKey()});
          status.textContent = "Plan ready. Review first-read documents, then click Run Approved Plan again.";
          return;
        }
        if (!firstReadPathsDirty) applyCheckedCandidatePaths({silent: true});
        const response = await fetch("/api/run-async", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            matter_id: $("matter").value,
            chat_id: $("chat").value,
            paths: pathPayload($("paths").value),
            objective: $("objective").value,
            conversation_history: activeConversationHistory(),
            live_synthesis: $("live").checked,
            use_llm_planning: $("usePlanner").checked,
            top_k: Number($("topk").value || 36),
            selected_paths: pathPayload($("firstReadPaths").value),
            pinned_paths: pinnedSourcePaths,
            plan_note: $("planNote").value
          })
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Run failed");
        activeJobId = data.job_id || "";
        stopRun.disabled = !activeJobId;
        syncCommandButtons();
        renderLiveEvents(data.events || []);
        await pollRunJob(data.job_id);
      } catch (error) {
        status.textContent = error.message;
      } finally {
        run.disabled = false;
        stopRun.disabled = true;
        syncCommandButtons();
      }
    });
    loadTrace.addEventListener("click", async () => {
      loadTrace.disabled = true;
      status.textContent = "Loading trace";
      try {
        const response = await fetch("/api/trace?path=" + encodeURIComponent($("tracepath").value));
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Trace load failed");
        render(data);
        await refreshTraceList({setStatus: false});
        status.textContent = `Loaded saved run: ${filenameFromPath($("tracepath").value)}`;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        loadTrace.disabled = false;
      }
    });
    listTraces.addEventListener("click", async () => {
      listTraces.disabled = true;
      status.textContent = "Listing traces";
      try {
        await refreshTraceList({setStatus: true});
      } catch (error) {
        status.textContent = error.message;
      } finally {
        listTraces.disabled = false;
      }
    });
    rerunTrace.addEventListener("click", async () => {
      rerunTrace.disabled = true;
      syncCommandButtons();
      status.textContent = "Rerunning";
      try {
        if (rerunPlanNeedsRefresh()) {
          const plan = await requestRerunPlan();
          currentPlan = plan;
          renderPlan(plan, {mode: "rerun", pathKey: rerunPlanPathKey()});
          status.textContent = "Steering plan ready. Review first-read documents, then click Run Corrected Pass again.";
          return;
        }
        const response = await fetch("/api/rerun-async", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            trace_path: $("tracepath").value,
            nudge: $("nudge").value,
            chat_id: $("chat").value,
            paths: pathPayload($("rerunPaths").value),
            live_synthesis: $("live").checked,
            use_llm_planning: $("usePlanner").checked,
            top_k: Number($("topk").value || 36),
            selected_paths: firstReadPathsDirty ? pathPayload($("firstReadPaths").value) : [],
            selected_paths_locked: firstReadPathsDirty,
            excluded_paths: excludedSourcePaths,
            pinned_paths: pinnedSourcePaths,
            plan_note: $("planNote").value
          })
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Rerun failed");
        activeJobId = data.job_id || "";
        stopRun.disabled = !activeJobId;
        syncCommandButtons();
        renderLiveEvents(data.events || []);
        await pollRunJob(data.job_id);
      } catch (error) {
        status.textContent = error.message;
      } finally {
        rerunTrace.disabled = false;
        stopRun.disabled = true;
        syncCommandButtons();
      }
    });
    previewNudgePlan.addEventListener("click", async () => {
      previewNudgePlan.disabled = true;
      syncCommandButtons();
      status.textContent = "Previewing nudge plan";
      try {
        const plan = await requestRerunPlan();
        currentPlan = plan;
        renderPlan(plan, {mode: "rerun", pathKey: rerunPlanPathKey()});
        status.textContent = `Nudge plan ready: ${plan.first_read_count || 0} first-read document(s)`;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        previewNudgePlan.disabled = false;
        syncCommandButtons();
      }
    });
    async function choosePath(mode, targetId) {
      status.textContent = "Waiting for native picker";
      try {
        const currentPaths = pathPayload($(targetId).value);
        const initial = currentPaths.length ? currentPaths[currentPaths.length - 1] : "";
        const response = await fetch("/api/pick-path", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({mode, initial_dir: initial})
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Path picker failed");
        appendPaths(targetId, data.paths || []);
        status.textContent = (data.paths || []).length ? `Added ${(data.paths || []).length} path(s)` : "Picker cancelled";
      } catch (error) {
        status.textContent = error.message;
      }
    }
    function appendPaths(targetId, paths) {
      const existing = pathPayload($(targetId).value);
      const seen = new Set(existing.map(path => path.toLowerCase()));
      for (const path of paths || []) {
        const clean = String(path || "").trim();
        if (!clean || seen.has(clean.toLowerCase())) continue;
        existing.push(clean);
        seen.add(clean.toLowerCase());
      }
      $(targetId).value = existing.join("\n");
      if (targetId === "firstReadPaths" || targetId === "rerunPaths" || targetId === "paths") {
        renderNextPassSetup();
      }
    }
    function setFirstReadPaths(paths, {dirty = false} = {}) {
      suppressFirstReadDirty = true;
      $("firstReadPaths").value = (paths || []).join("\n");
      suppressFirstReadDirty = false;
      firstReadPathsDirty = dirty;
      renderNextPassSetup();
    }
    function syncCommandButtons() {
      topRun.disabled = run.disabled;
      topStop.disabled = stopRun.disabled;
      topPlan.disabled = planRun.disabled;
      topPreviewNudge.disabled = previewNudgePlan.disabled;
      topApplyNudge.disabled = rerunTrace.disabled;
      quickStop.disabled = stopRun.disabled;
      quickPreviewNudge.disabled = previewNudgePlan.disabled;
      quickRunNudge.disabled = rerunTrace.disabled;
    }
    function updateCommandStep(title, detail) {
      $("commandStepTitle").textContent = title || "Ready";
      $("commandStepDetail").textContent = detail || "Choose a corpus, ask a question, review the source plan, then run.";
      updateQuickDock(title, detail);
    }
    function updateQuickDock(title = null, detail = null) {
      const firstRead = pathPayload($("firstReadPaths").value);
      const loaded = hasLoadedRunForCorrection();
      const answerReady = Boolean($("answer").textContent.trim() || (lastRenderedTrace && (lastRenderedTrace.rendered_answer || lastRenderedTrace.draft_answer)));
      const currentTitle = title || $("commandStepTitle").textContent || "Ready";
      const currentDetail = detail || $("commandStepDetail").textContent || "Choose a corpus, ask a question, review the source plan, then run.";
      const next = loaded || answerReady
        ? "Use the dock to steer the next pass, pin or ignore sources, and rerun without starting over."
        : firstRead.length
          ? "The first-read plan is ready. Run it, or correct the source plan before reading."
          : "Start by reviewing the source plan so Irys reads intentionally.";
      $("quickDockSummary").innerHTML =
        `<strong>${escapeHtml(currentTitle)}</strong>` +
        `<small>${escapeHtml(currentDetail)}</small>` +
        `<small>${escapeHtml(next)}</small>`;
    }
    function appendInstructionToTextarea(targetId, text) {
      const clean = String(text || "").trim();
      if (!clean) return;
      const target = $(targetId);
      const existing = target.value.trim();
      if (!existing) {
        target.value = clean;
      } else if (!existing.toLowerCase().includes(clean.toLowerCase())) {
        target.value = `${existing}\n${clean}`;
      }
      renderNextPassSetup();
    }
    function setSourcePlanOpen(open) {
      const details = $("sourcePlanDetails");
      if (details) details.open = Boolean(open);
    }
    async function requestPlan({paths, selected_paths = []}) {
      const response = await fetch("/api/plan", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          objective: $("objective").value,
          paths,
          selected_paths,
          plan_note: $("planNote").value,
          use_llm_planning: $("usePlanner").checked,
          top_k: Number($("topk").value || 36)
        })
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || "Plan failed");
      return data;
    }
    async function requestRerunPlan() {
      const response = await fetch("/api/rerun-plan", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          trace_path: $("tracepath").value,
          nudge: $("nudge").value,
          paths: pathPayload($("rerunPaths").value),
          use_llm_planning: $("usePlanner").checked,
          top_k: Number($("topk").value || 36),
          selected_paths: firstReadPathsDirty ? pathPayload($("firstReadPaths").value) : [],
          selected_paths_locked: firstReadPathsDirty,
          excluded_paths: excludedSourcePaths,
          pinned_paths: pinnedSourcePaths,
          plan_note: $("planNote").value
        })
      });
      const plan = await response.json();
      if (!response.ok || plan.error) throw new Error(plan.error || "Nudge plan failed");
      return plan;
    }
    function renderPlan(plan, {mode = "initial", pathKey = initialPlanPathKey()} = {}) {
      const firstRead = plan.first_read_paths || [];
      $("objective").classList.remove("compact");
      setFirstReadPaths(firstRead, {dirty: false});
      currentPlan = plan;
      currentPlanNote = $("planNote").value;
      currentPlanObjective = $("objective").value;
      currentPlanMode = mode;
      currentPlanPathKey = pathKey;
      lastRenderedTrace = null;
      renderNextPassSetup();
      const candidates = (plan.top_candidates || []).slice(0, 6).map(item =>
        `- ${item.filename || item.path}: score ${item.score}; ${(item.reasons || []).join("; ")}`
      );
      const planner = plan.source_planner || {};
      const plannerMetrics = plan.planner_metrics || {};
      const rows = [
        ["Task", plan.interpreted_goal || ""],
        ["First read", `${plan.first_read_count || 0} of ${plan.discovered_count || 0} document(s). ${plan.not_read_first_count || 0} held back for later.`],
        ["Why these files", plan.document_strategy || ""],
        ["Worker source planner", formatPlannerSummary(planner, plannerMetrics)],
        ["Likely document types", (plan.likely_document_families || []).join(", ") || "No specific family inferred."],
        ["Needed information", (plan.needed_information || []).join("\n")],
        ["Top path matches", candidates.join("\n") || "No path-level candidates were scored."]
      ];
      $("planPreview").innerHTML = rows.map(([title, body]) => card(title, body)).join("");
      renderCandidateReview(plan.top_candidates || [], firstRead);
      renderRunBrief({plan});
      renderReviewChecklist({plan});
      renderInvestigationMap({plan});
      renderActionBoard();
      setSourcePlanOpen(true);
    }
    function planNeedsRefresh() {
      if (!pathPayload($("firstReadPaths").value).length) return true;
      if (!currentPlan) return false;
      if (currentPlanMode !== "initial") return true;
      if (currentPlanPathKey !== initialPlanPathKey()) return true;
      return currentPlanNote !== $("planNote").value || currentPlanObjective !== $("objective").value;
    }
    function initialPlanPathKey() {
      return pathPayload($("paths").value).join("\n");
    }
    function rerunPlanPathKey() {
      return [
        String($("tracepath").value || ""),
        String($("nudge").value || ""),
        pathPayload($("rerunPaths").value).join("\n"),
        String($("planNote").value || ""),
        excludedSourcePaths.join("\n"),
        pinnedSourcePaths.join("\n")
      ].join("\n---\n");
    }
    function rerunPlanNeedsRefresh() {
      return currentPlanMode !== "rerun" || currentPlanPathKey !== rerunPlanPathKey();
    }
    function renderTracePlan(metadata, trace = null) {
      const scope = metadata.corpus_scope_decision || null;
      if (!scope) return;
      const selected = scope.selected_paths || [];
      setFirstReadPaths(selected, {dirty: false});
      pinnedSourcePaths = Array.isArray(metadata.pinned_context_files) ? [...metadata.pinned_context_files] : [];
      renderPinnedSources();
      $("planNote").value = metadata.plan_note || $("planNote").value;
      currentPlanNote = $("planNote").value;
      currentPlanObjective = $("objective").value;
      const contract = latestAnswerContract(trace);
      const top = (scope.scored_paths || []).slice(0, 6).map(item =>
        `- ${item.filename || item.path}: score ${item.score}; ${(item.reasons || []).join("; ")}`
      );
      const planLike = {
        first_read_paths: selected,
        top_candidates: scope.scored_paths || [],
        source_planner: (scope.signals || {}).source_planner || null
      };
      currentPlan = planLike;
      currentPlanMode = "trace";
      currentPlanPathKey = "";
      $("planPreview").innerHTML = [
        ["Task", $("objective").value || ""],
        contract && contract.interpreted_goal ? ["Answer target", contract.interpreted_goal] : null,
        contract && contract.needed_information ? ["Answer needs", contract.needed_information.join("\n")] : null,
        contract && contract.search_queries ? ["Search targets", contract.search_queries.join("\n")] : null,
        ["First read", `${selected.length} of ${(scope.discovered_paths || []).length} document(s).`],
        ["Why these files", scope.reason || ""],
        ["Worker source planner", formatPlannerSummary((scope.signals || {}).source_planner || null, {})],
        ["Likely document types", ((scope.signals || {}).requested_families || []).join(", ") || "No specific family inferred."],
        ["Top path matches", top.join("\n") || "No path-level candidates were scored."]
      ].filter(Boolean).map(([title, body]) => card(title, body)).join("");
      renderCandidateReview(scope.scored_paths || [], selected);
      renderRunBrief({plan: planLike});
      renderReviewChecklist({plan: planLike});
      renderInvestigationMap({plan: planLike, trace});
    }
    function latestAnswerContract(trace) {
      const versions = trace && Array.isArray(trace.answer_contract_versions) ? trace.answer_contract_versions : [];
      return versions.length ? versions[versions.length - 1] : null;
    }
    function renderCandidateReview(candidates, selectedPaths) {
      const rows = Array.isArray(candidates) ? candidates.slice(0, 20) : [];
      if (!rows.length) {
        $("candidateReview").innerHTML = emptyState("No ranked document candidates were returned for this plan.");
        return;
      }
      const selected = new Set((selectedPaths || []).map(path => String(path || "").toLowerCase()));
      $("candidateReview").innerHTML = rows.map((item, index) => {
        const path = String(item.path || "");
        const checked = selected.has(path.toLowerCase()) ? " checked" : "";
        const reasons = Array.isArray(item.reasons) ? item.reasons.join("; ") : "";
        const title = `${index + 1}. ${item.filename || path || "Document"}`;
        const body = [
          `Score ${item.score ?? 0}`,
          reasons,
          path
        ].filter(Boolean).join("\n");
        return `<div class="item"><label class="candidate">` +
          `<input type="checkbox" class="candidate-check" data-path="${escapeAttr(path)}"${checked} />` +
          `<span><strong>${escapeHtml(title)}</strong><small><pre>${escapeHtml(body)}</pre></small></span>` +
          `</label></div>`;
      }).join("");
    }
    function applyCheckedCandidatePaths({silent = false} = {}) {
      const checked = Array.from(document.querySelectorAll(".candidate-check:checked"))
        .map(input => input.getAttribute("data-path") || "")
        .filter(Boolean);
      if (!checked.length) {
        if (!silent) status.textContent = "No checked document candidates";
        renderNextPassSetup();
        return;
      }
      setFirstReadPaths(checked, {dirty: true});
      if (!silent) status.textContent = `Applied ${checked.length} first-read document(s)`;
      renderNextPassSetup();
    }
    function formatPlannerSummary(planner, metrics) {
      if (!planner) return "Not available for this run.";
      const lines = [
        `Status: ${planner.status || "unknown"}`,
        planner.reason ? `Reason: ${planner.reason}` : "",
        planner.confidence ? `Confidence: ${planner.confidence}` : "",
        planner.selected_count !== undefined ? `Selected: ${planner.selected_count}` : "",
        metrics && metrics.total_tokens ? `Planner tokens: ${metrics.total_tokens}` : "",
        metrics && metrics.estimated_cost ? `Planner cost: $${Number(metrics.estimated_cost || 0).toFixed(4)}` : ""
      ].filter(Boolean);
      return lines.join("\n");
    }
    async function pollRunJob(jobId) {
      if (!jobId) throw new Error("Missing run job id");
      if (activeJobPoll) clearTimeout(activeJobPoll);
      return new Promise((resolve, reject) => {
        const poll = async () => {
          try {
            const response = await fetch("/api/run-status?job_id=" + encodeURIComponent(jobId));
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(data.error || "Run status failed");
            renderLiveEvents(data.events || []);
            status.textContent = data.status === "running" ? "Running" : data.status;
            if (data.status === "completed") {
              render(data.result || {});
              await refreshTraceList({setStatus: false});
              status.textContent = (data.result || {}).trace_path || "Completed";
              $("tracepath").value = (data.result || {}).trace_path || "";
              resolve();
              return;
            }
            if (data.status === "canceled") {
              status.textContent = "Stopped";
              resolve();
              return;
            }
            if (data.status === "failed") {
              reject(new Error(data.error || "Run failed"));
              return;
            }
            activeJobPoll = setTimeout(poll, 800);
          } catch (error) {
            reject(error);
          }
        };
        poll();
      });
    }
    function pathPayload(pathText) {
      const paths = String(pathText || "").split(/\r?\n/).map(line => line.trim()).filter(Boolean);
      return paths;
    }
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!target || !target.matches) return;
      if (target.matches(".action-board-action, .map-action")) {
        const action = target.getAttribute("data-action") || "";
        if (action === "focus_objective") {
          $("objective").focus();
          $("objective").scrollIntoView({behavior: "smooth", block: "center"});
        } else if (action === "review_plan") {
          setSourcePlanOpen(true);
          $("sourcePlanDetails").scrollIntoView({behavior: "smooth", block: "start"});
          if (!currentPlan || !pathPayload($("firstReadPaths").value).length) planRun.click();
        } else if (action === "run_plan") {
          run.click();
        } else if (action === "stop_run") {
          stopRun.click();
        } else if (action === "review_sources") {
          $("sourceSummary").scrollIntoView({behavior: "smooth", block: "start"});
        } else if (action === "review_answer") {
          $("answer").scrollIntoView({behavior: "smooth", block: "start"});
        } else if (action === "preview_steering") {
          previewNudgePlan.click();
        } else if (action === "focus_steering") {
          $("nudge").focus();
          $("nudge").scrollIntoView({behavior: "smooth", block: "center"});
        }
        return;
      }
      if (target.matches(".candidate-check")) {
        applyCheckedCandidatePaths({silent: true});
        status.textContent = `Updated first-read set to ${pathPayload($("firstReadPaths").value).length} document(s)`;
        return;
      }
      if (target.matches(".trace-load")) {
        $("tracepath").value = target.getAttribute("data-path") || "";
        loadTrace.click();
        return;
      }
      if (target.matches(".include-held-back")) {
        const path = target.getAttribute("data-path") || "";
        removeExcludedSourcePath(path);
        appendPaths("firstReadPaths", [path]);
        firstReadPathsDirty = true;
        appendSteeringInstruction(`Read ${filenameFromPath(path)} in the next pass.`);
        announceNextPassQueued(`Added ${filenameFromPath(path)} to the next first-read set.`);
        return;
      }
      if (target.matches(".deeper-source-next")) {
        const path = target.getAttribute("data-path") || "";
        removeExcludedSourcePath(path);
        appendPaths("firstReadPaths", [path]);
        firstReadPathsDirty = true;
        appendSteeringInstruction(`Read ${filenameFromPath(path)} more closely in the next pass.`);
        announceNextPassQueued(`Added ${filenameFromPath(path)} for a deeper next pass.`);
        return;
      }
      if (target.matches(".pin-source-next")) {
        const path = target.getAttribute("data-path") || "";
        removeExcludedSourcePath(path);
        addPinnedSourcePath(path);
        appendPaths("firstReadPaths", [path]);
        firstReadPathsDirty = true;
        appendSteeringInstruction(`Keep ${filenameFromPath(path)} pinned in synthesis next pass.`);
        announceNextPassQueued(`Pinned ${filenameFromPath(path)} for synthesis next pass.`);
        return;
      }
      if (target.matches(".unpin-source-next")) {
        const path = target.getAttribute("data-path") || "";
        removePinnedSourcePath(path);
        firstReadPathsDirty = true;
        status.textContent = `Unpinned ${filenameFromPath(path)} for the next pass`;
        return;
      }
      if (target.matches(".unignore-source-next")) {
        const path = target.getAttribute("data-path") || "";
        removeExcludedSourcePath(path);
        status.textContent = `Removed ${filenameFromPath(path)} from ignored sources`;
        return;
      }
      if (target.matches(".ignore-source-next")) {
        const path = target.getAttribute("data-path") || "";
        addExcludedSourcePath(path);
        removePinnedSourcePath(path);
        removePathFromFirstRead(path);
        firstReadPathsDirty = true;
        appendSteeringInstruction(`Ignore ${filenameFromPath(path)} in the next pass.`);
        announceNextPassQueued(`Marked ${filenameFromPath(path)} to ignore next pass.`);
      }
    });
    function render(data) {
      const trace = data.trace || {};
      lastRenderedTrace = trace;
      const metrics = trace.metrics || {};
      $("docCount").textContent = String((trace.documents || []).length);
      $("chunkCount").textContent = String((trace.chunks || []).length);
      $("tokens").textContent = String(metrics.total_tokens || 0);
      $("cost").textContent = "$" + Number(metrics.estimated_cost || 0).toFixed(4);
      if (data.matter_summary) renderCostSummary(data.matter_summary);
      const metadata = ((trace.task || {}).metadata || {});
      $("matter").value = metadata.matter_id || (trace.task || {}).task_id || $("matter").value;
      $("chat").value = metadata.chat_id || $("chat").value || "main";
      $("objective").value = (trace.task || {}).question || $("objective").value;
      $("objective").classList.add("compact");
      excludedSourcePaths = [];
      renderTracePlan(metadata, trace);
      renderLiveEvents(trace.events || []);
      setSourcePlanOpen(false);
      renderNextPassSetup();
      renderRunBrief({trace, comparison: data.comparison});
      renderReviewChecklist({trace, comparison: data.comparison});
      renderInvestigationMap({trace, comparison: data.comparison});
      $("answer").innerHTML = renderMarkdown(data.rendered_answer || trace.rendered_answer || "");
      renderActionBoard();
      updateConversationHistoryFromTrace(trace);
      $("chatHistory").innerHTML = renderChatHistory(activeConversationHistory());
      renderSourceReview(trace);
      $("diagnosis").innerHTML = renderRunHealth(trace);
      $("comparison").innerHTML = renderComparison(data.comparison);
      $("events").innerHTML = renderLimitedCards(trace.events || [], event => card(
        event.label + " - " + event.message,
        compactJsonForDisplay(event.fields || {})
      ), {limit: 50, itemName: "event"});
      $("documents").innerHTML = renderLimitedCards(trace.documents || [], doc => card(
        (doc.doc_id || "") + " - " + (doc.filename || ""),
        (doc.path || "") + "\nchars=" + (doc.text_chars || 0) + (doc.load_error ? "\nerror=" + doc.load_error : "")
      ), {limit: 100, itemName: "document"});
      $("artifacts").innerHTML = renderLimitedCards(trace.artifacts || [], artifact => card(
        artifact.filename || artifact.type || "Artifact",
        (artifact.path || "") + "\ntype=" + (artifact.type || "") + "\ndiagnostic=" + Boolean(artifact.diagnostic)
      ), {limit: 50, itemName: "artifact"});
      const evidence = ((trace.final_packet || {}).verified_evidence || []);
      $("evidence").innerHTML = renderLimitedCards(evidence, item => card(
        item.claim || "Evidence",
        (item.raw_support || "") + "\n" + compactJsonForDisplay(item.source || {})
      ), {limit: 50, itemName: "evidence record"});
      const answerSources = ((trace.final_packet || {}).answer_source_map || []);
      $("answerSources").innerHTML = renderLimitedCards(answerSources, item => card(
        "Section " + (item.section_index || ""),
        (item.answer_excerpt || "") + "\n" + compactJsonForDisplay(item.source_refs || [])
      ), {limit: 50, itemName: "answer-source record"});
    }
    async function refreshTraceList({setStatus = false} = {}) {
      const url = "/api/traces?matter_id=" + encodeURIComponent($("matter").value) +
        "&chat_id=" + encodeURIComponent($("chat").value);
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || "Trace list failed");
      $("traceList").innerHTML = renderTraceList(data.traces || []);
      renderCostSummary(data.matter_summary || data.summary || {});
      if (setStatus) status.textContent = `${(data.traces || []).length} traces`;
      return data;
    }
    function renderCostSummary(summary) {
      $("matterCost").textContent = "$" + Number(summary.estimated_cost || 0).toFixed(4);
      $("matterMessages").textContent = String(summary.messages || 0);
    }
    function card(title, body) {
      return `<div class="item"><strong>${escapeHtml(title)}</strong><small><pre>${escapeHtml(limitDisplayText(body || ""))}</pre></small></div>`;
    }
    function emptyState(text) {
      return `<div class="empty">${escapeHtml(text)}</div>`;
    }
    function limitDisplayText(value, maxChars = 2500) {
      const text = String(value || "");
      if (text.length <= maxChars) return text;
      return text.slice(0, maxChars) + `\n\n... ${text.length - maxChars} more characters are saved in the trace but not rendered here.`;
    }
    function compactJsonForDisplay(value, {arrayLimit = 50, stringLimit = 2000} = {}) {
      return JSON.stringify(value, (key, item) => {
        if (typeof item === "string" && item.length > stringLimit) {
          return item.slice(0, stringLimit) + ` ... ${item.length - stringLimit} more characters saved in trace`;
        }
        if (Array.isArray(item) && item.length > arrayLimit) {
          return item.slice(0, arrayLimit).concat([`... ${item.length - arrayLimit} more item(s) saved in trace`]);
        }
        return item;
      }, 2);
    }
    function formatInlineList(items, limit = 12) {
      const rows = Array.isArray(items) ? items : [];
      return rows.slice(0, limit).join(", ") + (rows.length > limit ? `, ... ${rows.length - limit} more saved in trace` : "");
    }
    function renderLimitedCards(items, renderer, {limit = 100, itemName = "item"} = {}) {
      const rows = Array.isArray(items) ? items : [];
      const visible = rows.slice(0, limit).map(renderer);
      if (rows.length > limit) {
        visible.push(emptyState(`${rows.length - limit} additional ${itemName}(s) are saved in the trace but not rendered here.`));
      }
      return visible.join("");
    }
    function renderSourceReview(trace) {
      const packet = trace.final_packet || {};
      const metadata = (trace.task || {}).metadata || {};
      const scope = metadata.corpus_scope_decision || {};
      const docs = Array.isArray(trace.documents) ? trace.documents : [];
      const chunks = Array.isArray(trace.chunks) ? trace.chunks : [];
      const evidence = Array.isArray(packet.verified_evidence) ? packet.verified_evidence : [];
      const answerSources = Array.isArray(packet.answer_source_map) ? packet.answer_source_map : [];
      const unresolved = Array.isArray(packet.unresolved) ? packet.unresolved : [];
      const pinnedSources = Array.isArray(packet.pinned_sources) ? packet.pinned_sources : [];
      const packetReview = packet.packet_review || {};
      const reviewerRelevant = sourceIdsToDocuments(packetReview.relevant_source_ids || [], docs);
      const reviewerLowValue = sourceIdsToDocuments(packetReview.low_value_source_ids || [], docs);
      const loadErrors = docs.filter(doc => doc && doc.load_error);
      const retrieved = Array.isArray(packet.retrieved_chunks) ? packet.retrieved_chunks : [];
      const summaryRows = [
        ["Corpus read", `${docs.length} document(s), ${chunks.length} searchable source passage(s).`],
        ["Evidence found", `${evidence.length} source item(s) carried into the answer packet.`],
        ["Pinned sources", pinnedSources.length ? `${pinnedSources.length} document(s) pinned into synthesis context.` : "No document was pinned into synthesis context."],
        ["Sources selected", `${retrieved.length} retrieved passage(s) were considered.`],
        ["Open questions", unresolved.length ? `${unresolved.length} issue(s) still flagged.` : "No obvious gaps were logged."]
      ];
      if (scope.reason) {
        const planner = (scope.signals || {}).source_planner || null;
        summaryRows.splice(1, 0, ["First-read plan", scope.reason]);
        if (planner) summaryRows.splice(2, 0, ["Worker source planner", formatPlannerSummary(planner, {})]);
      }
      if (reviewerRelevant.length) {
        summaryRows.push(["Reviewer kept", reviewerRelevant.map(doc => `- ${doc.filename || doc.path || doc.doc_id}`).join("\n")]);
      }
      if (reviewerLowValue.length) {
        summaryRows.push(["Reviewer marked low value", reviewerLowValue.map(doc => `- ${doc.filename || doc.path || doc.doc_id}`).join("\n")]);
      }
      if (loadErrors.length) {
        summaryRows.push(["Load issues", formatSimpleList(loadErrors.map(doc => `${doc.filename || doc.path}: ${doc.load_error}`), 20)]);
      }
      $("sourceSummary").innerHTML =
        summaryRows.map(([title, body]) => card(title, body)).join("") +
        renderReviewerSourceActions(packetReview, docs);
      renderPinnedSources();
      $("heldBackSources").innerHTML = renderHeldBackSources(scope);
      $("sourcesUsed").innerHTML = renderSourcesUsed(answerSources, evidence, docs);
      $("openQuestions").innerHTML = unresolved.length
        ? renderLimitedCards(unresolved, (item, index) => card(`Open question ${index + 1}`, item), {limit: 25, itemName: "open question"})
        : emptyState("No open question was logged for this run.");
    }
    function sourceIdsToDocuments(ids, docs) {
      const wanted = new Set((Array.isArray(ids) ? ids : []).map(id => String(id || "")));
      if (!wanted.size) return [];
      return (Array.isArray(docs) ? docs : []).filter(doc => wanted.has(String(doc.doc_id || "")));
    }
    function renderReviewerSourceActions(packetReview, docs) {
      const relevant = sourceIdsToDocuments((packetReview || {}).relevant_source_ids || [], docs);
      const lowValue = sourceIdsToDocuments((packetReview || {}).low_value_source_ids || [], docs);
      const cards = [];
      for (const doc of relevant.slice(0, 12)) {
        const path = doc.path || "";
        cards.push(
          `<div class="item"><strong>${escapeHtml("Reviewer kept: " + (doc.filename || doc.doc_id || "source"))}</strong>` +
          `<small>This source looked useful for the current answer packet.</small>` +
          `<div class="row">` +
          `<button class="secondary deeper-source-next" data-path="${escapeAttr(path)}">Read deeper next pass</button>` +
          `<button class="secondary pin-source-next" data-path="${escapeAttr(path)}">Pin to synthesis</button>` +
          `</div></div>`
        );
      }
      for (const doc of lowValue.slice(0, 12)) {
        const path = doc.path || "";
        cards.push(
          `<div class="item"><strong>${escapeHtml("Reviewer marked low value: " + (doc.filename || doc.doc_id || "source"))}</strong>` +
          `<small>The packet reviewer thought this source was less useful for the current answer. You can ignore it next pass, or force Irys to read it more deeply if that judgment looks wrong.</small>` +
          `<div class="row">` +
          `<button class="secondary ignore-source-next" data-path="${escapeAttr(path)}">Ignore next pass</button>` +
          `<button class="secondary deeper-source-next" data-path="${escapeAttr(path)}">Read deeper anyway</button>` +
          `</div></div>`
        );
      }
      if (relevant.length > 12 || lowValue.length > 12) {
        cards.push(emptyState("Additional reviewer source judgments are saved in the trace."));
      }
      return cards.join("");
    }
    function renderSourcesUsed(answerSources, evidence, docs) {
      if (answerSources.length) {
        return renderLimitedCards(answerSources, (item, index) => {
          const support = Array.isArray(item.support) ? item.support : [];
          const supportLines = support.map(source => {
            const docName = docNameFor(source.doc_id, docs);
            return `- ${docName || source.doc_id || source.ref || "source"}: ${source.support || ""}`;
          });
          const sourcePath = firstDocPathForSupport(support, docs);
          const body = [
            item.answer_excerpt || "",
            supportLines.length ? "\nSupport:" : "",
            ...supportLines
          ].filter(Boolean).join("\n");
          return sourceCard(`Answer section ${item.section_index || index + 1}`, body, sourcePath);
        }, {limit: 50, itemName: "answer source"});
      }
      if (evidence.length) {
        return renderLimitedCards(evidence, (item, index) => {
          const source = item.source || {};
          const docName = docNameFor(source.doc_id, docs) || source.doc_id || "source";
          return sourceCard(
            `Source ${index + 1}: ${docName}`,
            `${item.raw_support || item.claim || ""}\n${source.chunk_id ? "Passage: " + source.chunk_id : ""}`,
            docPathFor(source.doc_id, docs)
          );
        }, {limit: 50, itemName: "source"});
      }
      return emptyState("No source support was captured for this answer.");
    }
    function sourceCard(title, body, path) {
      const action = path
        ? `<div class="row"><button class="secondary deeper-source-next" data-path="${escapeAttr(path)}">Read deeper next pass</button>` +
          `<button class="secondary pin-source-next" data-path="${escapeAttr(path)}">Pin to synthesis</button>` +
          `<button class="secondary ignore-source-next" data-path="${escapeAttr(path)}">Ignore next pass</button></div>`
        : "";
      return `<div class="item"><strong>${escapeHtml(title)}</strong><small><pre>${escapeHtml(limitDisplayText(body || "", 2000))}</pre></small>${action}</div>`;
    }
    function renderPinnedSources() {
      const target = $("pinnedSources");
      if (!target) return;
      if (!pinnedSourcePaths.length) {
        target.innerHTML = emptyState("No document is pinned into the next synthesis pass.");
        return;
      }
      target.innerHTML = pinnedSourcePaths.map((path, index) =>
        `<div class="item"><strong>${escapeHtml(`Pinned ${index + 1}: ${filenameFromPath(path)}`)}</strong>` +
        `<small><pre>${escapeHtml(path)}</pre></small>` +
        `<button class="secondary unpin-source-next" data-path="${escapeAttr(path)}">Unpin</button></div>`
      ).join("");
    }
    function renderHeldBackSources(scope) {
      const discovered = Array.isArray(scope.discovered_paths) ? scope.discovered_paths : [];
      const selected = new Set((scope.selected_paths || []).map(path => String(path || "").toLowerCase()));
      const heldBack = discovered.filter(path => !selected.has(String(path || "").toLowerCase()));
      if (!heldBack.length) return emptyState("No documents were held back from the first-read corpus.");
      const scoreByPath = {};
      for (const item of scope.scored_paths || []) {
        scoreByPath[String(item.path || "").toLowerCase()] = item;
      }
      heldBack.sort((left, right) => {
        const leftScore = Number((scoreByPath[String(left || "").toLowerCase()] || {}).score || 0);
        const rightScore = Number((scoreByPath[String(right || "").toLowerCase()] || {}).score || 0);
        return rightScore - leftScore || String(left || "").localeCompare(String(right || ""));
      });
      return heldBack.slice(0, 25).map((path, index) => {
        const item = scoreByPath[String(path || "").toLowerCase()] || {};
        const body = [
          item.score !== undefined ? `Score ${item.score}` : "",
          Array.isArray(item.reasons) ? item.reasons.join("; ") : "",
          String(path || "")
        ].filter(Boolean).join("\n");
        return `<div class="item"><strong>${escapeHtml(`Held back ${index + 1}: ${filenameFromPath(path)}`)}</strong>` +
          `<small><pre>${escapeHtml(body)}</pre></small>` +
          `<button class="secondary include-held-back" data-path="${escapeAttr(path)}">Read next pass</button></div>`;
      }).join("") + (heldBack.length > 25 ? emptyState(`${heldBack.length - 25} additional document(s) were held back.`) : "");
    }
    function filenameFromPath(path) {
      return String(path || "").split(/[\\/]/).filter(Boolean).pop() || String(path || "Document");
    }
    function docNameFor(docId, docs) {
      if (!docId) return "";
      const match = docs.find(doc => String(doc.doc_id || "") === String(docId));
      return match ? (match.filename || match.path || String(docId)) : "";
    }
    function docPathFor(docId, docs) {
      if (!docId) return "";
      const match = docs.find(doc => String(doc.doc_id || "") === String(docId));
      return match ? (match.path || "") : "";
    }
    function firstDocPathForSupport(support, docs) {
      for (const item of support || []) {
        const path = docPathFor(item.doc_id, docs);
        if (path) return path;
      }
      return "";
    }
    function removePathFromFirstRead(path) {
      const target = String(path || "").toLowerCase();
      const kept = pathPayload($("firstReadPaths").value).filter(item => String(item || "").toLowerCase() !== target);
      setFirstReadPaths(kept, {dirty: true});
    }
    function addExcludedSourcePath(path) {
      const clean = String(path || "").trim();
      if (!clean) return;
      const key = clean.toLowerCase();
      if (!excludedSourcePaths.some(item => String(item || "").toLowerCase() === key)) excludedSourcePaths.push(clean);
      renderNextPassSetup();
    }
    function removeExcludedSourcePath(path) {
      const key = String(path || "").toLowerCase();
      excludedSourcePaths = excludedSourcePaths.filter(item => String(item || "").toLowerCase() !== key);
      renderNextPassSetup();
    }
    function addPinnedSourcePath(path) {
      const clean = String(path || "").trim();
      if (!clean) return;
      const key = clean.toLowerCase();
      if (!pinnedSourcePaths.some(item => String(item || "").toLowerCase() === key)) pinnedSourcePaths.push(clean);
      renderPinnedSources();
      renderNextPassSetup();
    }
    function removePinnedSourcePath(path) {
      const key = String(path || "").toLowerCase();
      pinnedSourcePaths = pinnedSourcePaths.filter(item => String(item || "").toLowerCase() !== key);
      renderPinnedSources();
      renderNextPassSetup();
    }
    function appendSteeringInstruction(text) {
      const clean = String(text || "").trim();
      if (!clean) return;
      const existing = $("nudge").value.trim();
      if (!existing) {
        $("nudge").value = clean;
      } else if (!existing.toLowerCase().includes(clean.toLowerCase())) {
        $("nudge").value = `${existing}\n${clean}`;
      }
      renderNextPassSetup();
    }
    function renderNextPassSetup() {
      const firstRead = pathPayload($("firstReadPaths").value);
      const addedPaths = pathPayload($("rerunPaths").value);
      const nudge = $("nudge").value.trim();
      const traceLoaded = Boolean($("tracepath").value.trim());
      const pills = [
        pill(`${firstRead.length} first-read`, firstRead.length ? "important" : ""),
        pill(`${pinnedSourcePaths.length} pinned`, pinnedSourcePaths.length ? "important" : ""),
        pill(`${excludedSourcePaths.length} ignored`, excludedSourcePaths.length ? "warn" : ""),
        pill(`${addedPaths.length} added path${addedPaths.length === 1 ? "" : "s"}`, addedPaths.length ? "important" : ""),
        pill(nudge ? "steering note ready" : "no steering note", nudge ? "important" : ""),
        pill(traceLoaded ? "saved run loaded" : "no saved run loaded", traceLoaded ? "important" : "")
      ].join("");
      $("nextPassSummary").innerHTML = pills;
      const nextAction = traceLoaded
        ? "Preview Steering Plan to re-check source routing, then Run Corrected Pass."
        : "Review Source Plan, adjust first-read documents, then Run Approved Plan.";
      const details = [
        `First-read documents: ${firstRead.length}${firstReadPathsDirty ? " (edited by you)" : ""}.`,
        `Pinned into synthesis: ${pinnedSourcePaths.length}.`,
        `Ignored next pass: ${excludedSourcePaths.length}.`,
        `Additional corpus paths: ${addedPaths.length}.`,
        `Steering note: ${nudge ? "present" : "empty"}.`,
        `Next action: ${nextAction}`
      ];
      const lists = [];
      if (firstRead.length) lists.push(["First-read set", firstRead.slice(0, 8).map(filenameFromPath)]);
      if (pinnedSourcePaths.length) lists.push(["Pinned", pinnedSourcePaths.slice(0, 8).map(filenameFromPath)]);
      if (excludedSourcePaths.length) lists.push(["Ignored", excludedSourcePaths.slice(0, 8).map(filenameFromPath)]);
      const listHtml = lists.map(([title, items]) =>
        `<div class="hint"><strong>${escapeHtml(title)}:</strong> ${escapeHtml(items.join(", "))}</div>`
      ).join("");
      const ignoredControls = excludedSourcePaths.length
        ? `<div class="row">${excludedSourcePaths.slice(0, 6).map(path =>
            `<button class="secondary unignore-source-next" data-path="${escapeAttr(path)}">Unignore ${escapeHtml(filenameFromPath(path))}</button>`
          ).join("")}</div>`
        : "";
      $("nextPassSetup").innerHTML =
        `<strong>Next Pass Setup</strong><small><pre>${escapeHtml(details.join("\n"))}</pre></small>${listHtml}${ignoredControls}`;
      updateQuickDock();
      updateCommandForQueuedNextPass();
      renderActionBoard();
    }
    function hasQueuedNextPassChanges() {
      return Boolean(
        firstReadPathsDirty ||
        pinnedSourcePaths.length ||
        excludedSourcePaths.length ||
        pathPayload($("rerunPaths").value).length ||
        $("nudge").value.trim()
      );
    }
    function hasLoadedRunForCorrection() {
      return Boolean($("tracepath").value.trim() || lastRenderedTrace);
    }
    function hasActiveRun() {
      const latestEvent = lastLiveEvents.length ? lastLiveEvents[lastLiveEvents.length - 1] : null;
      return Boolean(
        (activeJobId && !stopRun.disabled) ||
        (latestEvent && !isRunCompleteEvent(latestEvent) && !["ERROR", "STOP"].includes(String(latestEvent.label || "")))
      );
    }
    function updateCommandForQueuedNextPass() {
      if (!hasQueuedNextPassChanges() || hasActiveRun()) return;
      if (hasLoadedRunForCorrection()) {
        updateCommandStep(
          "Next pass changes queued",
          "Preview Steering checks the revised source plan. Run Corrected Pass executes the queued source and steering decisions."
        );
      } else if (pathPayload($("firstReadPaths").value).length) {
        updateCommandStep(
          "First-read plan edited",
          "Review the selected documents, then Run Approved Plan. Use Preview Corrected Plan if the plan correction should re-rank the corpus first."
        );
      }
    }
    function announceNextPassQueued(message) {
      const suffix = hasLoadedRunForCorrection()
        ? " Next: Preview Steering or Run Corrected Pass."
        : " Next: run the approved plan when the first-read set looks right.";
      status.textContent = `${message}${suffix}`;
      updateCommandForQueuedNextPass();
    }
    function pill(text, tone = "") {
      const cls = tone ? `action-pill ${tone}` : "action-pill";
      return `<span class="${cls}">${escapeHtml(text)}</span>`;
    }
    function renderActionBoard() {
      const target = $("actionBoard");
      if (!target) return;
      const objective = $("objective").value.trim();
      const corpusPathCount = pathPayload($("paths").value).length;
      const firstRead = pathPayload($("firstReadPaths").value);
      const nudge = $("nudge").value.trim();
      const trace = lastRenderedTrace || null;
      const latestEvent = lastLiveEvents.length ? lastLiveEvents[lastLiveEvents.length - 1] : null;
      const runActive = latestEvent && !isRunCompleteEvent(latestEvent) && !["ERROR", "STOP"].includes(String(latestEvent.label || ""));
      const hasAnswer = Boolean(
        (trace && (trace.rendered_answer || trace.draft_answer)) ||
        $("answer").textContent.trim()
      );
      const cards = [];
      cards.push(actionCard(
        "Objective",
        objective ? "Question ready" : "Add the question",
        objective
          ? limitDisplayText(objective, 180)
          : "Type the question, deliverable, or issue you want Irys to solve.",
        objective ? "ready" : "warn",
        "Focus Question",
        "focus_objective"
      ));
      cards.push(actionCard(
        "Sources",
        firstRead.length
          ? `${firstRead.length} first-read source${firstRead.length === 1 ? "" : "s"}`
          : corpusPathCount
            ? "Plan source focus"
            : "Choose corpus",
        firstRead.length
          ? firstRead.slice(0, 3).map(filenameFromPath).join("\n") + (firstRead.length > 3 ? `\n... ${firstRead.length - 3} more` : "")
          : corpusPathCount
            ? "Ask Irys to inspect the file inventory and decide what to read first."
            : "Select a local folder or files for this matter.",
        firstRead.length ? "ready" : corpusPathCount ? "next" : "warn",
        firstRead.length ? "Review Plan" : "Review Source Plan",
        "review_plan"
      ));
      cards.push(actionCard(
        "Current Work",
        runActive
          ? (latestEvent.fields || {}).summary || friendlyEventTitle(latestEvent)
          : hasAnswer
            ? "Answer ready"
            : firstRead.length
              ? "Ready to run"
              : "Waiting for plan",
        runActive
          ? formatUserEventFields(latestEvent.fields || {}) || "Working through the run."
          : hasAnswer
            ? "Review the answer, source support, and open questions. Steer the next pass if the focus is wrong."
            : firstRead.length
              ? "Run the approved first-read plan, or edit the plan before running."
              : "Review the source plan before running so the first read is intentional.",
        runActive ? "next" : hasAnswer ? "ready" : firstRead.length ? "next" : "warn",
        runActive ? "Stop" : hasAnswer ? "Review Sources" : firstRead.length ? "Run Approved Plan" : "Review Source Plan",
        runActive ? "stop_run" : hasAnswer ? "review_sources" : firstRead.length ? "run_plan" : "review_plan"
      ));
      cards.push(actionCard(
        "Steering",
        nudge ? "Correction ready" : hasAnswer ? "Steer next pass" : "Optional",
        nudge
          ? limitDisplayText(nudge, 180)
          : hasAnswer
            ? "Add a correction, pin or ignore sources, then preview the corrected source plan."
            : "After planning or answering, use this to redirect source focus without starting over.",
        nudge ? "next" : "neutral",
        nudge ? "Preview Steering" : "Write Steering",
        nudge ? "preview_steering" : "focus_steering"
      ));
      target.innerHTML = cards.join("");
    }
    function actionCard(label, title, body, tone, buttonText, action) {
      const safeTone = ["ready", "next", "warn", "neutral"].includes(tone) ? tone : "neutral";
      return `<div class="action-card ${safeTone}">` +
        `<b>${escapeHtml(label)}</b>` +
        `<strong>${escapeHtml(title || "")}</strong>` +
        `<small>${formatPlainText(body || "")}</small>` +
        `<button class="secondary action-board-action" data-action="${escapeAttr(action || "")}">${escapeHtml(buttonText || "Open")}</button>` +
        `</div>`;
    }
    function renderRunHealth(trace) {
      const diagnosis = trace.diagnosis || {};
      const packet = trace.final_packet || {};
      const metrics = trace.metrics || {};
      const statusText = diagnosis.status === "ready_for_review"
        ? "Ready for review"
        : diagnosis.status === "needs_attention"
          ? "Needs attention"
          : (diagnosis.status || "Not diagnosed");
      const rows = [
        ["Status", statusText],
        ["Evidence", `${diagnosis.evidence_count ?? (packet.verified_evidence || []).length} item(s) in the final packet.`],
        ["Source map", `${diagnosis.answer_source_map_count ?? (packet.answer_source_map || []).length} answer section link(s).`],
        ["Pinned sources", `${diagnosis.pinned_source_count ?? (packet.pinned_sources || []).length} document(s) pinned into synthesis context.`],
        ["Cost", `$${Number(metrics.estimated_cost || 0).toFixed(4)} for this message.`],
        ["Tokens by tier", formatTierMetric(metrics.tokens_by_tier || {})],
        ["Cost by tier", formatTierMetric(metrics.cost_by_tier || {}, {currency: true})]
      ];
      const unresolved = Array.isArray(packet.unresolved) ? packet.unresolved : [];
      if (unresolved.length) rows.push(["Needs review", unresolved.join("\n")]);
      return rows.map(([title, body]) => card(title, body)).join("");
    }
    function formatTierMetric(values, {currency = false} = {}) {
      const entries = Object.entries(values || {}).filter(([, value]) => Number(value || 0) !== 0);
      if (!entries.length) return "No model usage was recorded for this run.";
      return entries.map(([tier, value]) => {
        const label = tier.replace(/_/g, " ");
        const formatted = currency ? "$" + Number(value || 0).toFixed(4) : String(value || 0);
        return `${label}: ${formatted}`;
      }).join("\n");
    }
    function renderRunBrief({plan = null, trace = null, events = null, comparison = null} = {}) {
      const rows = [];
      const objective = String(((trace || {}).task || {}).question || $("objective").value || currentPlanObjective || "").trim();
      if (objective && trace) rows.push(["Question", objective]);
      if (trace) {
        const packet = trace.final_packet || {};
        const metadata = (trace.task || {}).metadata || {};
        const scope = metadata.corpus_scope_decision || {};
        const docs = Array.isArray(trace.documents) ? trace.documents : [];
        const chunks = Array.isArray(trace.chunks) ? trace.chunks : [];
        const evidence = Array.isArray(packet.verified_evidence) ? packet.verified_evidence : [];
        const unresolved = Array.isArray(packet.unresolved) ? packet.unresolved : [];
        const answerSources = Array.isArray(packet.answer_source_map) ? packet.answer_source_map : [];
        const selected = Array.isArray(scope.selected_paths) ? scope.selected_paths : [];
        rows.push(["Corpus read", `${docs.length} document(s), ${chunks.length} searchable source passage(s).`]);
        if (selected.length) rows.push(["First-read focus", selected.slice(0, 8).map(path => `- ${filenameFromPath(path)}`).join("\n") + (selected.length > 8 ? `\n- ... ${selected.length - 8} more` : "")]);
        rows.push(["Evidence status", `${evidence.length} evidence item(s), ${answerSources.length} answer-source link(s), ${unresolved.length} open question(s).`]);
        if (comparison) rows.push(["What changed", comparisonTakeaway(comparison)]);
        rows.push(["Recommended next action", recommendedNextAction(trace, comparison)]);
      } else if (plan) {
        const firstRead = plan.first_read_paths || [];
        const firstReadLines = firstRead.length
          ? firstRead.slice(0, 8).map(path => `- ${filenameFromPath(path)}`).join("\n") + (firstRead.length > 8 ? `\n- ... ${firstRead.length - 8} more` : "")
          : "No narrow first-read set yet.";
        rows.push(["Planned first read", `${firstRead.length} document(s) selected first out of ${plan.discovered_count || "the discovered corpus"}.\nStart with:\n${firstReadLines}`]);
        rows.push(["Why these documents", plan.document_strategy || ((plan.source_planner || {}).reason) || "Irys ranked source paths against the question and document names."]);
        rows.push(["What you can change", "Uncheck irrelevant first-read documents, add missing files, or write a Plan Correction before running."]);
      }
      if (Array.isArray(events) && events.length) {
        const latest = events[events.length - 1] || {};
        const fields = latest.fields || {};
        rows.push(["Current step", `${fields.summary || friendlyEventTitle(latest)}\n${formatUserEventFields(fields) || "Working through the next step."}`]);
      }
      $("runBrief").innerHTML = rows.length
        ? rows.map(([title, body]) => briefCard(title, body)).join("")
        : emptyState("The brief will explain the current plan, what Irys is doing, and what you can change next.");
    }
    function renderReviewChecklist({plan = null, trace = null, events = null, comparison = null} = {}) {
      const target = $("reviewChecklist");
      if (!target) return;
      const cards = [];
      if (trace) {
        const packet = trace.final_packet || {};
        const diagnosis = trace.diagnosis || {};
        const metrics = trace.metrics || {};
        const metadata = (trace.task || {}).metadata || {};
        const scope = metadata.corpus_scope_decision || {};
        const docs = Array.isArray(trace.documents) ? trace.documents : [];
        const chunks = Array.isArray(trace.chunks) ? trace.chunks : [];
        const evidence = Array.isArray(packet.verified_evidence) ? packet.verified_evidence : [];
        const answerSources = Array.isArray(packet.answer_source_map) ? packet.answer_source_map : [];
        const unresolved = Array.isArray(packet.unresolved) ? packet.unresolved : [];
        const selected = Array.isArray(scope.selected_paths) ? scope.selected_paths : [];
        const loadErrors = docs.filter(doc => doc && doc.load_error);
        const coverage = diagnosis.source_coverage || packet.source_coverage || {};
        const missingDocs = Array.isArray(coverage.missing_documents) ? coverage.missing_documents : [];
        const ready = !loadErrors.length && !unresolved.length && evidence.length && (trace.rendered_answer || trace.draft_answer);
        const statusTone = loadErrors.length ? "bad" : ready ? "good" : "warn";
        cards.push(auditCard(
          "Ready to rely on?",
          ready ? "Ready for lawyer review" : "Needs review",
          loadErrors.length
            ? `${loadErrors.length} file(s) could not be read. Fix those before relying on the answer.`
            : ready
              ? "The answer has source evidence and no logged open questions. Review the sources below before using it."
              : "The run finished without enough support signals. Check evidence, source links, and open questions.",
          statusTone
        ));
        cards.push(auditCard(
          "Source focus",
          selected.length ? `${selected.length} first-read document(s)` : `${docs.length} document(s) read`,
          selected.length
            ? selected.slice(0, 5).map(filenameFromPath).join("\n") + (selected.length > 5 ? `\n... ${selected.length - 5} more` : "")
            : `${docs.length} document(s) and ${chunks.length} searchable passage(s) were loaded.`,
          selected.length || docs.length ? "good" : "warn"
        ));
        cards.push(auditCard(
          "Source support",
          `${evidence.length} evidence item(s)`,
          `${answerSources.length} answer-source link(s). ${missingDocs.length ? `${missingDocs.length} selected document(s) were not represented in retrieved evidence.` : "Selected source coverage is represented in the packet."}`,
          evidence.length && !missingDocs.length ? "good" : "warn"
        ));
        cards.push(auditCard(
          "Open questions",
          unresolved.length ? `${unresolved.length} needs review` : "None logged",
          unresolved.length ? unresolved.slice(0, 4).join("\n") + (unresolved.length > 4 ? `\n... ${unresolved.length - 4} more` : "") : "No unresolved gap was logged in the final packet.",
          unresolved.length ? "warn" : "good"
        ));
        cards.push(auditCard(
          "Cost",
          "$" + Number(metrics.estimated_cost || 0).toFixed(4),
          `${metrics.total_tokens || 0} token(s). Cheap worker share: ${formatPercent((metrics.token_share_by_tier || {}).cheap_worker)}. Strong synthesis share: ${formatPercent((metrics.token_share_by_tier || {}).strong_synthesizer)}.`,
          "neutral"
        ));
        cards.push(auditCard(
          "Next action",
          comparison && comparison.answer_changed ? "Compare runs" : "Review sources",
          recommendedNextAction(trace, comparison),
          unresolved.length || loadErrors.length ? "warn" : "neutral"
        ));
        target.innerHTML = cards.join("");
        return;
      }
      if (plan) {
        const firstRead = Array.isArray(plan.first_read_paths) ? plan.first_read_paths : [];
        const discovered = plan.discovered_count || firstRead.length || "the discovered corpus";
        const planner = plan.source_planner || {};
        cards.push(auditCard(
          "Plan state",
          "Review before running",
          "Check that the selected first-read documents match the question. Edit the list or add a plan correction if the focus is wrong.",
          "neutral"
        ));
        cards.push(auditCard(
          "First-read focus",
          `${firstRead.length} document(s)`,
          firstRead.length
            ? firstRead.slice(0, 6).map(filenameFromPath).join("\n") + (firstRead.length > 6 ? `\n... ${firstRead.length - 6} more` : "")
            : "No narrow first-read set has been selected yet.",
          firstRead.length ? "good" : "warn"
        ));
        cards.push(auditCard(
          "Corpus triage",
          `${firstRead.length} of ${discovered}`,
          plan.document_strategy || planner.reason || "Irys ranked the corpus against the objective and file names.",
          "neutral"
        ));
        cards.push(auditCard(
          "Next action",
          "Approve or correct",
          "If the plan is right, run it. If it missed the right source family, write a plan correction and preview again.",
          "neutral"
        ));
        target.innerHTML = cards.join("");
        return;
      }
      if (Array.isArray(events) && events.length) {
        const latest = events[events.length - 1] || {};
        target.innerHTML = auditCard("Current work", friendlyEventTitle(latest), formatUserEventFields(latest.fields || {}) || "Working through the run.", "neutral");
        return;
      }
      target.innerHTML = emptyState("After a plan or run, this checklist will show whether the answer has enough source support, what still needs review, and what to do next.");
    }
    function renderInvestigationMap({plan = null, trace = null, events = null, comparison = null} = {}) {
      const target = $("investigationMap");
      if (!target) return;
      const cards = [];
      if (trace && Object.keys(trace).length) {
        const packet = trace.final_packet || {};
        const metadata = (trace.task || {}).metadata || {};
        const scope = metadata.corpus_scope_decision || {};
        const docs = Array.isArray(trace.documents) ? trace.documents : [];
        const evidence = Array.isArray(packet.verified_evidence) ? packet.verified_evidence : [];
        const answerSources = Array.isArray(packet.answer_source_map) ? packet.answer_source_map : [];
        const unresolved = Array.isArray(packet.unresolved) ? packet.unresolved : [];
        const packetReview = packet.packet_review || {};
        const coverage = ((trace.diagnosis || {}).source_coverage) || packet.source_coverage || latestSourceCoverage(trace);
        const selected = Array.isArray(scope.selected_paths) ? scope.selected_paths : [];
        const objective = String((trace.task || {}).question || $("objective").value || "").trim();
        cards.push(mapCard(
          "Question",
          "What Irys is answering",
          objective || "No question was saved with this run.",
          "Review Answer",
          "review_answer"
        ));
        cards.push(mapCard(
          "Source plan",
          selected.length ? `${selected.length} first-read document(s)` : `${docs.length} loaded document(s)`,
          [
            scope.reason || "Irys used the active corpus and objective to decide what to read first.",
            selected.length ? "Read first:\n" + formatPathList(selected, 8) : "",
            scope.signals && scope.signals.requested_families ? "Likely source families: " + formatInlineList(scope.signals.requested_families, 8) : ""
          ].filter(Boolean).join("\n\n"),
          "Review Plan",
          "review_plan"
        ));
        cards.push(mapCard(
          "Evidence used",
          `${evidence.length} evidence item(s)`,
          formatEvidenceCoverageForMap(coverage, evidence, answerSources),
          "Review Sources",
          "review_sources"
        ));
        cards.push(mapCard(
          "Worker review",
          packetReviewTitle(packetReview),
          formatPacketReviewForMap(packetReview, docs),
          "Steer Next Pass",
          "focus_steering"
        ));
        cards.push(mapCard(
          "Held back",
          heldBackTitle(scope),
          formatHeldBackForMap(scope),
          "Adjust Sources",
          "review_plan"
        ));
        cards.push(mapCard(
          "Next correction",
          unresolved.length ? `${unresolved.length} open item(s)` : "Review and decide",
          [
            recommendedNextAction(trace, comparison),
            comparison ? comparisonTakeaway(comparison) : "",
            unresolved.length ? "Open items:\n" + formatSimpleList(unresolved, 5) : ""
          ].filter(Boolean).join("\n\n"),
          "Write Steering",
          "focus_steering"
        ));
        target.innerHTML = cards.join("");
        return;
      }
      if (plan) {
        const firstRead = Array.isArray(plan.first_read_paths) ? plan.first_read_paths : [];
        const planner = plan.source_planner || {};
        cards.push(mapCard(
          "Question",
          "Planned answer target",
          plan.interpreted_goal || $("objective").value || "Add the question or work product request.",
          "Edit Question",
          "focus_objective"
        ));
        cards.push(mapCard(
          "First read",
          `${firstRead.length} document(s) selected`,
          firstRead.length ? formatPathList(firstRead, 10) : "No first-read documents were selected yet.",
          "Review Plan",
          "review_plan"
        ));
        cards.push(mapCard(
          "Why these sources",
          planner.status ? "Worker-planned" : "Path-ranked",
          plan.document_strategy || planner.reason || "Irys ranked file names, paths, and source-role clues against the objective.",
          "Correct Plan",
          "review_plan"
        ));
        cards.push(mapCard(
          "Needed information",
          "What the answer must establish",
          formatSimpleList(plan.needed_information || [], 8) || "No separate needed-information list was returned.",
          "Edit Question",
          "focus_objective"
        ));
        cards.push(mapCard(
          "Held back",
          `${plan.not_read_first_count || 0} document(s) held back`,
          plan.not_read_first_count
            ? "Held-back documents remain available for the next pass if the packet review finds a gap."
            : "No held-back count was returned for this plan.",
          "Review Plan",
          "review_plan"
        ));
        cards.push(mapCard(
          "Next correction",
          "Approve or fix before reading",
          "If this source focus matches the question, run the approved plan. If it missed the right source family, write one correction and preview again.",
          "Correct Plan",
          "review_plan"
        ));
        target.innerHTML = cards.join("");
        return;
      }
      if (Array.isArray(events) && events.length) {
        const latest = events[events.length - 1] || {};
        target.innerHTML = mapCard(
          "Current work",
          friendlyEventTitle(latest),
          formatUserEventFields(latest.fields || {}) || "Irys is working through the current run.",
          "Steer",
          "focus_steering"
        );
        return;
      }
      target.innerHTML = emptyState("After planning or running, this will show the question, source choices, evidence coverage, worker review, held-back documents, and next correction path.");
    }
    function mapCard(label, title, body, buttonText, action) {
      const button = action
        ? `<button class="secondary map-action" data-action="${escapeAttr(action)}">${escapeHtml(buttonText || "Open")}</button>`
        : "";
      return `<div class="map-card"><b>${escapeHtml(label || "")}</b>` +
        `<strong>${escapeHtml(title || "")}</strong>` +
        `<small>${formatPlainText(body || "")}</small>${button}</div>`;
    }
    function latestSourceCoverage(trace) {
      const iterations = Array.isArray((trace || {}).retrieval_iterations) ? trace.retrieval_iterations : [];
      for (let index = iterations.length - 1; index >= 0; index -= 1) {
        if (iterations[index] && iterations[index].source_coverage) return iterations[index].source_coverage;
      }
      return {};
    }
    function formatPathList(paths, limit = 8) {
      const rows = Array.isArray(paths) ? paths : [];
      if (!rows.length) return "";
      return rows.slice(0, limit).map(path => `- ${filenameFromPath(path)}`).join("\n") +
        (rows.length > limit ? `\n- ... ${rows.length - limit} more` : "");
    }
    function formatDocRecordList(records, limit = 8) {
      const rows = Array.isArray(records) ? records : [];
      if (!rows.length) return "";
      return rows.slice(0, limit).map(item => `- ${item.filename || (item.path ? filenameFromPath(item.path) : "") || item.doc_id || "source"}`).join("\n") +
        (rows.length > limit ? `\n- ... ${rows.length - limit} more` : "");
    }
    function formatEvidenceCoverageForMap(coverage, evidence, answerSources) {
      const rows = coverage && typeof coverage === "object" ? coverage : {};
      const represented = Array.isArray(rows.represented_documents) ? rows.represented_documents : [];
      const missing = Array.isArray(rows.missing_documents) ? rows.missing_documents : [];
      const lines = [
        `${(evidence || []).length} evidence item(s) reached the answer packet.`,
        `${(answerSources || []).length} answer section link(s) connect final text back to sources.`
      ];
      if (represented.length) lines.push("Evidence came from:\n" + formatDocRecordList(represented, 7));
      if (missing.length) {
        const warning = sourceCoverageWarning(rows);
        if (warning) lines.push(warning);
        lines.push("Loaded but not represented in retrieved evidence:\n" + formatDocRecordList(missing, 5));
      }
      return lines.join("\n\n");
    }
    function packetReviewTitle(packetReview) {
      const review = packetReview || {};
      if (review.status === "not_run") return "No worker review";
      if (review.status === "error") return "Worker review failed";
      if (review.continue_retrieval === true) return "Worker requested more evidence";
      if (review.sufficient === false) return "Worker flagged gaps";
      if (review.status === "used") return "Worker cleared or narrowed packet";
      return review.status || "Review unavailable";
    }
    function formatPacketReviewForMap(packetReview, docs) {
      const review = packetReview || {};
      const lines = [];
      if (review.assessment) lines.push(String(review.assessment));
      if (review.continue_retrieval !== undefined) {
        lines.push(`Retrieval decision: ${review.continue_retrieval ? "look for more evidence before relying on the answer" : "current packet can proceed to drafting"}.`);
      }
      if (Array.isArray(review.missing_information) && review.missing_information.length) {
        lines.push("Missing information:\n" + formatSimpleList(review.missing_information, 6));
      }
      if (Array.isArray(review.coverage_risks) && review.coverage_risks.length) {
        lines.push("Coverage risks:\n" + formatSimpleList(review.coverage_risks, 6));
      }
      const relevant = sourceIdsToDocuments(review.relevant_source_ids || [], docs);
      if (relevant.length) lines.push("Reviewer kept:\n" + formatDocRecordList(relevant, 6));
      const lowValue = sourceIdsToDocuments(review.low_value_source_ids || [], docs);
      if (lowValue.length) lines.push("Reviewer marked lower value:\n" + formatDocRecordList(lowValue, 6));
      return lines.join("\n\n") || "No packet review details were saved for this run.";
    }
    function heldBackTitle(scope) {
      const discovered = Array.isArray((scope || {}).discovered_paths) ? scope.discovered_paths : [];
      const selected = Array.isArray((scope || {}).selected_paths) ? scope.selected_paths : [];
      const count = Math.max(0, discovered.length - selected.length);
      return count ? `${count} document(s) available later` : "No held-back inventory";
    }
    function formatHeldBackForMap(scope) {
      const discovered = Array.isArray((scope || {}).discovered_paths) ? scope.discovered_paths : [];
      const selected = new Set(((scope || {}).selected_paths || []).map(path => String(path || "").toLowerCase()));
      const heldBack = discovered.filter(path => !selected.has(String(path || "").toLowerCase()));
      if (!heldBack.length) return "No documents were held back from the first read.";
      const scoreByPath = {};
      for (const item of ((scope || {}).scored_paths || [])) {
        scoreByPath[String(item.path || "").toLowerCase()] = item;
      }
      heldBack.sort((left, right) => {
        const leftScore = Number((scoreByPath[String(left || "").toLowerCase()] || {}).score || 0);
        const rightScore = Number((scoreByPath[String(right || "").toLowerCase()] || {}).score || 0);
        return rightScore - leftScore || String(left || "").localeCompare(String(right || ""));
      });
      return formatPathList(heldBack, 8);
    }
    function briefCard(title, body) {
      return `<div class="item brief-card"><strong>${escapeHtml(title)}</strong><small>${formatPlainText(body || "")}</small></div>`;
    }
    function auditCard(label, value, body, tone = "neutral") {
      const safeTone = ["good", "warn", "bad", "neutral"].includes(tone) ? tone : "neutral";
      return `<div class="audit-card ${safeTone}"><b>${escapeHtml(label)}</b><strong>${escapeHtml(value || "")}</strong><small>${formatPlainText(body || "")}</small></div>`;
    }
    function formatPlainText(value) {
      return escapeHtml(value).replace(/\n/g, "<br>");
    }
    function formatPercent(value) {
      const number = Number(value || 0);
      if (!Number.isFinite(number) || number <= 0) return "0%";
      return Math.round(number * 100) + "%";
    }
    function recommendedNextAction(trace, comparison) {
      const packet = trace.final_packet || {};
      const docs = Array.isArray(trace.documents) ? trace.documents : [];
      const loadErrors = docs.filter(doc => doc && doc.load_error);
      const unresolved = Array.isArray(packet.unresolved) ? packet.unresolved : [];
      const evidence = Array.isArray(packet.verified_evidence) ? packet.verified_evidence : [];
      const answerSources = Array.isArray(packet.answer_source_map) ? packet.answer_source_map : [];
      if (loadErrors.length) return "Some files could not be read. Fix or remove those files, then rerun.";
      if (unresolved.length) return "Review the open questions. Add a nudge or more documents if any gap matters.";
      if (evidence.length && !answerSources.length) return "Evidence was found but answer-source links are thin. Review Sources Used before relying on the answer.";
      if (comparison && comparison.answer_changed) return "Review the run comparison before accepting the new answer.";
      return "Review the answer and Sources Used. If the focus is wrong, write a nudge and rerun.";
    }
    function renderComparison(comparison) {
      if (!comparison) return "";
      const documents = comparison.document_delta || {};
      const evidence = comparison.evidence_delta || {};
      const unresolved = comparison.unresolved_delta || {};
      const metrics = comparison.metrics_delta || {};
      const rows = [
        ["Run", `${comparison.parent_task_id || ""} -> ${comparison.child_task_id || ""}`],
        ["Plain-English takeaway", comparisonTakeaway(comparison)],
        ["Answer changed", String(Boolean(comparison.answer_changed))],
        ["Documents", `+${(documents.added || []).length} / -${(documents.removed || []).length} / kept ${documents.kept_count || 0}`],
        ["Evidence", `+${evidence.added_count || 0} / -${evidence.removed_count || 0} / kept ${evidence.kept_count || 0}`],
        ["Unresolved", `+${(unresolved.added || []).length} / -${(unresolved.removed || []).length} / kept ${unresolved.kept_count || 0}`],
        ["Tokens delta", String(metrics.total_tokens || 0)],
        ["Cost delta", "$" + Number(metrics.estimated_cost || 0).toFixed(4)]
      ];
      if (comparison.status === "unavailable") rows.push(["Comparison unavailable", comparison.error || "unknown error"]);
      if ((documents.added || []).length) rows.push(["Documents added", formatSimpleList(documents.added)]);
      if ((documents.removed || []).length) rows.push(["Documents removed", formatSimpleList(documents.removed)]);
      if ((evidence.added || []).length) rows.push(["New evidence", formatEvidenceDelta(evidence.added)]);
      if ((evidence.removed || []).length) rows.push(["Evidence no longer used", formatEvidenceDelta(evidence.removed)]);
      if ((unresolved.added || []).length) rows.push(["New open questions", formatSimpleList(unresolved.added)]);
      if ((unresolved.removed || []).length) rows.push(["Cleared open questions", formatSimpleList(unresolved.removed)]);
      return rows.map(([title, body]) => card(title, body)).join("");
    }
    function comparisonTakeaway(comparison) {
      if (!comparison) return "No prior run is loaded for comparison.";
      if (comparison.status === "unavailable") return `Comparison is unavailable: ${comparison.error || "unknown error"}`;
      const documents = comparison.document_delta || {};
      const evidence = comparison.evidence_delta || {};
      const unresolved = comparison.unresolved_delta || {};
      const metrics = comparison.metrics_delta || {};
      const addedDocs = (documents.added || []).length;
      const removedDocs = (documents.removed || []).length;
      const addedOpen = (unresolved.added || []).length;
      const removedOpen = (unresolved.removed || []).length;
      const lines = [];
      if (comparison.answer_changed) {
        lines.push("The answer changed. Review the changed answer against Sources Used before accepting it.");
      } else {
        lines.push("The answer text did not materially change.");
      }
      if (addedDocs || removedDocs) {
        lines.push(`Source focus changed: ${addedDocs} document(s) added and ${removedDocs} removed.`);
      }
      if ((evidence.added_count || 0) || (evidence.removed_count || 0)) {
        lines.push(`Evidence changed: ${evidence.added_count || 0} item(s) added and ${evidence.removed_count || 0} removed.`);
      }
      if (addedOpen || removedOpen) {
        lines.push(`Open questions changed: ${addedOpen} new and ${removedOpen} cleared.`);
      }
      if (Number(metrics.estimated_cost || 0) !== 0) {
        lines.push(`Cost delta: $${Number(metrics.estimated_cost || 0).toFixed(4)}.`);
      }
      return lines.join("\n");
    }
    function formatSimpleList(items, limit = 12) {
      return (items || []).slice(0, limit).map(item => `- ${item}`).join("\n") +
        ((items || []).length > limit ? `\n- ... ${(items || []).length - limit} more` : "");
    }
    function formatEvidenceDelta(items) {
      return (items || []).slice(0, 8).map(item => {
        const source = [item.doc_id, item.chunk_id].filter(Boolean).join(" / ") || "source";
        const support = item.support || item.claim || "";
        return `- ${source}: ${support}`;
      }).join("\n") + ((items || []).length > 8 ? `\n- ... ${(items || []).length - 8} more` : "");
    }
    function renderLiveEvents(events) {
      lastLiveEvents = Array.isArray(events) ? events : [];
      if (!Array.isArray(events) || !events.length) {
        $("liveEvents").innerHTML = "";
        $("runTimeline").innerHTML = "";
        $("currentStep").innerHTML = "<strong>Idle</strong><small>No run has started.</small>";
        updateCommandStep("Ready", "Choose a corpus, ask a question, review the source plan, then run.");
        if (currentPlan) renderRunBrief({plan: currentPlan});
        if (currentPlan) renderReviewChecklist({plan: currentPlan});
        if (currentPlan) renderInvestigationMap({plan: currentPlan});
        renderActionBoard();
        return;
      }
      const latest = events[events.length - 1] || {};
      $("runTimeline").innerHTML = renderRunTimeline(events);
      $("currentStep").innerHTML = renderCurrentStep(latest);
      $("liveEvents").innerHTML = renderRecentLiveEvents(events);
      renderRunBrief({plan: currentPlan, events});
      renderReviewChecklist({plan: currentPlan, events});
      renderInvestigationMap({plan: currentPlan, trace: lastRenderedTrace, events});
      renderActionBoard();
    }
    function renderRecentLiveEvents(events) {
      const rows = Array.isArray(events) ? events : [];
      const visibleLimit = 8;
      const visible = rows.slice(-visibleLimit).map(renderUserEvent);
      if (rows.length > visibleLimit) {
        visible.unshift(emptyState(`${rows.length - visibleLimit} earlier update(s) are saved in the trace and advanced diagnostic data.`));
      }
      return visible.join("");
    }
    function renderRunTimeline(events) {
      const stages = [
        {key: "scope", title: "Select sources", detail: "Choose the first documents to read.", labels: ["SCOPE", "STEER"]},
        {key: "read", title: "Read documents", detail: "Extract text and source passages from the active corpus.", labels: ["READ", "LOAD"]},
        {key: "plan", title: "Plan answer", detail: "Clarify the answer target and search needs.", labels: ["PLAN", "CONTRACT"]},
        {key: "search", title: "Find evidence", detail: "Search for relevant passages and source support.", labels: ["SEARCH", "EVIDENCE", "EXTRACT"]},
        {key: "analyze", title: "Organize evidence", detail: "Use worker notes where live synthesis is enabled.", labels: ["ANALYZE"]},
        {key: "draft", title: "Draft answer", detail: "Create the user-facing answer or work product.", labels: ["SYNTH"]},
        {key: "save", title: "Save trace", detail: "Write the answer, artifacts, costs, and trace.", labels: ["SAVE", "DONE"]}
      ];
      const labelToIndex = {};
      stages.forEach((stage, index) => stage.labels.forEach(label => { labelToIndex[label] = index; }));
      let latestIndex = -1;
      let hasError = false;
      for (const event of events || []) {
        const label = String(event.label || "EVENT");
        if (label === "ERROR" || label === "STOP") hasError = true;
        if (labelToIndex[label] !== undefined) latestIndex = Math.max(latestIndex, labelToIndex[label]);
      }
      const latestEvent = (events || [])[events.length - 1] || {};
      const latestLabel = String(latestEvent.label || "");
      const runComplete = latestLabel === "DONE" || isRunCompleteEvent(latestEvent);
      const activeIndex = Math.max(0, latestIndex);
      return stages.map((stage, index) => {
        let state = "pending";
        if (index < activeIndex || runComplete) state = "done";
        if (index === activeIndex && !runComplete) state = hasError ? "error" : "active";
        const status = state === "done" ? "Done" : state === "active" ? "Now" : state === "error" ? "Needs attention" : "Waiting";
        return `<div class="timeline-step ${state}"><b>${escapeHtml(stage.title)}</b><span>${escapeHtml(status + ": " + stage.detail)}</span></div>`;
      }).join("");
    }
    function renderCurrentStep(event) {
      const fields = event.fields || {};
      const title = fields.summary || friendlyEventTitle(event);
      const coverageWarning = sourceCoverageWarning(fields.source_coverage);
      const details = isRunCompleteEvent(event)
        ? "Review the answer, sources, and trace."
        : coverageWarning || (fields.next_step ? `Next: ${fields.next_step}` : "Waiting for the next update.");
      updateCommandStep(title, details);
      return `<strong>${escapeHtml(title)}</strong><small>${escapeHtml(details)}</small>`;
    }
    function isRunCompleteEvent(event) {
      const label = String((event || {}).label || "");
      const message = String((event || {}).message || "").toLowerCase();
      const summary = String((((event || {}).fields || {}).summary) || "").toLowerCase();
      return label === "DONE" || (label === "SYNTH" && (message.includes("generated") || summary.includes("final answer generated")));
    }
    function renderUserEvent(event) {
      const fields = event.fields || {};
      const title = fields.summary || friendlyEventTitle(event);
      const body = formatUserEventFields(fields);
      return card(title, body || friendlyEventTitle(event));
    }
    function friendlyEventTitle(event) {
      const label = String(event.label || "EVENT");
      const message = String(event.message || "");
      const titles = {
        SCOPE: "Checking the selected corpus.",
        RUN: "Starting the run.",
        READ: "Reading documents.",
        LOAD: "Loaded the corpus.",
        PLAN: "Planning the answer.",
        CONTRACT: "Planning the answer.",
        SEARCH: "Looking for relevant passages.",
        EVIDENCE: "Building the evidence packet.",
        EXTRACT: "Building the evidence packet.",
        ANALYZE: "Organizing evidence.",
        SYNTH: "Drafting the answer.",
        SAVE: "Saving the run.",
        DONE: "Run complete.",
        STOP: "Run stopped.",
        STEER: "Applying your steering note.",
        ERROR: "Run failed."
      };
      return titles[label] || `${label}${message ? ": " + message : ""}`;
    }
    function formatUserEventFields(fields) {
      const lines = [];
      if (fields.filename) lines.push(`Document: ${fields.filename}`);
      if (fields.current && fields.total) lines.push(`Progress: ${fields.current} of ${fields.total}`);
      if (fields.document_count !== undefined) lines.push(`Documents found: ${fields.document_count}`);
      if (fields.documents !== undefined) lines.push(`Documents loaded: ${fields.documents}`);
      if (fields.chunks !== undefined) lines.push(`Searchable passages: ${fields.chunks}`);
      if (fields.chunk_count !== undefined) lines.push(`Passages from this document: ${fields.chunk_count}`);
      if (fields.text_chars !== undefined) lines.push(`Extracted characters: ${fields.text_chars}`);
      if (fields.load_error) lines.push(`Load issue: ${fields.load_error}`);
      if (fields.error) lines.push(`Error: ${fields.error}`);
      if (fields.detail) lines.push(`Detail: ${fields.detail}`);
      if (fields.user_nudge) lines.push(`Your instruction: ${fields.user_nudge}`);
      if (fields.planner) lines.push(`Planner: ${fields.planner === "cheap_worker" ? "cheap worker source planner" : fields.planner}`);
      if (fields.source_selection_mode) lines.push(`Source plan: ${formatSourceSelectionMode(fields.source_selection_mode)}`);
      if (fields.reason) lines.push(`Why: ${fields.reason}`);
      if (fields.continue_retrieval !== undefined) {
        lines.push(`Retrieval decision: ${fields.continue_retrieval ? "look for more evidence before drafting" : "current packet is enough to draft"}`);
      }
      if (fields.search_queries) lines.push(`Search targets: ${formatInlineList(fields.search_queries, 12)}`);
      if (fields.queries) lines.push(`Search targets: ${formatInlineList(fields.queries, 12)}`);
      if (fields.revised_queries) lines.push(`Revised search targets: ${formatInlineList(fields.revised_queries, 12)}`);
      if (fields.needed_information) lines.push(`Plan needs: ${formatInlineList(fields.needed_information, 12)}`);
      if (fields.missing_information) lines.push(`Missing information: ${formatInlineList(fields.missing_information, 12)}`);
      if (fields.coverage_risks) lines.push(`Coverage risks: ${formatInlineList(fields.coverage_risks, 12)}`);
      if (fields.relevant_source_ids) lines.push(`Relevant source IDs: ${formatInlineList(fields.relevant_source_ids, 16)}`);
      if (fields.low_value_source_ids) lines.push(`Low-value source IDs: ${formatInlineList(fields.low_value_source_ids, 16)}`);
      if (fields.selected_documents) lines.push(`Reading first: ${formatInlineList(fields.selected_documents, 12)}`);
      if (fields.held_back_documents) lines.push(`Held-back examples: ${formatInlineList(fields.held_back_documents, 12)}`);
      if (fields.pinned_documents) lines.push(`Pinned documents: ${formatInlineList(fields.pinned_documents, 12)}`);
      if (fields.skipped_document_count) lines.push(`Held back for now: ${fields.skipped_document_count} document(s).`);
      if (fields.source_coverage) {
        const warning = sourceCoverageWarning(fields.source_coverage);
        if (warning) lines.push(warning);
        lines.push(`Source coverage: ${formatSourceCoverageSummary(fields.source_coverage)}`);
      }
      if (fields.selected_sources) {
        lines.push("Sources selected:");
        for (const source of fields.selected_sources.slice(0, 12)) {
          lines.push(`- ${source.document || source.chunk_id}: ${source.preview || ""}`);
        }
        if (fields.selected_sources.length > 12) lines.push(`- ... ${fields.selected_sources.length - 12} more saved in trace`);
      }
      if (fields.evidence_preview) {
        lines.push("Evidence found:");
        for (const item of fields.evidence_preview.slice(0, 12)) {
          lines.push(`- ${item.support || item.claim || ""}`);
        }
        if (fields.evidence_preview.length > 12) lines.push(`- ... ${fields.evidence_preview.length - 12} more saved in trace`);
      }
      if (fields.analysis_preview) lines.push(`Worker notes: ${fields.analysis_preview}`);
      if (fields.answer_preview) lines.push(`Draft preview: ${fields.answer_preview}`);
      if (fields.sample_documents) lines.push(`Examples: ${formatInlineList(fields.sample_documents, 12)}`);
      if (fields.omitted_document_count) lines.push(`Plus ${fields.omitted_document_count} more.`);
      if (fields.steer_hint) lines.push(`Steer: ${fields.steer_hint}`);
      if (fields.next_step) lines.push(`Next: ${fields.next_step}`);
      if (!lines.length) {
        for (const [key, value] of Object.entries(fields)) {
          if (key === "summary") continue;
          lines.push(`${key}: ${typeof value === "string" ? limitDisplayText(value, 2000) : compactJsonForDisplay(value)}`);
        }
      }
      return lines.join("\n");
    }
    function formatSourceSelectionMode(mode) {
      if (mode === "replan_from_nudge") return "re-planning from your steering note";
      if (mode === "locked_by_user") return "using the first-read documents you locked";
      return String(mode || "");
    }
    function formatSourceCoverageSummary(coverage) {
      const rows = coverage && typeof coverage === "object" ? coverage : {};
      const representedDocs = Array.isArray(rows.represented_documents) ? rows.represented_documents : [];
      const missingDocs = Array.isArray(rows.missing_documents) ? rows.missing_documents : [];
      const covered = Number(rows.represented_document_count || representedDocs.length || 0);
      const uncovered = Number(rows.missing_document_count || missingDocs.length || 0);
      const parts = [];
      if (covered || uncovered) parts.push(`${covered} covered, ${uncovered} not retrieved`);
      if (representedDocs.length) {
        parts.push("retrieved: " + representedDocs.slice(0, 6).map(item => item.filename || item.doc_id || item.document || "").filter(Boolean).join(", "));
      }
      if (missingDocs.length) {
        parts.push("not retrieved: " + missingDocs.slice(0, 6).map(item => item.filename || item.doc_id || item.document || item).filter(Boolean).join(", "));
      }
      return parts.join("\n") || compactJsonForDisplay(coverage);
    }
    function sourceCoverageWarning(coverage) {
      const rows = coverage && typeof coverage === "object" ? coverage : {};
      const covered = Number(rows.represented_document_count || 0);
      const uncovered = Number(rows.missing_document_count || 0);
      const loaded = Number(rows.loaded_document_count || covered + uncovered || 0);
      if (loaded > 1 && uncovered > 0) {
        return `Coverage warning: ${uncovered} of ${loaded} loaded document(s) did not contribute retrieved evidence. Review Sources Used before accepting an absence claim.`;
      }
      return "";
    }
    function activeChatKey() {
      return `${$("matter").value || "local-matter"}::${$("chat").value || "main"}`;
    }
    function activeConversationHistory() {
      return conversationByChat[activeChatKey()] || [];
    }
    function updateConversationHistoryFromTrace(trace) {
      const packet = trace.final_packet || {};
      const metadata = ((trace.task || {}).metadata || {});
      const key = `${metadata.matter_id || (trace.task || {}).task_id || $("matter").value || "local-matter"}::${metadata.chat_id || $("chat").value || "main"}`;
      const history = Array.isArray(packet.conversation_history) ? [...packet.conversation_history] : [];
      const user = (trace.task || {}).question || "";
      const assistant = trace.rendered_answer || "";
      if (user || assistant) history.push({user, assistant});
      conversationByChat[key] = history;
    }
    function renderChatHistory(history) {
      if (!Array.isArray(history) || !history.length) return "";
      return history.map((turn, index) => card(
        "Turn " + (index + 1),
        "User: " + (turn.user || "") + "\n\nFinal answer: " + (turn.assistant || "")
      )).join("");
    }
    function renderTraceList(traces) {
      if (!Array.isArray(traces) || !traces.length) return "";
      return traces.map(trace => {
        const title = `${trace.chat_id || "main"} - ${trace.started_at || ""}`;
        const body = `${trace.question || ""}\n${trace.path || ""}\n` +
          `tokens=${trace.total_tokens || 0} cost=$${Number(trace.estimated_cost || 0).toFixed(4)}`;
        return `<div class="item"><strong>${escapeHtml(title)}</strong><small><pre>${escapeHtml(body)}</pre></small>` +
          `<button class="secondary trace-load" data-path="${escapeAttr(trace.path || "")}">Load</button></div>`;
      }).join("");
    }
    function renderMarkdown(markdown) {
      const lines = String(markdown || "").split(/\r?\n/);
      const html = [];
      let listType = "";
      let paragraph = [];
      const closeList = () => {
        if (listType) {
          html.push(`</${listType}>`);
          listType = "";
        }
      };
      const flushParagraph = () => {
        if (paragraph.length) {
          html.push(`<p>${formatInline(paragraph.join(" "))}</p>`);
          paragraph = [];
        }
      };
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          flushParagraph();
          closeList();
          continue;
        }
        const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
          flushParagraph();
          closeList();
          const level = heading[1].length;
          html.push(`<h${level}>${formatInline(heading[2])}</h${level}>`);
          continue;
        }
        const bullet = trimmed.match(/^[-*]\s+(.+)$/);
        if (bullet) {
          flushParagraph();
          if (listType !== "ul") {
            closeList();
            listType = "ul";
            html.push("<ul>");
          }
          html.push(`<li>${formatInline(bullet[1])}</li>`);
          continue;
        }
        const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
        if (numbered) {
          flushParagraph();
          if (listType !== "ol") {
            closeList();
            listType = "ol";
            html.push("<ol>");
          }
          html.push(`<li>${formatInline(numbered[1])}</li>`);
          continue;
        }
        paragraph.push(trimmed);
      }
      flushParagraph();
      closeList();
      return html.join("");
    }
    function formatInline(value) {
      return escapeHtml(value)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>");
    }
    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }
    function escapeAttr(value) {
      return escapeHtml(value).replaceAll("'", "&#39;");
    }
  </script>
</body>
</html>
"""
