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
                    response = {"summary": trace_summary(trace), "trace": trace}
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
                            top_k=int(payload.get("top_k", 12) or 12),
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
                    top_k=int(payload.get("top_k", 12) or 12),
                    max_files=parse_optional_int(payload.get("max_files")),
                    selected_paths=parse_paths(payload.get("selected_paths", [])) or None,
                    plan_note=str(payload.get("plan_note") or ""),
                    use_llm_planning=bool(payload.get("use_llm_planning", False)),
                    verbose=False,
                )
                response = result.to_dict()
                response["summary"] = trace_summary(result.state.to_trace())
                response["trace"] = result.state.to_trace()
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
                    top_k=int(payload.get("top_k", 12) or 12),
                    max_files=parse_optional_int(payload.get("max_files")),
                    verbose=False,
                    event_callback=event_callback,
                    should_cancel=should_cancel,
                    selected_paths=parse_paths(payload.get("selected_paths", [])) or None,
                    plan_note=str(payload.get("plan_note") or ""),
                    use_llm_planning=bool(payload.get("use_llm_planning", False)),
                )
                response = result.to_dict()
                response["summary"] = trace_summary(result.state.to_trace())
                response["trace"] = result.state.to_trace()
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
            append_run_job_event(job_id, "ERROR", "product matter run failed", error=error)
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
        top_k=int(payload.get("top_k", 12) or 12),
        max_files=parse_optional_int(payload.get("max_files")),
        verbose=False,
        parent_trace_path=str(parent_path),
        user_nudge=nudge,
        selected_paths=selected_paths_for_rerun_payload(payload),
        plan_note=plan_note,
        use_llm_planning=bool(payload.get("use_llm_planning", False)),
        event_callback=event_callback,
        should_cancel=should_cancel,
    )
    child_trace = result.state.to_trace()
    response = result.to_dict()
    response["summary"] = trace_summary(child_trace)
    response["trace"] = child_trace
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
        top_k=int(payload.get("top_k", 12) or 12),
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
    }
    h1 {
      font-size: 18px;
      font-weight: 700;
      margin: 0;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(420px, 1fr) minmax(360px, 0.85fr);
      min-height: calc(100vh - 56px);
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
    .control-help {
      margin-top: 10px;
      background: #f7f9fb;
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
      main { grid-template-columns: 1fr; }
      section { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Irys Matter Runner</h1>
    <div class="status" id="status"></div>
  </header>
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
          Folders are read recursively. No artificial file-count or per-document character cap is applied to local corpus paths. Matter ID groups saved runs and costs. Chat ID keeps separate conversations inside the same matter. Smart source planning reviews the file inventory before the first read and falls back to path scoring if model planning is unavailable. Draft final answer is on by default and calls the configured drafting model; turn it off only for a cheap dry run. Evidence chunks controls how many retrieved chunks are used for the answer packet. Message Cost is the loaded run; Matter Cost totals saved traces for the matter.
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
        <label for="topk">Evidence chunks</label>
        <input id="topk" type="number" min="1" max="50" value="12" />
      </div>
      <div class="row">
        <button id="run">Run Approved Plan</button>
        <button class="secondary" id="stopRun" disabled>Stop Run</button>
        <button class="secondary" id="clear">Clear</button>
      </div>
    </section>
    <section>
      <h2>Objective</h2>
      <textarea id="objective" class="objective"></textarea>
      <div class="row">
        <button class="secondary" id="planRun">Review Plan</button>
      </div>
      <h2 style="margin-top:16px">Run Brief</h2>
      <div class="list" id="runBrief">
        <div class="empty">The brief will explain the current plan, what Irys is doing, and what you can change next.</div>
      </div>
      <h2 style="margin-top:16px">First-Read Review</h2>
      <div class="list" id="candidateReview">
        <div class="empty">Review Plan will show the highest-ranked documents here. Check or uncheck what Irys should read first.</div>
      </div>
      <div class="row">
        <button class="secondary" id="applyCheckedCandidates">Apply Checked Documents</button>
        <button class="secondary" id="selectAllCandidates">Select All Shown</button>
        <button class="secondary" id="restoreRecommendedCandidates">Restore Recommended</button>
      </div>
      <label for="firstReadPaths">First-Read Documents</label>
      <textarea id="firstReadPaths" placeholder="Review Plan fills this with the files Irys intends to read first. Edit it before running if the plan is wrong."></textarea>
      <label for="planNote">Plan Correction</label>
      <textarea id="planNote" placeholder="Correct the plan here: focus on 10-Ks, start with the latest amendment, ignore draft folders, include quarterly reports, or compare all agreements."></textarea>
      <h2 style="margin-top:16px">Detailed Plan</h2>
      <div class="list" id="planPreview">
        <div class="empty">Choose a corpus and objective, then review the plan before running.</div>
      </div>
      <div class="metric-grid">
        <div class="metric"><b>Documents</b><span id="docCount">0</span></div>
        <div class="metric"><b>Chunks</b><span id="chunkCount">0</span></div>
        <div class="metric"><b>Tokens</b><span id="tokens">0</span></div>
        <div class="metric"><b>Message Cost</b><span id="cost">$0.00</span></div>
        <div class="metric"><b>Matter Cost</b><span id="matterCost">$0.00</span></div>
        <div class="metric"><b>Matter Messages</b><span id="matterMessages">0</span></div>
      </div>
      <h2>Answer</h2>
      <div class="answer" id="answer"></div>
      <h2 style="margin-top:16px">Source Review</h2>
      <div class="list" id="sourceSummary"></div>
      <h2 style="margin-top:16px">Documents Held Back</h2>
      <div class="list" id="heldBackSources"></div>
      <h2 style="margin-top:16px">Sources Used</h2>
      <div class="list" id="sourcesUsed"></div>
      <h2 style="margin-top:16px">Open Questions</h2>
      <div class="list" id="openQuestions"></div>
      <h2 style="margin-top:16px">Chat History</h2>
      <div class="list" id="chatHistory"></div>
    </section>
    <section>
      <h2>Workstream</h2>
      <div class="item" id="currentStep"><strong>Idle</strong><small>No run has started.</small></div>
      <h2 style="margin-top:16px">What Irys Is Doing</h2>
      <div class="list" id="liveEvents"></div>
      <label for="nudge">Steer the Next Pass</label>
      <textarea id="nudge" placeholder="Example: focus on the 2024 10-K only, ignore Form 4s, compare against the latest amendment, read the guaranty more closely, or answer only the EPS question."></textarea>
      <div class="hint">This keeps the same matter and reruns with your correction. By default the next pass re-plans which files to read; edit or apply First-Read Documents only when you want to force a specific source set.</div>
      <label for="rerunPaths">Additional Corpus Paths</label>
      <textarea id="rerunPaths" placeholder="Optional: add another local file or folder path for the next pass."></textarea>
      <div class="row">
        <button class="secondary" id="chooseRerunFolder">Choose Folder</button>
        <button class="secondary" id="chooseRerunFile">Choose File</button>
        <button class="secondary" id="chooseRerunFiles">Choose Files</button>
      </div>
      <div class="row">
        <button class="secondary" id="previewNudgePlan">Preview Nudge Plan</button>
        <button class="secondary" id="rerunTrace">Apply Nudge</button>
      </div>
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
    const stopRun = $("stopRun");
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
    let firstReadPathsDirty = false;
    let suppressFirstReadDirty = false;
    let excludedSourcePaths = [];
    $("clear").addEventListener("click", () => {
      $("objective").value = "";
      $("answer").innerHTML = "";
      $("chatHistory").innerHTML = "";
      $("tracepath").value = "";
      $("nudge").value = "";
      $("rerunPaths").value = "";
      setFirstReadPaths([], {dirty: false});
      $("planNote").value = "";
      $("planPreview").innerHTML = emptyState("Choose a corpus and objective, then review the plan before running.");
      $("runBrief").innerHTML = emptyState("The brief will explain the current plan, what Irys is doing, and what you can change next.");
      $("candidateReview").innerHTML = emptyState("Review Plan will show the highest-ranked documents here. Check or uncheck what Irys should read first.");
      currentPlan = null;
      currentPlanNote = "";
      currentPlanObjective = "";
      excludedSourcePaths = [];
      $("diagnosis").innerHTML = "";
      $("currentStep").innerHTML = "<strong>Idle</strong><small>No run has started.</small>";
      $("liveEvents").innerHTML = "";
      $("sourceSummary").innerHTML = "";
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
      status.textContent = "";
    });
    $("firstReadPaths").addEventListener("input", () => {
      if (!suppressFirstReadDirty) firstReadPathsDirty = true;
    });
    stopRun.addEventListener("click", async () => {
      if (!activeJobId) return;
      stopRun.disabled = true;
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
      status.textContent = "Planning";
      try {
        const plan = await requestPlan({paths: pathPayload($("paths").value)});
        currentPlan = plan;
        renderPlan(plan);
        status.textContent = `Plan ready: ${plan.first_read_count || 0} first-read document(s)`;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        planRun.disabled = false;
      }
    });
    run.addEventListener("click", async () => {
      run.disabled = true;
      status.textContent = "Running";
      try {
        if (planNeedsRefresh()) {
          const plan = await requestPlan({paths: pathPayload($("paths").value)});
          currentPlan = plan;
          renderPlan(plan);
        }
        applyCheckedCandidatePaths({silent: true});
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
            top_k: Number($("topk").value || 12),
            selected_paths: pathPayload($("firstReadPaths").value),
            plan_note: $("planNote").value
          })
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Run failed");
        activeJobId = data.job_id || "";
        stopRun.disabled = !activeJobId;
        renderLiveEvents(data.events || []);
        await pollRunJob(data.job_id);
      } catch (error) {
        status.textContent = error.message;
      } finally {
        run.disabled = false;
        stopRun.disabled = true;
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
        status.textContent = $("tracepath").value;
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
      status.textContent = "Rerunning";
      try {
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
            top_k: Number($("topk").value || 12),
            selected_paths: firstReadPathsDirty ? pathPayload($("firstReadPaths").value) : [],
            selected_paths_locked: firstReadPathsDirty,
            excluded_paths: excludedSourcePaths,
            plan_note: $("planNote").value
          })
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Rerun failed");
        activeJobId = data.job_id || "";
        stopRun.disabled = !activeJobId;
        renderLiveEvents(data.events || []);
        await pollRunJob(data.job_id);
      } catch (error) {
        status.textContent = error.message;
      } finally {
        rerunTrace.disabled = false;
        stopRun.disabled = true;
      }
    });
    previewNudgePlan.addEventListener("click", async () => {
      previewNudgePlan.disabled = true;
      status.textContent = "Previewing nudge plan";
      try {
        const response = await fetch("/api/rerun-plan", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            trace_path: $("tracepath").value,
            nudge: $("nudge").value,
            paths: pathPayload($("rerunPaths").value),
            use_llm_planning: $("usePlanner").checked,
            top_k: Number($("topk").value || 12),
            selected_paths: firstReadPathsDirty ? pathPayload($("firstReadPaths").value) : [],
            selected_paths_locked: firstReadPathsDirty,
            excluded_paths: excludedSourcePaths,
            plan_note: $("planNote").value
          })
        });
        const plan = await response.json();
        if (!response.ok || plan.error) throw new Error(plan.error || "Nudge plan failed");
        currentPlan = plan;
        renderPlan(plan);
        status.textContent = `Nudge plan ready: ${plan.first_read_count || 0} first-read document(s)`;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        previewNudgePlan.disabled = false;
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
    }
    function setFirstReadPaths(paths, {dirty = false} = {}) {
      suppressFirstReadDirty = true;
      $("firstReadPaths").value = (paths || []).join("\n");
      suppressFirstReadDirty = false;
      firstReadPathsDirty = dirty;
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
          top_k: Number($("topk").value || 12)
        })
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || "Plan failed");
      return data;
    }
    function renderPlan(plan) {
      const firstRead = plan.first_read_paths || [];
      setFirstReadPaths(firstRead, {dirty: false});
      currentPlan = plan;
      currentPlanNote = $("planNote").value;
      currentPlanObjective = $("objective").value;
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
    }
    function planNeedsRefresh() {
      if (!pathPayload($("firstReadPaths").value).length) return true;
      if (!currentPlan) return false;
      return currentPlanNote !== $("planNote").value || currentPlanObjective !== $("objective").value;
    }
    function renderTracePlan(metadata, trace = null) {
      const scope = metadata.corpus_scope_decision || null;
      if (!scope) return;
      const selected = scope.selected_paths || [];
      setFirstReadPaths(selected, {dirty: false});
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
        return;
      }
      setFirstReadPaths(checked, {dirty: true});
      if (!silent) status.textContent = `Applied ${checked.length} first-read document(s)`;
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
        if (!$("nudge").value.trim()) $("nudge").value = `Read ${filenameFromPath(path)} in the next pass.`;
        status.textContent = `Added ${filenameFromPath(path)} to the next first-read set`;
        return;
      }
      if (target.matches(".ignore-source-next")) {
        const path = target.getAttribute("data-path") || "";
        addExcludedSourcePath(path);
        removePathFromFirstRead(path);
        firstReadPathsDirty = true;
        if (!$("nudge").value.trim()) $("nudge").value = `Ignore ${filenameFromPath(path)} in the next pass.`;
        status.textContent = `Marked ${filenameFromPath(path)} to ignore next pass`;
      }
    });
    function render(data) {
      const trace = data.trace || {};
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
      renderTracePlan(metadata, trace);
      excludedSourcePaths = [];
      renderRunBrief({trace, comparison: data.comparison});
      $("answer").innerHTML = renderMarkdown(data.rendered_answer || trace.rendered_answer || "");
      updateConversationHistoryFromTrace(trace);
      $("chatHistory").innerHTML = renderChatHistory(activeConversationHistory());
      renderSourceReview(trace);
      $("diagnosis").innerHTML = renderRunHealth(trace);
      $("comparison").innerHTML = renderComparison(data.comparison);
      renderLiveEvents(trace.events || []);
      $("events").innerHTML = (trace.events || []).map(event => card(
        event.label + " - " + event.message,
        JSON.stringify(event.fields || {}, null, 2)
      )).join("");
      $("documents").innerHTML = (trace.documents || []).map(doc => card(
        (doc.doc_id || "") + " - " + (doc.filename || ""),
        (doc.path || "") + "\nchars=" + (doc.text_chars || 0) + (doc.load_error ? "\nerror=" + doc.load_error : "")
      )).join("");
      $("artifacts").innerHTML = (trace.artifacts || []).map(artifact => card(
        artifact.filename || artifact.type || "Artifact",
        (artifact.path || "") + "\ntype=" + (artifact.type || "") + "\ndiagnostic=" + Boolean(artifact.diagnostic)
      )).join("");
      const evidence = ((trace.final_packet || {}).verified_evidence || []);
      $("evidence").innerHTML = evidence.map(item => card(
        item.claim || "Evidence",
        (item.raw_support || "") + "\n" + JSON.stringify(item.source || {}, null, 2)
      )).join("");
      const answerSources = ((trace.final_packet || {}).answer_source_map || []);
      $("answerSources").innerHTML = answerSources.map(item => card(
        "Section " + (item.section_index || ""),
        (item.answer_excerpt || "") + "\n" + JSON.stringify(item.source_refs || [], null, 2)
      )).join("");
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
      return `<div class="item"><strong>${escapeHtml(title)}</strong><small><pre>${escapeHtml(body || "")}</pre></small></div>`;
    }
    function emptyState(text) {
      return `<div class="empty">${escapeHtml(text)}</div>`;
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
      const loadErrors = docs.filter(doc => doc && doc.load_error);
      const retrieved = Array.isArray(packet.retrieved_chunks) ? packet.retrieved_chunks : [];
      const summaryRows = [
        ["Corpus read", `${docs.length} document(s), ${chunks.length} searchable chunk(s).`],
        ["Evidence found", `${evidence.length} source item(s) carried into the answer packet.`],
        ["Sources selected", `${retrieved.length} retrieved passage(s) were considered.`],
        ["Open questions", unresolved.length ? `${unresolved.length} issue(s) still flagged.` : "No obvious gaps were logged."]
      ];
      if (scope.reason) {
        const planner = (scope.signals || {}).source_planner || null;
        summaryRows.splice(1, 0, ["First-read plan", scope.reason]);
        if (planner) summaryRows.splice(2, 0, ["Worker source planner", formatPlannerSummary(planner, {})]);
      }
      if (loadErrors.length) {
        summaryRows.push(["Load issues", loadErrors.map(doc => `${doc.filename || doc.path}: ${doc.load_error}`).join("\n")]);
      }
      $("sourceSummary").innerHTML = summaryRows.map(([title, body]) => card(title, body)).join("");
      $("heldBackSources").innerHTML = renderHeldBackSources(scope);
      $("sourcesUsed").innerHTML = renderSourcesUsed(answerSources, evidence, docs);
      $("openQuestions").innerHTML = unresolved.length
        ? unresolved.map((item, index) => card(`Open question ${index + 1}`, item)).join("")
        : emptyState("No open question was logged for this run.");
    }
    function renderSourcesUsed(answerSources, evidence, docs) {
      if (answerSources.length) {
        return answerSources.map((item, index) => {
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
        }).join("");
      }
      if (evidence.length) {
        return evidence.map((item, index) => {
          const source = item.source || {};
          const docName = docNameFor(source.doc_id, docs) || source.doc_id || "source";
          return sourceCard(
            `Source ${index + 1}: ${docName}`,
            `${item.raw_support || item.claim || ""}\n${source.chunk_id ? "Chunk: " + source.chunk_id : ""}`,
            docPathFor(source.doc_id, docs)
          );
        }).join("");
      }
      return emptyState("No source support was captured for this answer.");
    }
    function sourceCard(title, body, path) {
      const action = path
        ? `<button class="secondary ignore-source-next" data-path="${escapeAttr(path)}">Ignore next pass</button>`
        : "";
      return `<div class="item"><strong>${escapeHtml(title)}</strong><small><pre>${escapeHtml(body || "")}</pre></small>${action}</div>`;
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
    }
    function removeExcludedSourcePath(path) {
      const key = String(path || "").toLowerCase();
      excludedSourcePaths = excludedSourcePaths.filter(item => String(item || "").toLowerCase() !== key);
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
        ["Cost", `$${Number(metrics.estimated_cost || 0).toFixed(4)} for this message.`]
      ];
      const unresolved = Array.isArray(packet.unresolved) ? packet.unresolved : [];
      if (unresolved.length) rows.push(["Needs review", unresolved.join("\n")]);
      return rows.map(([title, body]) => card(title, body)).join("");
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
        rows.push(["Corpus read", `${docs.length} document(s), ${chunks.length} searchable chunk(s).`]);
        if (selected.length) rows.push(["First-read focus", selected.slice(0, 8).map(path => `- ${filenameFromPath(path)}`).join("\n") + (selected.length > 8 ? `\n- ... ${selected.length - 8} more` : "")]);
        rows.push(["Evidence status", `${evidence.length} evidence item(s), ${answerSources.length} answer-source link(s), ${unresolved.length} open question(s).`]);
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
    function briefCard(title, body) {
      return `<div class="item brief-card"><strong>${escapeHtml(title)}</strong><small>${formatPlainText(body || "")}</small></div>`;
    }
    function formatPlainText(value) {
      return escapeHtml(value).replace(/\n/g, "<br>");
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
    function formatSimpleList(items) {
      return (items || []).slice(0, 12).map(item => `- ${item}`).join("\n") +
        ((items || []).length > 12 ? `\n- ... ${(items || []).length - 12} more` : "");
    }
    function formatEvidenceDelta(items) {
      return (items || []).slice(0, 8).map(item => {
        const source = [item.doc_id, item.chunk_id].filter(Boolean).join(" / ") || "source";
        const support = item.support || item.claim || "";
        return `- ${source}: ${support}`;
      }).join("\n") + ((items || []).length > 8 ? `\n- ... ${(items || []).length - 8} more` : "");
    }
    function renderLiveEvents(events) {
      if (!Array.isArray(events) || !events.length) {
        $("liveEvents").innerHTML = "";
        $("currentStep").innerHTML = "<strong>Idle</strong><small>No run has started.</small>";
        if (currentPlan) renderRunBrief({plan: currentPlan});
        return;
      }
      const latest = events[events.length - 1] || {};
      $("currentStep").innerHTML = renderCurrentStep(latest);
      $("liveEvents").innerHTML = events.slice(-30).map(renderUserEvent).join("");
      renderRunBrief({plan: currentPlan, events});
    }
    function renderCurrentStep(event) {
      const fields = event.fields || {};
      const title = fields.summary || friendlyEventTitle(event);
      const details = fields.next_step ? `Next: ${fields.next_step}` : "Waiting for the next update.";
      return `<strong>${escapeHtml(title)}</strong><small>${escapeHtml(details)}</small>`;
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
      if (fields.chunks !== undefined) lines.push(`Searchable chunks: ${fields.chunks}`);
      if (fields.chunk_count !== undefined) lines.push(`Chunks from this document: ${fields.chunk_count}`);
      if (fields.text_chars !== undefined) lines.push(`Extracted characters: ${fields.text_chars}`);
      if (fields.load_error) lines.push(`Load issue: ${fields.load_error}`);
      if (fields.user_nudge) lines.push(`Your instruction: ${fields.user_nudge}`);
      if (fields.source_selection_mode) lines.push(`Source plan: ${formatSourceSelectionMode(fields.source_selection_mode)}`);
      if (fields.reason) lines.push(`Why: ${fields.reason}`);
      if (fields.search_queries) lines.push(`Search targets: ${fields.search_queries.join("; ")}`);
      if (fields.queries) lines.push(`Search targets: ${fields.queries.join("; ")}`);
      if (fields.selected_documents) lines.push(`Reading first: ${fields.selected_documents.join(", ")}`);
      if (fields.skipped_document_count) lines.push(`Held back for now: ${fields.skipped_document_count} document(s).`);
      if (fields.selected_sources) {
        lines.push("Sources selected:");
        for (const source of fields.selected_sources) {
          lines.push(`- ${source.document || source.chunk_id}: ${source.preview || ""}`);
        }
      }
      if (fields.evidence_preview) {
        lines.push("Evidence found:");
        for (const item of fields.evidence_preview) {
          lines.push(`- ${item.support || item.claim || ""}`);
        }
      }
      if (fields.analysis_preview) lines.push(`Worker notes: ${fields.analysis_preview}`);
      if (fields.answer_preview) lines.push(`Draft preview: ${fields.answer_preview}`);
      if (fields.sample_documents) lines.push(`Examples: ${fields.sample_documents.join(", ")}`);
      if (fields.omitted_document_count) lines.push(`Plus ${fields.omitted_document_count} more.`);
      if (fields.steer_hint) lines.push(`Steer: ${fields.steer_hint}`);
      if (fields.next_step) lines.push(`Next: ${fields.next_step}`);
      if (!lines.length) {
        for (const [key, value] of Object.entries(fields)) {
          if (key === "summary") continue;
          lines.push(`${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`);
        }
      }
      return lines.join("\n");
    }
    function formatSourceSelectionMode(mode) {
      if (mode === "replan_from_nudge") return "re-planning from your steering note";
      if (mode === "locked_by_user") return "using the first-read documents you locked";
      return String(mode || "");
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
