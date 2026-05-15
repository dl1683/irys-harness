from __future__ import annotations

import base64
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import HarnessConfig
from .product import compare_product_traces, run_product_matter, sanitize_matter_id
from .trace import load_trace, trace_summary


MAX_UPLOAD_BYTES = 25_000_000


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
            self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/run", "/api/rerun"}:
                self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self.read_json()
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
                upload_paths = save_uploaded_files(
                    payload.get("uploads", []),
                    upload_root=Path(output_dir) / "_uploads" / sanitize_matter_id(matter_id),
                )
                paths.extend(str(path) for path in upload_paths)
                result = run_product_matter(
                    objective=str(payload.get("objective", "")),
                    paths=paths,
                    matter_id=matter_id,
                    config=config,
                    trace_dir=trace_dir,
                    output_dir=output_dir,
                    live_synthesis=bool(payload.get("live_synthesis", False)),
                    top_k=int(payload.get("top_k", 12) or 12),
                    max_files=int(payload.get("max_files", 80) or 80),
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


def rerun_from_trace(
    payload: dict[str, Any],
    *,
    config: HarnessConfig,
    trace_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    parent_path = resolve_trace_path(str(payload.get("trace_path") or ""), trace_dir)
    parent = load_trace(parent_path)
    task = parent.get("task") or {}
    original_objective = str(task.get("question") or "")
    nudge = str(payload.get("nudge") or "").strip()
    if not nudge:
        raise ValueError("nudge is required")
    paths = [str(item) for item in task.get("context_files", []) if str(item).strip()]
    if not paths:
        raise ValueError("parent trace does not contain context_files")
    base_matter_id = str(task.get("task_id") or "matter")
    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    matter_id = sanitize_matter_id(f"{base_matter_id}-nudge-{suffix}")
    paths.extend(parse_paths(payload.get("paths", [])))
    upload_paths = save_uploaded_files(
        payload.get("uploads", []),
        upload_root=Path(output_dir) / "_uploads" / matter_id,
    )
    paths.extend(str(path) for path in upload_paths)
    objective = f"{original_objective}\n\nUser steering note: {nudge}"
    result = run_product_matter(
        objective=objective,
        paths=paths,
        matter_id=matter_id,
        config=config,
        trace_dir=trace_dir,
        output_dir=output_dir,
        live_synthesis=bool(payload.get("live_synthesis", False)),
        top_k=int(payload.get("top_k", 12) or 12),
        max_files=int(payload.get("max_files", 80) or 80),
        verbose=False,
        parent_trace_path=str(parent_path),
        user_nudge=nudge,
    )
    child_trace = result.state.to_trace()
    response = result.to_dict()
    response["summary"] = trace_summary(child_trace)
    response["trace"] = child_trace
    response["comparison"] = compare_product_traces(parent, child_trace)
    return response


def parse_paths(raw_paths: Any) -> list[str]:
    if isinstance(raw_paths, str):
        return [line.strip() for line in raw_paths.splitlines() if line.strip()]
    return [str(item).strip() for item in raw_paths or [] if str(item).strip()]


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


def resolve_trace_path(raw_path: str, trace_dir: str | Path) -> Path:
    root = Path(trace_dir).resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("trace path must be under the configured trace directory")
    return resolved


def save_uploaded_files(raw_uploads: Any, *, upload_root: str | Path) -> list[Path]:
    if not raw_uploads:
        return []
    if not isinstance(raw_uploads, list):
        raise ValueError("uploads must be a list")
    root = Path(upload_root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, item in enumerate(raw_uploads, 1):
        if not isinstance(item, dict):
            raise ValueError("each upload must be an object")
        filename = safe_upload_filename(str(item.get("filename") or f"upload-{index}.txt"))
        encoded = str(item.get("content_base64") or "")
        if not encoded:
            raise ValueError(f"upload {filename} is missing content_base64")
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception as exc:  # noqa: BLE001 - normalize base64 parser errors.
            raise ValueError(f"upload {filename} is not valid base64") from exc
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload {filename} exceeds {MAX_UPLOAD_BYTES} bytes")
        target = unique_upload_path(root, filename)
        target.write_bytes(data)
        paths.append(target)
    return paths


def safe_upload_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "upload.txt"
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "-", name).strip(" .")
    return cleaned[:120] or "upload.txt"


def unique_upload_path(root: Path, filename: str) -> Path:
    target = root / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for index in range(2, 1000):
        candidate = root / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError(f"too many uploads named {filename}")


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
      <label for="matter">Matter ID</label>
      <input id="matter" value="local-matter" />
      <label for="paths">Corpus Paths</label>
      <textarea id="paths" spellcheck="false"></textarea>
      <label for="files">Corpus Files</label>
      <input id="files" type="file" multiple />
      <div class="row">
        <label class="toggle"><input id="live" type="checkbox" /> Live synthesis</label>
        <input id="topk" type="number" min="1" max="50" value="12" />
      </div>
      <div class="row">
        <button id="run">Run</button>
        <button class="secondary" id="clear">Clear</button>
      </div>
    </section>
    <section>
      <h2>Objective</h2>
      <textarea id="objective" class="objective"></textarea>
      <div class="metric-grid">
        <div class="metric"><b>Documents</b><span id="docCount">0</span></div>
        <div class="metric"><b>Chunks</b><span id="chunkCount">0</span></div>
        <div class="metric"><b>Tokens</b><span id="tokens">0</span></div>
        <div class="metric"><b>Cost</b><span id="cost">$0.00</span></div>
      </div>
      <h2>Answer</h2>
      <div class="answer"><pre id="answer"></pre></div>
    </section>
    <section>
      <h2>Trace</h2>
      <label for="tracepath">Trace Path</label>
      <input id="tracepath" />
      <div class="row">
        <button class="secondary" id="loadTrace">Load Trace</button>
      </div>
      <label for="nudge">Nudge</label>
      <textarea id="nudge"></textarea>
      <label for="rerunPaths">Additional Corpus Paths</label>
      <textarea id="rerunPaths"></textarea>
      <label for="rerunFiles">Additional Corpus Files</label>
      <input id="rerunFiles" type="file" multiple />
      <div class="row">
        <button class="secondary" id="rerunTrace">Rerun</button>
      </div>
      <h2 style="margin-top:16px">Diagnosis</h2>
      <div class="list" id="diagnosis"></div>
      <h2 style="margin-top:16px">Comparison</h2>
      <div class="list" id="comparison"></div>
      <h2 style="margin-top:16px">Events</h2>
      <div class="list" id="events"></div>
      <h2 style="margin-top:16px">Documents</h2>
      <div class="list" id="documents"></div>
      <h2 style="margin-top:16px">Evidence</h2>
      <div class="list" id="evidence"></div>
      <h2 style="margin-top:16px">Answer Sources</h2>
      <div class="list" id="answerSources"></div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const status = $("status");
    const run = $("run");
    const loadTrace = $("loadTrace");
    const rerunTrace = $("rerunTrace");
    $("clear").addEventListener("click", () => {
      $("objective").value = "";
      $("files").value = "";
      $("answer").textContent = "";
      $("tracepath").value = "";
      $("nudge").value = "";
      $("rerunPaths").value = "";
      $("rerunFiles").value = "";
      $("diagnosis").innerHTML = "";
      $("comparison").innerHTML = "";
      $("events").innerHTML = "";
      $("documents").innerHTML = "";
      $("evidence").innerHTML = "";
      $("answerSources").innerHTML = "";
      status.textContent = "";
    });
    run.addEventListener("click", async () => {
      run.disabled = true;
      status.textContent = "Running";
      try {
        const uploads = await readUploads($("files").files);
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            matter_id: $("matter").value,
            paths: $("paths").value,
            uploads,
            objective: $("objective").value,
            live_synthesis: $("live").checked,
            top_k: Number($("topk").value || 12)
          })
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Run failed");
        render(data);
        status.textContent = data.trace_path;
        $("tracepath").value = data.trace_path || "";
      } catch (error) {
        status.textContent = error.message;
      } finally {
        run.disabled = false;
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
        status.textContent = $("tracepath").value;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        loadTrace.disabled = false;
      }
    });
    rerunTrace.addEventListener("click", async () => {
      rerunTrace.disabled = true;
      status.textContent = "Rerunning";
      try {
        const uploads = await readUploads($("rerunFiles").files);
        const response = await fetch("/api/rerun", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            trace_path: $("tracepath").value,
            nudge: $("nudge").value,
            paths: $("rerunPaths").value,
            uploads,
            live_synthesis: $("live").checked,
            top_k: Number($("topk").value || 12)
          })
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Rerun failed");
        render(data);
        status.textContent = data.trace_path;
        $("tracepath").value = data.trace_path || "";
      } catch (error) {
        status.textContent = error.message;
      } finally {
        rerunTrace.disabled = false;
      }
    });
    async function readUploads(fileList) {
      const files = Array.from(fileList || []);
      const uploads = [];
      for (const file of files) {
        const dataUrl = await readAsDataUrl(file);
        const comma = dataUrl.indexOf(",");
        uploads.push({
          filename: file.name,
          content_base64: comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl
        });
      }
      return uploads;
    }
    function readAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(reader.error || new Error("File read failed"));
        reader.readAsDataURL(file);
      });
    }
    function render(data) {
      const trace = data.trace || {};
      const metrics = trace.metrics || {};
      $("docCount").textContent = String((trace.documents || []).length);
      $("chunkCount").textContent = String((trace.chunks || []).length);
      $("tokens").textContent = String(metrics.total_tokens || 0);
      $("cost").textContent = "$" + Number(metrics.estimated_cost || 0).toFixed(4);
      $("matter").value = (trace.task || {}).task_id || $("matter").value;
      $("objective").value = (trace.task || {}).question || $("objective").value;
      $("answer").textContent = data.rendered_answer || trace.rendered_answer || "";
      $("diagnosis").innerHTML = Object.entries(trace.diagnosis || {}).map(([key, value]) => card(
        key,
        typeof value === "string" || typeof value === "number" ? String(value) : JSON.stringify(value, null, 2)
      )).join("");
      $("comparison").innerHTML = renderComparison(data.comparison);
      $("events").innerHTML = (trace.events || []).map(event => card(
        event.label + " - " + event.message,
        JSON.stringify(event.fields || {}, null, 2)
      )).join("");
      $("documents").innerHTML = (trace.documents || []).map(doc => card(
        (doc.doc_id || "") + " - " + (doc.filename || ""),
        (doc.path || "") + "\nchars=" + (doc.text_chars || 0) + (doc.load_error ? "\nerror=" + doc.load_error : "")
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
    function card(title, body) {
      return `<div class="item"><strong>${escapeHtml(title)}</strong><small><pre>${escapeHtml(body || "")}</pre></small></div>`;
    }
    function renderComparison(comparison) {
      if (!comparison) return "";
      const evidence = comparison.evidence_delta || {};
      const unresolved = comparison.unresolved_delta || {};
      const metrics = comparison.metrics_delta || {};
      const rows = [
        ["Run", `${comparison.parent_task_id || ""} -> ${comparison.child_task_id || ""}`],
        ["Answer changed", String(Boolean(comparison.answer_changed))],
        ["Evidence", `+${evidence.added_count || 0} / -${evidence.removed_count || 0} / kept ${evidence.kept_count || 0}`],
        ["Unresolved", `+${(unresolved.added || []).length} / -${(unresolved.removed || []).length} / kept ${unresolved.kept_count || 0}`],
        ["Tokens delta", String(metrics.total_tokens || 0)],
        ["Cost delta", "$" + Number(metrics.estimated_cost || 0).toFixed(4)]
      ];
      if (comparison.status === "unavailable") rows.push(["Comparison unavailable", comparison.error || "unknown error"]);
      return rows.map(([title, body]) => card(title, body)).join("");
    }
    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }
  </script>
</body>
</html>
"""
