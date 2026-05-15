from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import HarnessConfig
from .product import run_product_matter
from .trace import load_trace, trace_summary


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
                    self.send_json({"summary": trace_summary(trace), "trace": trace})
                except Exception as exc:  # noqa: BLE001 - UI endpoint should return structured error.
                    self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            parsed = urlparse(self.path)
            if parsed.path != "/api/run":
                self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self.read_json()
                paths_raw = payload.get("paths", [])
                if isinstance(paths_raw, str):
                    paths = [line.strip() for line in paths_raw.splitlines() if line.strip()]
                else:
                    paths = [str(item) for item in paths_raw if str(item).strip()]
                result = run_product_matter(
                    objective=str(payload.get("objective", "")),
                    paths=paths,
                    matter_id=str(payload.get("matter_id", "matter")),
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
      <div class="list" id="events"></div>
      <h2 style="margin-top:16px">Documents</h2>
      <div class="list" id="documents"></div>
      <h2 style="margin-top:16px">Evidence</h2>
      <div class="list" id="evidence"></div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const status = $("status");
    const run = $("run");
    $("clear").addEventListener("click", () => {
      $("objective").value = "";
      $("answer").textContent = "";
      $("events").innerHTML = "";
      $("documents").innerHTML = "";
      $("evidence").innerHTML = "";
      status.textContent = "";
    });
    run.addEventListener("click", async () => {
      run.disabled = true;
      status.textContent = "Running";
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            matter_id: $("matter").value,
            paths: $("paths").value,
            objective: $("objective").value,
            live_synthesis: $("live").checked,
            top_k: Number($("topk").value || 12)
          })
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Run failed");
        render(data);
        status.textContent = data.trace_path;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        run.disabled = false;
      }
    });
    function render(data) {
      const trace = data.trace || {};
      const metrics = trace.metrics || {};
      $("docCount").textContent = String((trace.documents || []).length);
      $("chunkCount").textContent = String((trace.chunks || []).length);
      $("tokens").textContent = String(metrics.total_tokens || 0);
      $("cost").textContent = "$" + Number(metrics.estimated_cost || 0).toFixed(4);
      $("answer").textContent = data.rendered_answer || "";
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
    }
    function card(title, body) {
      return `<div class="item"><strong>${escapeHtml(title)}</strong><small><pre>${escapeHtml(body || "")}</pre></small></div>`;
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
