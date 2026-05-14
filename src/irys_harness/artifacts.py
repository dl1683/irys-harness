from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document
from openpyxl import Workbook

ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
XLSX_CELL_CHAR_LIMIT = 500


def render_deliverables(
    *,
    output_dir: str | Path,
    deliverables: list[str],
    title: str,
    packet: dict[str, Any],
) -> list[dict[str, Any]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for filename in deliverables:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".xlsx":
            render_xlsx(path, title=title, deliverable=filename, packet=packet)
        else:
            render_docx(path, title=title, deliverable=filename, packet=packet)
        artifacts.append(
            {
                "filename": filename,
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return artifacts


def render_docx(path: Path, *, title: str, deliverable: str, packet: dict[str, Any]) -> None:
    doc = Document()
    doc.add_heading(safe_text(title), level=1)
    doc.add_paragraph(safe_text(f"Deliverable: {deliverable}"))
    deliverable_plan = find_deliverable_plan(packet, deliverable)
    if deliverable_plan:
        doc.add_heading("Deliverable Contract", level=2)
        for section in deliverable_plan.get("required_sections", [])[:12]:
            doc.add_paragraph(safe_text(section), style=None)
    draft_answer = packet.get("draft_answer")
    if draft_answer:
        doc.add_heading("Draft Work Product", level=2)
        if is_probably_encoded_artifact(str(draft_answer)):
            draft_answer = packet.get("plain_text_fallback") or packet.get("cheap_worker_summary") or ""
        for block in str(draft_answer).split("\n\n"):
            if block.strip():
                doc.add_paragraph(safe_text(block.strip()))
    appendix = packet.get("artifact_appendix")
    if appendix:
        doc.add_heading("Structured Findings Appendix", level=2)
        for block in str(appendix).split("\n\n"):
            if block.strip():
                doc.add_paragraph(safe_text(block.strip()))
    doc.add_heading("Candidate Evidence Packet", level=2)
    for item in packet.get("verified_evidence", []):
        claim = item.get("claim", "")
        support = item.get("raw_support", "")
        source = item.get("source", {})
        doc.add_paragraph(safe_text(f"Claim: {claim}"))
        doc.add_paragraph(safe_text(f"Source: {source.get('doc_id')} / {source.get('chunk_id')}"))
        doc.add_paragraph(safe_text(support, limit=1200))
    unresolved = packet.get("unresolved", [])
    if unresolved:
        doc.add_heading("Unresolved", level=2)
        for item in unresolved:
            doc.add_paragraph(safe_text(item))
    doc.save(path)


def render_xlsx(path: Path, *, title: str, deliverable: str, packet: dict[str, Any]) -> None:
    workbook = Workbook()
    deliverable_plan = find_deliverable_plan(packet, deliverable)
    sheet_plans = normalize_sheet_plans(deliverable_plan)
    if "Evidence" not in {plan["name"] for plan in sheet_plans}:
        sheet_plans.append({"name": "Evidence", "purpose": "Source evidence and trace support"})

    text = "\n\n".join(
        str(packet.get(key, ""))
        for key in ["draft_answer", "cheap_worker_summary"]
        if packet.get(key)
    )
    tables = extract_markdown_tables(text)

    for index, sheet_plan in enumerate(sheet_plans):
        sheet = workbook.active if index == 0 else workbook.create_sheet()
        sheet.title = safe_sheet_title(sheet_plan["name"], used=workbook.sheetnames)
        render_planned_sheet(
            sheet,
            title=title,
            deliverable=deliverable,
            sheet_plan=sheet_plan,
            packet=packet,
            tables=tables,
            source_text=text,
        )
    workbook.save(path)


def find_deliverable_plan(packet: dict[str, Any], deliverable: str) -> dict[str, Any] | None:
    contract = packet.get("deliverable_contract") or {}
    for item in contract.get("deliverables", []) or []:
        if str(item.get("filename", "")).lower() == deliverable.lower():
            return item
    return None


def normalize_sheet_plans(deliverable_plan: dict[str, Any] | None) -> list[dict[str, str]]:
    raw_plans = []
    if deliverable_plan:
        raw_plans = deliverable_plan.get("workbook_sheets", []) or []
    if not raw_plans:
        raw_plans = [
            {"name": "Issue Matrix", "purpose": "Task-specific issues, calculations, or comparisons"},
            {"name": "Source Evidence", "purpose": "Evidence supporting the workbook"},
        ]
    plans: list[dict[str, str]] = []
    for item in raw_plans:
        if isinstance(item, str):
            plans.append({"name": item, "purpose": ""})
        else:
            plans.append(
                {
                    "name": str(item.get("name") or "Sheet"),
                    "purpose": str(item.get("purpose") or ""),
                }
            )
    return plans


def render_planned_sheet(
    sheet: Any,
    *,
    title: str,
    deliverable: str,
    sheet_plan: dict[str, str],
    packet: dict[str, Any],
    tables: list[dict[str, Any]],
    source_text: str,
) -> None:
    sheet.append(["Task", safe_cell(title)])
    sheet.append(["Deliverable", safe_cell(deliverable)])
    sheet.append(["Sheet Purpose", safe_cell(sheet_plan.get("purpose", ""))])
    sheet.append([])
    if sheet_plan["name"].lower() == "evidence":
        append_evidence_rows(sheet, packet)
        return

    matched_tables = match_tables_to_sheet(tables, sheet_plan["name"])
    if matched_tables:
        for table in matched_tables[:3]:
            append_markdown_table(sheet, table)
            sheet.append([])
    else:
        append_issue_template(sheet, sheet_plan["name"])
    append_relevant_text(sheet, source_text, sheet_plan["name"])
    append_evidence_rows(sheet, packet, max_rows=10)


def append_issue_template(sheet: Any, sheet_name: str) -> None:
    lower = sheet_name.lower()
    if any(term in lower for term in ["calculation", "model", "limitation", "shift", "utilization"]):
        sheet.append(["Item", "Source Input", "Formula / Standard", "Result", "Conclusion", "Source"])
    elif any(term in lower for term in ["register", "inventory", "matrix", "comparison", "log"]):
        sheet.append(["Item", "Document / Party", "Requirement", "Finding", "Risk", "Source"])
    else:
        sheet.append(["Issue", "Finding", "Impact", "Remediation", "Source"])


def append_evidence_rows(sheet: Any, packet: dict[str, Any], *, max_rows: int | None = None) -> None:
    sheet.append(["Claim", "Doc ID", "Chunk ID", "Support"])
    evidence = packet.get("verified_evidence", [])
    if max_rows is not None:
        evidence = evidence[:max_rows]
    for item in evidence:
        source = item.get("source", {})
        sheet.append(
            [
                safe_cell(item.get("claim", "")),
                safe_cell(source.get("doc_id", "")),
                safe_cell(source.get("chunk_id", "")),
                safe_cell(item.get("raw_support", "")),
            ]
        )


def extract_markdown_tables(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    tables: list[dict[str, Any]] = []
    index = 0
    heading = ""
    while index < len(lines) - 1:
        heading_match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", lines[index])
        if heading_match:
            heading = heading_match.group(1).strip()
            index += 1
            continue
        if "|" not in lines[index] or not re.search(r"\|\s*:?-{2,}:?\s*(\||$)", lines[index + 1]):
            index += 1
            continue
        header = split_markdown_row(lines[index])
        rows: list[list[str]] = []
        index += 2
        while index < len(lines) and "|" in lines[index]:
            row = split_markdown_row(lines[index])
            if row:
                rows.append(row)
            index += 1
        if header and rows:
            tables.append({"heading": heading, "headers": header, "rows": rows})
    return tables


def split_markdown_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [safe_cell(cell.strip()) for cell in stripped.split("|")]


def match_tables_to_sheet(tables: list[dict[str, Any]], sheet_name: str) -> list[dict[str, Any]]:
    sheet_tokens = keyword_tokens(sheet_name)
    scored: list[tuple[int, dict[str, Any]]] = []
    for table in tables:
        table_text = str(table.get("heading", "")) + " " + " ".join(table.get("headers", []))
        table_text += " " + " ".join(" ".join(row) for row in table.get("rows", [])[:4])
        score = len(sheet_tokens.intersection(keyword_tokens(table_text)))
        if score:
            scored.append((score, table))
    if scored:
        return [table for _, table in sorted(scored, key=lambda item: item[0], reverse=True)]
    return tables[:1]


def append_markdown_table(sheet: Any, table: dict[str, Any]) -> None:
    headers = [safe_cell(item) for item in table.get("headers", [])]
    if headers:
        sheet.append(headers)
    for row in table.get("rows", [])[:80]:
        sheet.append([safe_cell(item) for item in row])


def append_relevant_text(sheet: Any, text: str, sheet_name: str) -> None:
    snippets = relevant_snippets(text, sheet_name)
    if not snippets:
        return
    sheet.append([])
    sheet.append(["Relevant Worker Text"])
    for snippet in snippets[:8]:
        sheet.append([safe_cell(snippet)])


def relevant_snippets(text: str, sheet_name: str) -> list[str]:
    tokens = keyword_tokens(sheet_name)
    sections = re.split(r"(?m)^#{1,4}\s+", text)
    matches: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if tokens.intersection(keyword_tokens(section[:800])):
            matches.append(section[:5000])
    return matches


def is_probably_encoded_artifact(text: str) -> bool:
    stripped = str(text or "").strip()
    lower = stripped[:4000].lower()
    if any(marker in lower for marker in ["<base64_file", "<content>", "<?xml", "pk\x03\x04"]):
        return True
    if re.match(r"^```(?:xml|base64|html)?\s*<", stripped, flags=re.IGNORECASE | re.DOTALL):
        return True
    base64ish = re.sub(r"\s+", "", stripped[:12000])
    return len(base64ish) > 4000 and bool(re.fullmatch(r"[A-Za-z0-9+/=]+", base64ish))


def keyword_tokens(text: str) -> set[str]:
    stop = {"and", "the", "for", "with", "from", "source", "evidence", "sheet"}
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in stop
    }


def safe_sheet_title(title: str, *, used: list[str]) -> str:
    cleaned = re.sub(r"[\[\]:*?/\\]", " ", title).strip() or "Sheet"
    cleaned = re.sub(r"\s+", " ", cleaned)[:31].strip() or "Sheet"
    candidate = cleaned
    suffix = 2
    while candidate in used:
        suffix_text = f" {suffix}"
        candidate = f"{cleaned[: 31 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    return candidate


def safe_text(value: Any, *, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = ILLEGAL_XML_RE.sub("", text)
    if limit is not None:
        return text[:limit]
    return text


def safe_cell(value: Any, *, limit: int = XLSX_CELL_CHAR_LIMIT) -> str:
    return safe_text(value, limit=limit)
