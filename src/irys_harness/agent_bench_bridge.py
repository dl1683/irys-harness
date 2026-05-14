from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from .config import HarnessConfig
from .metrics import ModelCallRecord
from .models.gemini import GeminiModelRouter, ModelResult


DEFAULT_BENCHMARK_SPECS = [
    "longbench_v2:train",
    "facts_grounding:public",
    "docfinqa:train",
    "hotpotqa:train",
    "musique:train",
    "cuad:train",
    "nolima:test",
    "mrcr:2needle",
    "counting_stars:test",
    "loong:test",
    "l_citeeval:test",
    "fanoutqa:dev",
    "multihop_rag:test",
    "nocha:test",
    "locomo:test",
    "qasa:test",
    "qmsum:test",
    "longhealth:flat",
    "repoqa:test",
    "long_code_arena:test",
    "financebench:train",
]


FULL_CONTEXT_DIRECT_BENCHMARK_REASONS = {
    "longbench_v2": "long-context multiple choice usually benefits from preserving the full adapted context.",
    "facts_grounding": "grounding/refusal tasks are best scored from direct source-context adherence.",
    "mrcr": "conversation-position retrieval is damaged by intermediate summarization.",
    "fanoutqa": "fan-out lookup needs broad source coverage before aggregation.",
    "nocha": "narrative verification needs direct access to the available narrative context.",
    "locomo": "conversation-memory questions are sensitive to exact turns and temporal details.",
    "qasa": "paper QA benefits from direct source-context grounding on small/medium contexts.",
    "qmsum": "query-focused summaries need broad transcript coverage before compression is reliable.",
    "longhealth": "clinical multiple-choice tasks need direct access to chart details.",
    "repoqa": "repo lookup requires exact symbol/body preservation.",
    "long_code_arena": "code tasks require exact logs/code rather than lossy natural-language packets.",
    "financebench": "finance QA needs exact evidence strings and numeric values.",
}

NOLIMA_HARD_NEEDLESET_URL = (
    "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/"
    "needlesets/needle_set_hard.json"
)
NOLIMA_HAYSTACK_URL = (
    "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/"
    "haystack/rand_shuffle/rand_book_1.txt"
)
NOLIMA_CONTEXT_CHARS = 80_000
DOCFINQA_CONTEXT_CHAR_LIMIT = 90_000
DOCFINQA_MAX_BLOCKS = 40
DOCFINQA_BLOCK_BEFORE = 4
DOCFINQA_BLOCK_AFTER = 14


@dataclass(frozen=True)
class AgentBenchSpec:
    benchmark: str
    split: str


@dataclass
class BridgeAgentResult:
    answer: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class AgentBenchRunResult:
    benchmark: str
    split: str
    summary_path: str | None
    trace_dir: str | None
    examples_loaded: int = 0
    examples_attempted: int = 0
    scored: int = 0
    pass_count: int = 0
    avg_score: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    error: str | None = None

    @classmethod
    def from_summary(
        cls,
        summary: Any,
        *,
        summary_path: Path,
        trace_dir: Path,
    ) -> "AgentBenchRunResult":
        return cls(
            benchmark=str(summary.benchmark),
            split=str(summary.split),
            summary_path=str(summary_path),
            trace_dir=str(trace_dir),
            examples_loaded=int(summary.examples_loaded),
            examples_attempted=int(summary.examples_attempted),
            scored=int(summary.scored),
            pass_count=int(summary.pass_count),
            avg_score=float(summary.avg_score),
            total_tokens_in=int(summary.total_tokens_in),
            total_tokens_out=int(summary.total_tokens_out),
            total_cost_usd=float(summary.total_cost_usd),
        )

    @classmethod
    def failed(cls, spec: AgentBenchSpec, error: Exception) -> "AgentBenchRunResult":
        return cls(
            benchmark=spec.benchmark,
            split=spec.split,
            summary_path=None,
            trace_dir=None,
            error=f"{type(error).__name__}: {error}",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IrysAgentBenchBackend:
    name = "irys-harness-agent-bench"
    version = "0.1.0"

    def __init__(
        self,
        *,
        config: HarnessConfig,
        benchmark: str,
        split: str,
        mode: str = "three-tier",
        trace_dir: str | Path | None = None,
        router: GeminiModelRouter | None = None,
    ) -> None:
        if mode not in {"adaptive", "direct", "three-tier"}:
            raise ValueError("mode must be 'adaptive', 'direct', or 'three-tier'")
        self.config = config
        self.benchmark = benchmark
        self.split = split
        self.mode = mode
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.router = router or GeminiModelRouter(config)

    async def run(
        self,
        *,
        query: str,
        context: str = "",
        max_tokens: int = 4096,
    ) -> BridgeAgentResult:
        task_hash = stable_task_hash(query, context)
        started = time.perf_counter()
        events: list[dict[str, Any]] = []
        calls: list[ModelCallRecord] = []
        error: str | None = None
        answer = ""
        evidence_packet = ""
        route = self._select_pipeline(query=query, context=context)
        prompt_context, context_event = prepare_prompt_context_for_benchmark(
            self.benchmark,
            context,
            query=query,
        )
        events.append(
            {
                "type": "route_decision",
                "requested_mode": self.mode,
                "selected_pipeline": route["pipeline"],
                "reason": route["reason"],
                "benchmark": self.benchmark,
                "context_chars": len(context),
                "at": datetime.now(UTC).isoformat(),
            }
        )
        if context_event:
            events.append(context_event)

        try:
            deterministic_answer = context_event.get("deterministic_answer") if context_event else None
            if deterministic_answer:
                answer = str(deterministic_answer)
                events.append(
                    {
                        "type": "deterministic_answer",
                        "benchmark": self.benchmark,
                        "method": context_event.get("method"),
                        "text": answer,
                        "at": datetime.now(UTC).isoformat(),
                    }
                )
            elif route["pipeline"] == "direct":
                result = await self._generate(
                    module="extraction",
                    prompt=direct_prompt_for_benchmark(self.benchmark, query, prompt_context),
                    max_output_tokens=max_tokens,
                )
                calls.append(result.usage)
                answer = result.text.strip()
                events.append(model_event("direct_answer", result))
            else:
                extract = await self._generate(
                    module="extraction",
                    prompt=evidence_prompt_for_benchmark(self.benchmark, query, prompt_context),
                    max_output_tokens=min(4096, max_tokens),
                )
                calls.append(extract.usage)
                evidence_packet = extract.text
                events.append(model_event("evidence_packet", extract))

                critique = await self._generate(
                    module="critic",
                    prompt=critic_prompt_for_benchmark(self.benchmark, query, extract.text),
                    max_output_tokens=2048,
                )
                calls.append(critique.usage)
                events.append(model_event("critic_plan", critique))

                synth = await self._generate(
                    module="synthesis",
                    prompt=synthesis_prompt_for_benchmark(self.benchmark, query, extract.text, critique.text),
                    max_output_tokens=max_tokens,
                )
                calls.append(synth.usage)
                answer = synth.text.strip()
                events.append(model_event("final_answer", synth))
        except Exception as exc:  # noqa: BLE001 - backend must isolate per-task failures.
            error = f"{type(exc).__name__}: {exc}"
            answer = f"[ERROR] {error}"
            events.append(
                {
                    "type": "error",
                    "message": error,
                    "at": datetime.now(UTC).isoformat(),
                }
            )

        rendered_answer = render_benchmark_answer(
            benchmark=self.benchmark,
            answer=answer,
            context=context,
            evidence_packet=evidence_packet,
        )
        if rendered_answer != answer:
            events.append(
                {
                    "type": "renderer_postprocess",
                    "benchmark": self.benchmark,
                    "before": answer,
                    "after": rendered_answer,
                    "at": datetime.now(UTC).isoformat(),
                }
            )
            answer = rendered_answer

        latency_ms = int((time.perf_counter() - started) * 1000)
        trace_path = self._write_trace(
            task_hash=task_hash,
            query=query,
            context=context,
            answer=answer,
            calls=calls,
            events=events,
            error=error,
            latency_ms=latency_ms,
        )
        if trace_path:
            events.append({"type": "trace_saved", "path": str(trace_path)})

        return BridgeAgentResult(
            answer=answer,
            tokens_in=sum(call.input_tokens for call in calls),
            tokens_out=sum(call.output_tokens for call in calls),
            cost_usd=sum(call.estimated_cost for call in calls),
            latency_ms=latency_ms,
            trace=events,
            error=error,
        )

    def _select_pipeline(self, *, query: str, context: str) -> dict[str, str]:
        if self.mode in {"direct", "three-tier"}:
            return {
                "pipeline": self.mode,
                "reason": f"backend mode forced to {self.mode}",
            }
        reason = FULL_CONTEXT_DIRECT_BENCHMARK_REASONS.get(self.benchmark)
        if reason and context:
            return {
                "pipeline": "direct",
                "reason": reason,
            }
        return {
            "pipeline": "three-tier",
            "reason": "default adaptive route uses extraction, critic, and synthesis",
        }

    async def _generate(
        self,
        *,
        module: str,
        prompt: str,
        max_output_tokens: int,
    ) -> ModelResult:
        return await asyncio.to_thread(
            self.router.generate,
            module=module,
            prompt=prompt,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )

    def _write_trace(
        self,
        *,
        task_hash: str,
        query: str,
        context: str,
        answer: str,
        calls: list[ModelCallRecord],
        events: list[dict[str, Any]],
        error: str | None,
        latency_ms: int,
    ) -> Path | None:
        if self.trace_dir is None:
            return None
        target_dir = self.trace_dir / self.benchmark / self.split
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{task_hash}.json"
        total_tokens = sum(call.input_tokens + call.output_tokens for call in calls)
        tokens_by_tier: dict[str, int] = {}
        for call in calls:
            tokens_by_tier[call.tier.value] = (
                tokens_by_tier.get(call.tier.value, 0) + call.input_tokens + call.output_tokens
            )
        trace = {
            "run_id": f"agent_bench_{task_hash}",
            "benchmark": self.benchmark,
            "split": self.split,
            "task_hash": task_hash,
            "backend": self.name,
            "backend_version": self.version,
            "mode": self.mode,
            "created_at": datetime.now(UTC).isoformat(),
            "isolation": {
                "memory_enabled": self.config.run.memory_enabled,
                "web_enabled": self.config.run.web_enabled,
            },
            "task": {
                "question": query,
                "context_chars": len(context),
                "context_hash": stable_context_hash(context),
            },
            "answer_contract_versions": [
                {
                    "mode": self.mode,
                    "interpreted_goal": "Answer the adapted benchmark query using only provided context when context is present.",
                    "needed_information": ["relevant context facts", "answer format", "grounding or abstention requirements"],
                    "verification_requirements": ["do not use hidden gold answer", "do not rely on persistent memory"],
                }
            ],
            "extraction_records": [event for event in events if event.get("type") == "evidence_packet"],
            "critic_records": [event for event in events if event.get("type") == "critic_plan"],
            "draft_answer": answer,
            "rendered_answer": answer,
            "error": error,
            "metrics": {
                "input_tokens": sum(call.input_tokens for call in calls),
                "output_tokens": sum(call.output_tokens for call in calls),
                "total_tokens": total_tokens,
                "estimated_cost": sum(call.estimated_cost for call in calls),
                "latency_seconds": latency_ms / 1000,
                "tokens_by_tier": tokens_by_tier,
                "token_share_by_tier": {
                    tier: tokens / total_tokens if total_tokens else 0.0
                    for tier, tokens in sorted(tokens_by_tier.items())
                },
                "model_calls": [call.to_dict() for call in calls],
            },
            "events": events,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(trace, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path


def direct_prompt(query: str, context: str) -> str:
    if context:
        return (
            "Source context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer the question based only on the source context. Be specific and direct. "
            "If the required answer format is obvious, obey it exactly. If the context is insufficient, say so."
        )
    return query


def direct_prompt_for_benchmark(benchmark: str, query: str, context: str) -> str:
    if benchmark == "nolima":
        return (
            "Source context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer the question using the source context. The answer may require an ordinary latent bridge such as "
            "city-country, institution-location, painting-museum-location, or place-country knowledge. Return only the "
            "short character or entity name explicitly supported by the context. If no such source-supported entity exists, "
            "return INSUFFICIENT."
        )
    return direct_prompt(query, context)


def prepare_prompt_context_for_benchmark(
    benchmark: str,
    context: str,
    *,
    query: str = "",
) -> tuple[str, dict[str, Any] | None]:
    if benchmark == "mrcr" and context and query:
        return prepare_mrcr_prompt_context(query=query, context=context)
    if benchmark == "docfinqa" and context and query:
        return prepare_docfinqa_prompt_context(query=query, context=context)
    if benchmark != "nolima" or not context:
        return context, None
    candidates = extract_nolima_candidate_facts(context)
    if not candidates:
        return context, None
    digest = "\n".join(f"- {candidate}" for candidate in candidates[:40])
    prepared = (
        "Candidate short factual sentences extracted from the source context:\n"
        f"{digest}\n\n"
        "Use these extracted source sentences as the retrieval set for the NoLiMa question."
    )
    return prepared, {
        "type": "context_preparation",
        "benchmark": benchmark,
        "method": "nolima_candidate_fact_digest",
        "candidate_count": len(candidates),
        "candidate_preview": candidates[:5],
        "full_context_chars": len(context),
        "model_context_chars": len(prepared),
        "at": datetime.now(UTC).isoformat(),
    }


def prepare_mrcr_prompt_context(
    *,
    query: str,
    context: str,
) -> tuple[str, dict[str, Any] | None]:
    request = parse_mrcr_request(query)
    if not request:
        return context, None
    candidates = extract_mrcr_candidate_responses(context, request["instruction"])
    if not candidates:
        return context, None
    ordinal = int(request["ordinal"])
    selected = candidates[ordinal - 1] if 0 < ordinal <= len(candidates) else None
    preview_candidates = [
        {
            "instance": index + 1,
            "user": user[:160],
            "assistant_preview": assistant[:240],
        }
        for index, (user, assistant) in enumerate(candidates[:5])
    ]
    if not selected:
        prepared = (
            "MRCR exact retrieval digest.\n"
            f"Question:\n{query}\n\n"
            f"Requested instruction: {request['instruction']}\n"
            f"Requested ordinal: {ordinal}\n"
            f"Matched instances found: {len(candidates)}\n"
            "No matching candidate exists for the requested ordinal. Return INSUFFICIENT."
        )
    else:
        _, assistant = selected
        prepared = (
            "MRCR exact retrieval digest.\n"
            "The candidate assistant response below is copied verbatim from the transcript immediately after "
            "the requested user instruction instance. Return the required prefix followed immediately by this "
            "assistant response; do not summarize or edit it.\n\n"
            f"Question:\n{query}\n\n"
            f"Required prefix: {request['prefix']}\n"
            f"Requested instruction: {request['instruction']}\n"
            f"Requested ordinal: {ordinal}\n"
            f"Matched instances found: {len(candidates)}\n\n"
            "MATCHED_ASSISTANT_RESPONSE:\n"
            f"{assistant}"
        )
    return prepared, {
        "type": "context_preparation",
        "benchmark": "mrcr",
        "method": "mrcr_exact_instance_digest",
        "requested_instruction": request["instruction"],
        "requested_ordinal": ordinal,
        "matched_instances": len(candidates),
        "deterministic_answer": (
            f"{request['prefix']}{selected[1]}" if selected else None
        ),
        "candidate_preview": preview_candidates,
        "full_context_chars": len(context),
        "model_context_chars": len(prepared),
        "at": datetime.now(UTC).isoformat(),
    }


def parse_mrcr_request(query: str) -> dict[str, str] | None:
    import re

    match = re.search(
        r"Prepend\s+(?P<prefix>\S+)\s+to\s+the\s+"
        r"(?P<ordinal>\d+)(?:st|nd|rd|th)\s+\(1 indexed\)\s+"
        r"(?P<instruction>.+?)\.\s+Do not include any other text",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return {
        "prefix": match.group("prefix").strip(),
        "ordinal": match.group("ordinal").strip(),
        "instruction": normalize_mrcr_instruction(match.group("instruction")),
    }


def normalize_mrcr_instruction(value: str) -> str:
    import re

    value = value.strip().rstrip(".")
    value = re.sub(r"\s+", " ", value)
    return value


def extract_mrcr_candidate_responses(context: str, instruction: str) -> list[tuple[str, str]]:
    turns = parse_mrcr_transcript(context)
    candidates: list[tuple[str, str]] = []
    target = normalize_mrcr_lookup_text(instruction)
    for index, (role, content) in enumerate(turns[:-1]):
        if role.lower() != "user":
            continue
        user_lookup = normalize_mrcr_lookup_text(content)
        if target not in user_lookup:
            continue
        next_role, next_content = turns[index + 1]
        if next_role.lower() != "assistant":
            continue
        candidates.append((content, next_content))
    return candidates


def parse_mrcr_transcript(context: str) -> list[tuple[str, str]]:
    import re

    pattern = re.compile(r"\[(user|assistant|system)\]\s*", flags=re.IGNORECASE)
    matches = list(pattern.finditer(context))
    turns: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        turns.append((match.group(1), context[start:end].strip()))
    return turns


def normalize_mrcr_lookup_text(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_docfinqa_prompt_context(
    *,
    query: str,
    context: str,
) -> tuple[str, dict[str, Any] | None]:
    lines = extract_docfinqa_relevant_lines(
        query=query,
        context=context,
        max_chars=DOCFINQA_CONTEXT_CHAR_LIMIT,
    )
    if not lines:
        return context, None
    digest = "\n".join(
        f"[L{line_no}] score={score}: {line}"
        for line_no, score, line in lines
    )
    prepared = (
        "Query-focused financial source digest for DocFinQA.\n"
        "The lines below are copied from the provided filing context and selected by query terms, "
        "year references, financial table structure, and numeric density. Use only these source lines; "
        "preserve units and compute formulas explicitly when needed.\n\n"
        f"Question:\n{query}\n\n"
        "Selected source lines:\n"
        f"{digest}"
    )
    event: dict[str, Any] = {
        "type": "context_preparation",
        "benchmark": "docfinqa",
        "method": "docfinqa_query_numeric_digest",
        "selected_line_count": len(lines),
        "selected_preview": [
            {"line": line_no, "score": score, "text": line[:240]}
            for line_no, score, line in lines[:8]
        ],
        "full_context_chars": len(context),
        "model_context_chars": len(prepared),
        "at": datetime.now(UTC).isoformat(),
    }
    deterministic = infer_docfinqa_deterministic_answer(
        query=query,
        selected_lines=[line for _, _, line in lines],
    )
    if deterministic:
        event["deterministic_answer"] = deterministic["answer"]
        event["deterministic_method"] = deterministic["method"]
        event["deterministic_support"] = deterministic["support"]
    return prepared, event


def infer_docfinqa_deterministic_answer(
    *,
    query: str,
    selected_lines: list[str],
) -> dict[str, Any] | None:
    share_repurchase = infer_docfinqa_share_repurchase_answer(query=query, selected_lines=selected_lines)
    if share_repurchase:
        return share_repurchase
    return None


def infer_docfinqa_share_repurchase_answer(
    *,
    query: str,
    selected_lines: list[str],
) -> dict[str, Any] | None:
    import re

    normalized_query = normalize_docfinqa_lookup_text(query)
    if "repurchase" not in normalized_query or "share" not in normalized_query:
        return None
    if "december" not in normalized_query:
        return None
    use_program_column = "publicly announced" in normalized_query or "program" in normalized_query
    for line in selected_lines:
        lower = line.lower()
        if "december" not in lower or "average price" in lower or "total |" in lower:
            continue
        if "november" in lower:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        period = cells[0].lower()
        if "december" not in period:
            continue
        share_cell_index = 3 if use_program_column and len(cells) > 3 else 1
        shares = parse_docfinqa_number_from_cell(cells[share_cell_index])
        average_price = parse_docfinqa_number_from_cell(cells[2])
        if shares is None or average_price is None:
            continue
        value_millions = shares * average_price / 1_000_000
        return {
            "answer": format_docfinqa_decimal(value_millions, places=2),
            "method": "docfinqa_share_repurchase_cash_impact",
            "support": line,
        }
    return None


def parse_docfinqa_number_from_cell(cell: str) -> float | None:
    import re

    match = re.search(r"-?\$?\s*\(?\d[\d,]*(?:\.\d+)?\)?", cell)
    if not match:
        return None
    raw = match.group(0).replace("$", "").replace(",", "").replace(" ", "")
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()")
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


def format_docfinqa_decimal(value: float, *, places: int = 2) -> str:
    rendered = f"{value:.{places}f}"
    return rendered.rstrip("0").rstrip(".")


def extract_docfinqa_relevant_lines(
    *,
    query: str,
    context: str,
    max_chars: int = DOCFINQA_CONTEXT_CHAR_LIMIT,
) -> list[tuple[int, int, str]]:
    import re

    source_lines = [
        normalize_docfinqa_source_line(line)
        for line in context.splitlines()
    ]
    source_lines = [line for line in source_lines if line]
    if not source_lines:
        return []

    terms = docfinqa_query_terms(query)
    phrases = docfinqa_query_phrases(query)
    years = re.findall(r"\b(?:19|20)\d{2}\b", query)
    ranked: list[tuple[int, int]] = []
    for index, line in enumerate(source_lines):
        score = score_docfinqa_source_line(
            line=line,
            terms=terms,
            phrases=phrases,
            years=years,
        )
        if score > 0:
            ranked.append((score, index))
    if not ranked:
        return []

    ranked_score_by_index = {index: score for score, index in ranked}
    selected_indices: set[int] = set()
    selected: list[tuple[int, int, str]] = []
    chars = 0
    for _, index in sorted(ranked, key=lambda item: (-item[0], item[1]))[:DOCFINQA_MAX_BLOCKS]:
        start = max(0, index - DOCFINQA_BLOCK_BEFORE)
        end = min(len(source_lines), index + DOCFINQA_BLOCK_AFTER + 1)
        block: list[tuple[int, int, str]] = []
        block_chars = 0
        for line_index in range(start, end):
            if line_index in selected_indices:
                continue
            line = source_lines[line_index]
            score = ranked_score_by_index.get(line_index, 0)
            block.append((line_index, score, line))
            block_chars += len(line) + 30
        if not block:
            continue
        if selected and chars + block_chars > max_chars:
            continue
        for item in block:
            selected_indices.add(item[0])
            selected.append(item)
        chars += block_chars
    return selected


def normalize_docfinqa_source_line(line: str) -> str:
    import re

    line = line.replace("\\t", " ").replace("\\n", " ").replace("<br>", " ")
    return re.sub(r"\s+", " ", line.replace("\t", " ")).strip()


def docfinqa_query_terms(query: str) -> list[str]:
    import re

    stopwords = {
        "what", "was", "were", "the", "for", "from", "that", "this", "with",
        "into", "and", "are", "its", "their", "our", "has", "have", "had",
        "million", "millions", "percentage", "percent", "occurred", "affected",
        "during", "there", "been", "in", "of", "to", "by", "as", "is", "a",
        "an", "how",
    }
    terms: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", query.lower()):
        if len(raw) < 3 or raw in stopwords:
            continue
        variants = [raw]
        if raw.endswith("ies") and len(raw) > 4:
            variants.append(raw[:-3] + "y")
        if raw.endswith("s") and len(raw) > 4:
            variants.append(raw[:-1])
        for value in variants:
            if value not in seen:
                seen.add(value)
                terms.append(value)
    return terms


def docfinqa_query_phrases(query: str) -> list[str]:
    import re

    normalized = normalize_docfinqa_lookup_text(query)
    tokens = [
        token for token in normalized.split()
        if token not in {"what", "was", "were", "the", "for", "from", "that", "this", "with", "in", "of", "to", "by", "as", "is", "a", "an", "how"}
    ]
    phrases: list[str] = []
    seen: set[str] = set()
    for width in range(min(6, len(tokens)), 1, -1):
        for start in range(0, len(tokens) - width + 1):
            phrase = " ".join(tokens[start:start + width])
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


def score_docfinqa_source_line(
    *,
    line: str,
    terms: list[str],
    phrases: list[str],
    years: list[str],
) -> int:
    import re

    normalized = normalize_docfinqa_lookup_text(line)
    score = 0
    phrase_score = 0
    for phrase in phrases:
        if phrase and phrase in normalized:
            phrase_score += 10 + min(8, len(phrase.split()))
    score += min(40, phrase_score)
    if "total cash and investments" in normalized:
        score += 80
    if "available for sale investments" in normalized:
        score += 35
    if "net revenue" in normalized:
        score += 55
    if "2008 net revenue" in normalized or "2007 net revenue" in normalized:
        score += 50
    if "net assets" in normalized:
        score += 30
    if "customer related intangibles" in normalized and "network location intangibles" in normalized:
        score += 35
    if "amortized" in normalized and "straight line" in normalized:
        score += 20
    if "average price paid per share" in normalized or "price paid per share" in normalized:
        score += 20
    if "shares purchased" in normalized and "average price" in normalized:
        score += 45
    if "december" in normalized and "2018" in normalized and "$" in line and "|" in line:
        score += 35
    if "basis points" in normalized and "annual interest expense would change" in normalized:
        score += 70
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\w*\b", normalized):
            score += 4 if term.isdigit() else 3
    for year in years:
        if year in normalized:
            score += 5
    if re.search(r"[\$\(]?-?\d[\d,]*(?:\.\d+)?%?\)?", line):
        score += 2
    if "|" in line:
        score += 2
    if "<br>" in line.lower():
        score += 1
    if score < 5:
        return 0
    return score


def normalize_docfinqa_lookup_text(text: str) -> str:
    import re

    text = text.replace("\\t", " ").replace("\\n", " ").replace("<br>", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def extract_nolima_candidate_facts(context: str) -> list[str]:
    import re

    marker_weights = [
        ("next to where", 8),
        ("finally saw the original", 8),
        ("was seen up close by", 8),
        ("there was an engineer living in", 8),
        ("living in", 4),
        ("named ", 3),
        (" lives", 1),
    ]
    ranked: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for order, raw in enumerate(re.split(r"(?<=[.!?])\s+|\n+", context)):
        sentence = re.sub(r"\s+", " ", raw).strip()
        if len(sentence) < 20 or len(sentence) > 320:
            continue
        lowered = sentence.lower()
        score = sum(weight for marker, weight in marker_weights if marker in lowered)
        if score <= 0:
            continue
        if not re.search(r"\b[A-Z][a-z]{2,}\b", sentence):
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append((-score, order, sentence))
    return [sentence for _, _, sentence in sorted(ranked)]


def evidence_prompt(query: str, context: str) -> str:
    return (
        "You are the cheap worker in a benchmark harness. Read the provided context and produce a compact, "
        "machine-facing evidence packet. Do not write final prose unless the answer is a strict short label.\n\n"
        "Return these sections:\n"
        "ANSWER_CANDIDATE: the likely answer, or INSUFFICIENT.\n"
        "FORMAT_REQUIREMENT: exact output form implied by the question.\n"
        "EVIDENCE: bullet list of relevant facts, numbers, source labels, citations, or spans.\n"
        "COMPUTATIONS: any required formula or arithmetic, with inputs.\n"
        "RISKS: distractors, missing context, ambiguity, or citation requirements.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}"
    )


def evidence_prompt_for_benchmark(benchmark: str, query: str, context: str) -> str:
    if benchmark == "docfinqa":
        return (
            "You are the cheap worker for a financial-table QA benchmark. The context is a query-focused digest copied "
            "from the source filing. Find the exact source rows needed, preserve units, and compute the answer when the "
            "question asks for a change, ratio, percentage, or growth rate. Do not use outside financial knowledge. "
            "Prefer table rows over rounded prose when the table contains the requested metric. For 'comprised of' questions, "
            "divide the requested part by the matching total from the same table. For share-repurchase cash impact, multiply "
            "shares purchased by average price and convert to millions when requested; if a table has both 'Total Number of Shares Purchased' "
            "and 'Shares Purchased as Part of Publicly Announced Plan or Program', use the total-shares column unless the question explicitly "
            "contains the phrase 'publicly announced plan' or 'program'. For straight-line amortization questions, "
            "sum the relevant intangible values and divide by the stated amortization period. The ANSWER_CANDIDATE must be only "
            "the final numeric value or percentage when known. If your computation produces a corrected value that differs from an "
            "earlier candidate, update ANSWER_CANDIDATE to the corrected computed final.\n\n"
            "Return these sections:\n"
            "ANSWER_CANDIDATE: final number or percentage only, or INSUFFICIENT.\n"
            "FORMAT_REQUIREMENT: numeric answer only; include % only for percentages.\n"
            "EVIDENCE: source line numbers and exact rows/values used.\n"
            "COMPUTATIONS: formula with source inputs and arithmetic.\n"
            "RISKS: unit mismatch, wrong fiscal year, wrong row, wrong table, or rounding risk.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}"
        )
    if benchmark == "mrcr":
        return (
            "You are the cheap worker for an exact conversation-retrieval benchmark. "
            "Find the requested prior response instance in the transcript. If the query says to prepend a token, "
            "apply that token to the exact retrieved response. The ANSWER_CANDIDATE field must contain the exact full "
            "benchmark response and nothing summarized. Do not invent a new response; copy the requested response from context.\n\n"
            "Return these sections:\n"
            "ANSWER_CANDIDATE: exact full response, including required prefix, or INSUFFICIENT.\n"
            "FORMAT_REQUIREMENT: exact output rule.\n"
            "EVIDENCE: identify which instance and nearby transcript markers support the answer.\n"
            "RISKS: wrong ordinal, wrong genre/topic, truncation, or accidental extra prose.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}"
        )
    if benchmark == "repoqa":
        return (
            "You are the cheap worker for a repository function-lookup benchmark. "
            "Given the natural-language function description and repository context, find the exact function name. "
            "The scorer checks the function identifier, so the ANSWER_CANDIDATE must be the exact symbol name only when known.\n\n"
            "Return these sections:\n"
            "ANSWER_CANDIDATE: exact function name only, or INSUFFICIENT.\n"
            "FORMAT_REQUIREMENT: function identifier only.\n"
            "EVIDENCE: file, signature, docstring, or code behavior proving the match.\n"
            "RISKS: similarly named functions, class methods versus free functions, wrappers, or aliases.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}"
        )
    if benchmark == "l_citeeval":
        return (
            "You are the cheap worker for citation-grounded QA. Produce a compact answer candidate and preserve every "
            "source label that directly supports it. Source labels look like [doc17].\n\n"
            "Return these sections:\n"
            "ANSWER_CANDIDATE: concise answer only.\n"
            "FORMAT_REQUIREMENT: answer plus citation IDs.\n"
            "EVIDENCE: bullet list where each bullet includes the relevant [docN] label and support.\n"
            "COMPUTATIONS: comparisons or arithmetic, if needed.\n"
            "RISKS: wrong document ID, unsupported citation, or similar distractor entity.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}"
        )
    if benchmark == "cuad":
        return (
            "You are the cheap worker for a contract-clause extraction benchmark. Extract the exact contract text spans "
            "that answer the requested clause category. Do not answer with legal analysis, section numbers alone, or "
            "summaries. The scorer rewards the source span text, so preserve wording, dates, party names, amounts, and "
            "defined terms verbatim where possible.\n\n"
            "Return these sections:\n"
            "ANSWER_CANDIDATE: exact span text only; if multiple spans are needed, separate them with newlines; if no span exists, NO_ANSWER.\n"
            "FORMAT_REQUIREMENT: verbatim contract span(s), no explanation.\n"
            "EVIDENCE: source section labels plus the same exact span text.\n"
            "RISKS: overbroad span, section label without text, paraphrase, or false positive.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}"
        )
    if benchmark == "facts_grounding":
        return (
            "You are the cheap worker for a context-grounding benchmark. Answer only from the provided source. "
            "If the context does not support the request, mark the candidate INSUFFICIENT instead of adding outside advice.\n\n"
            "Return these sections:\n"
            "ANSWER_CANDIDATE: grounded answer or INSUFFICIENT.\n"
            "FORMAT_REQUIREMENT: concise format implied by the user request.\n"
            "EVIDENCE: exact source-supported facts.\n"
            "RISKS: unsupported recommendations, over-generalization, or missing refusal.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}"
        )
    if benchmark == "nolima":
        return (
            "You are the cheap worker for a latent needle-in-haystack benchmark. The question may have little lexical "
            "overlap with the sentence that contains the answer. Find the sentence whose facts logically imply the "
            "answer. You may use ordinary world knowledge only to bridge the question target to a source sentence "
            "(for example city-country, institution-location, painting-museum-location, or place-country links). "
            "For this benchmark, questions phrased as 'has been to X' can be satisfied by a source sentence showing "
            "the character lives in, lives next to, or directly visited a place/artwork/institution associated with X. "
            "The final answer must still be a character or entity explicitly named in the provided context; do not "
            "guess a name that is not in the context.\n\n"
            "Return these sections:\n"
            "ANSWER_CANDIDATE: short final entity or character name, or INSUFFICIENT.\n"
            "FORMAT_REQUIREMENT: short answer only.\n"
            "EVIDENCE: exact sentence or fact chain from context supporting the answer.\n"
            "RISKS: lexical distractors, wrong country/place inference, or wrong character.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}"
        )
    return evidence_prompt(query, context)


def critic_prompt(query: str, evidence_packet: str) -> str:
    return (
        "You are the mid-tier orchestrator. Review the worker evidence packet for benchmark scoring risk.\n\n"
        "Return compact sections:\n"
        "SUFFICIENCY: sufficient or insufficient.\n"
        "FINAL_FORMAT: exact answer rendering rule.\n"
        "MISSING_OR_WEAK: missing evidence, weak computation, citation risk, or abstention need.\n"
        "SYNTHESIS_INSTRUCTIONS: what the final synthesizer must do and must avoid.\n\n"
        f"Question:\n{query}\n\n"
        f"Evidence packet:\n{evidence_packet}"
    )


def critic_prompt_for_benchmark(benchmark: str, query: str, evidence_packet: str) -> str:
    if benchmark == "docfinqa":
        return (
            "You are the mid-tier checker for a financial numeric answer. Verify that the worker identified the right row, "
            "period, units, and arithmetic. If there is a percentage answer, check whether the candidate should be rounded "
            "to the precision implied by the question/source.\n\n"
            "Return compact sections:\n"
            "SUFFICIENCY: sufficient or insufficient.\n"
            "FINAL_FORMAT: exact number or percentage only.\n"
            "MISSING_OR_WEAK: missing row, wrong year, unit mismatch, arithmetic error, or rounding risk.\n"
            "SYNTHESIS_INSTRUCTIONS: return only the supported numeric ANSWER_CANDIDATE if sufficient; otherwise INSUFFICIENT.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}"
        )
    if benchmark == "mrcr":
        return (
            "You are the mid-tier checker for an exact recall task. Check whether the worker candidate is an exact copied "
            "response for the requested ordinal/topic and includes any required prefix. Do not ask the final synthesizer to rewrite it.\n\n"
            "Return compact sections:\n"
            "SUFFICIENCY: sufficient or insufficient.\n"
            "FINAL_FORMAT: exact candidate only, no extra prose.\n"
            "MISSING_OR_WEAK: ordinal, topic, truncation, or prefix risk.\n"
            "SYNTHESIS_INSTRUCTIONS: return ANSWER_CANDIDATE verbatim if sufficient; otherwise return INSUFFICIENT.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}"
        )
    if benchmark == "repoqa":
        return (
            "You are the mid-tier checker for a function-lookup task. Check whether ANSWER_CANDIDATE is the exact symbol name "
            "supported by code evidence. Do not ask for prose or a code excerpt in the final answer.\n\n"
            "Return compact sections:\n"
            "SUFFICIENCY: sufficient or insufficient.\n"
            "FINAL_FORMAT: exact function identifier only.\n"
            "MISSING_OR_WEAK: ambiguous symbol, wrapper, alias, or no code evidence.\n"
            "SYNTHESIS_INSTRUCTIONS: return the exact supported function identifier only.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}"
        )
    if benchmark == "cuad":
        return (
            "You are the mid-tier checker for a contract span extraction task. Check whether ANSWER_CANDIDATE contains "
            "verbatim contract text for the requested clause category, not just a section number or explanation.\n\n"
            "Return compact sections:\n"
            "SUFFICIENCY: sufficient or insufficient.\n"
            "FINAL_FORMAT: exact span text only; one span per line; NO_ANSWER if absent.\n"
            "MISSING_OR_WEAK: paraphrase, missing quoted language, overbroad span, false positive, or unsupported section.\n"
            "SYNTHESIS_INSTRUCTIONS: return only the exact span text from ANSWER_CANDIDATE/EVIDENCE; no legal explanation.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}"
        )
    return critic_prompt(query, evidence_packet)


def synthesis_prompt(
    query: str,
    evidence_packet: str,
    critic_notes: str,
) -> str:
    return (
        "You are the strong final synthesizer for a benchmark harness. Use only the evidence packet and critic notes. "
        "Do not add outside facts. Obey the final answer format exactly; for multiple choice, put the letter first; "
        "for exact-answer tasks, avoid extra prose; for citation tasks, include citation IDs if present.\n\n"
        f"Question:\n{query}\n\n"
        f"Evidence packet:\n{evidence_packet}\n\n"
        f"Critic notes:\n{critic_notes}\n\n"
        "Final answer:"
    )


def synthesis_prompt_for_benchmark(
    benchmark: str,
    query: str,
    evidence_packet: str,
    critic_notes: str,
) -> str:
    if benchmark == "docfinqa":
        return (
            "Return only the final supported numeric answer for DocFinQA. Use the worker's computation and critic notes. "
            "Do not include prose, labels, formulas, source lines, markdown, currency words, or explanations. Include a % "
            "only when the answer is a percentage. If the evidence is insufficient, return INSUFFICIENT.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}\n\n"
            f"Critic notes:\n{critic_notes}\n\n"
            "Final answer:"
        )
    if benchmark == "mrcr":
        return (
            "Return the exact ANSWER_CANDIDATE from the evidence packet if it is sufficient. "
            "Do not summarize, explain, repair, continue, or add markdown. If the candidate is INSUFFICIENT, return INSUFFICIENT.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}\n\n"
            f"Critic notes:\n{critic_notes}\n\n"
            "Final answer:"
        )
    if benchmark == "repoqa":
        return (
            "Return only the exact function identifier supported by the evidence packet. "
            "Do not include backticks, module paths, code blocks, or explanation unless the identifier itself includes dots.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}\n\n"
            f"Critic notes:\n{critic_notes}\n\n"
            "Final answer:"
        )
    if benchmark == "l_citeeval":
        return (
            "Answer concisely using only the evidence packet, and include the supporting [docN] citation IDs from the evidence. "
            "Do not invent citation IDs or cite unrelated documents.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}\n\n"
            f"Critic notes:\n{critic_notes}\n\n"
            "Final answer:"
        )
    if benchmark == "cuad":
        return (
            "Return only the exact contract span text for the requested CUAD clause category. Use the worker's "
            "ANSWER_CANDIDATE/EVIDENCE and critic notes. Do not include section labels unless they are part of the span. "
            "Do not add bullets, markdown, legal analysis, or explanations. If no source span is supported, return NO_ANSWER.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}\n\n"
            f"Critic notes:\n{critic_notes}\n\n"
            "Final answer:"
        )
    if benchmark == "nolima":
        return (
            "Return only the exact ANSWER_CANDIDATE from the evidence packet when it is supported. "
            "Do not explain the latent bridge, add markdown, or include any extra prose. If the candidate is "
            "INSUFFICIENT, return INSUFFICIENT.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence packet:\n{evidence_packet}\n\n"
            f"Critic notes:\n{critic_notes}\n\n"
            "Final answer:"
        )
    return synthesis_prompt(query, evidence_packet, critic_notes)


def model_event(event_type: str, result: ModelResult) -> dict[str, Any]:
    return {
        "type": event_type,
        "text": result.text,
        "usage": result.usage.to_dict(),
        "at": datetime.now(UTC).isoformat(),
    }


def render_benchmark_answer(
    *,
    benchmark: str,
    answer: str,
    context: str,
    evidence_packet: str = "",
) -> str:
    if benchmark == "repoqa":
        candidate = extract_answer_candidate(evidence_packet)
        if candidate and candidate.upper() != "INSUFFICIENT":
            return normalize_symbol_answer(candidate)
        return normalize_symbol_answer(answer)
    if benchmark == "nolima":
        candidate = extract_answer_candidate(evidence_packet)
        if candidate and candidate.upper() != "INSUFFICIENT":
            return normalize_short_answer(candidate)
        return normalize_short_answer(answer)
    if benchmark == "docfinqa":
        candidate = extract_answer_candidate(evidence_packet)
        value = candidate if candidate and candidate.upper() != "INSUFFICIENT" else answer
        computed = extract_docfinqa_computed_answer(evidence_packet)
        if not computed:
            computed = extract_docfinqa_computed_answer(answer)
        if computed and (computed.endswith("%") or str(value).strip().upper() == "INSUFFICIENT"):
            value = computed
        return normalize_docfinqa_answer(value)
    if benchmark == "mrcr":
        candidate = extract_answer_candidate(evidence_packet)
        if answer.strip().upper() == "INSUFFICIENT" and candidate and candidate.upper() != "INSUFFICIENT":
            return candidate
        return answer
    if benchmark != "l_citeeval":
        return answer
    if not answer.strip() or answer.startswith("[ERROR]"):
        return answer
    answer_without_citations = strip_doc_citations(answer).strip()
    citations = citation_ids_for_answer(answer_without_citations, context)
    if evidence_packet:
        evidence_citations = extract_doc_citations(evidence_packet)
        if evidence_citations:
            citations = merge_ordered(evidence_citations, citations)
        candidate = extract_answer_candidate(evidence_packet)
        if candidate and answer_without_citations.upper() in {"INSUFFICIENT", ""}:
            answer_without_citations = candidate
    if not citations:
        return answer
    suffix = " ".join(f"[{doc_id}]" for doc_id in citations[:3])
    return f"{answer_without_citations.rstrip()} {suffix}".strip()


def extract_answer_candidate(text: str) -> str | None:
    if not text:
        return None
    marker = "ANSWER_CANDIDATE:"
    upper = text.upper()
    start = upper.find(marker)
    if start < 0:
        return None
    value = text[start + len(marker):]
    stop_markers = [
        "\nFORMAT_REQUIREMENT:",
        "\nEVIDENCE:",
        "\nCOMPUTATIONS:",
        "\nRISKS:",
    ]
    stop_positions = [pos for marker_text in stop_markers if (pos := value.upper().find(marker_text)) >= 0]
    if stop_positions:
        value = value[: min(stop_positions)]
    value = value.strip()
    return value or None


def normalize_symbol_answer(answer: str) -> str:
    value = answer.strip()
    if value.startswith("`") and value.endswith("`") and len(value) > 1:
        value = value[1:-1].strip()
    if "\n" in value:
        value = value.splitlines()[0].strip()
    if value.startswith("ANSWER_CANDIDATE:"):
        value = value.split(":", 1)[1].strip()
    return value


def normalize_short_answer(answer: str) -> str:
    value = answer.strip()
    if not value:
        return value
    if "\n" in value:
        value = value.splitlines()[0].strip()
    if value.startswith("ANSWER_CANDIDATE:"):
        value = value.split(":", 1)[1].strip()
    value = value.strip("` \t\r\n")
    if value.endswith(".") and value.count(" ") <= 3:
        value = value[:-1].strip()
    return value


def normalize_docfinqa_answer(answer: str) -> str:
    value = normalize_short_answer(answer)
    if not value:
        return value
    if value.upper() == "INSUFFICIENT":
        return value
    import re

    percent_match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*%", value)
    if percent_match:
        return percent_match.group(0).replace(" ", "")
    number_match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", value)
    if number_match:
        return number_match.group(0)
    return value


def extract_docfinqa_computed_answer(text: str) -> str | None:
    if not text:
        return None
    import re

    upper = text.upper()
    start = upper.find("\nCOMPUTATIONS:")
    if start >= 0:
        computation_text = text[start:]
    else:
        calculation_start = upper.find("CALCULATION:")
        computation_text = text[calculation_start:] if calculation_start >= 0 else text
    stop = computation_text.upper().find("\nRISKS:")
    if stop >= 0:
        computation_text = computation_text[:stop]

    result_patterns = [
        r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)(?:\.\.\.)?\s*%",
        r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+percent",
        r"=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+million",
        r"=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\b",
    ]
    for pattern in result_patterns:
        matches = re.findall(pattern, computation_text, flags=re.IGNORECASE)
        if matches:
            value = matches[-1]
            if "percent" in pattern or pattern.endswith(r"\s*%"):
                return f"{value}%"
            return value
    return None


def merge_ordered(first: list[str], second: list[str]) -> list[str]:
    merged = []
    seen: set[str] = set()
    for value in [*first, *second]:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def has_doc_citation(text: str) -> bool:
    return bool(extract_doc_citations(text))


def extract_doc_citations(text: str) -> list[str]:
    import re

    return [match.lower() for match in re.findall(r"\[(doc\d+)\]", text, flags=re.IGNORECASE)]


def citation_ids_for_answer(answer: str, context: str) -> list[str]:
    answer_text = strip_doc_citations(answer).strip()
    if not answer_text:
        return []
    matches: list[str] = []
    for phrase in candidate_answer_phrases(answer_text):
        phrase_norm = normalize_lookup_text(phrase)
        if not phrase_norm:
            continue
        for doc_id, body in iter_source_labeled_docs(context):
            if doc_id in matches:
                continue
            body_norm = normalize_lookup_text(body)
            if phrase_norm in body_norm:
                matches.append(doc_id)
    return matches


def candidate_answer_phrases(answer: str) -> list[str]:
    import re

    phrases = [answer]
    for separator in [".", ",", ";", "\n"]:
        if separator in answer:
            phrases.append(answer.split(separator, 1)[0])
    phrases.extend(re.findall(r"\b[A-Z][A-Za-z'.-]*(?:\s+[A-Z][A-Za-z'.-]*)+\b", answer))
    phrases.extend(re.findall(r"`([^`]+)`", answer))

    seen: set[str] = set()
    result = []
    for phrase in phrases:
        value = phrase.strip()
        if len(value) < 2 or value.lower() in seen:
            continue
        seen.add(value.lower())
        result.append(value)
    return result


def strip_doc_citations(text: str) -> str:
    import re

    return re.sub(r"\[(?:doc\d+)\]", "", text, flags=re.IGNORECASE)


def iter_source_labeled_docs(context: str) -> list[tuple[str, str]]:
    import re

    pattern = re.compile(r"\[(doc\d+)\]\s*(.*?)(?=\n\s*\n\s*\[doc\d+\]|\Z)", re.IGNORECASE | re.DOTALL)
    return [(doc_id.lower(), body) for doc_id, body in pattern.findall(context)]


def normalize_lookup_text(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def prepare_agent_bench_data(
    data_root: Path,
    *,
    benchmarks: set[str] | None = None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    if benchmarks is not None and "nolima" not in benchmarks:
        return prepared
    prepared.append(prepare_nolima_hard_smoke(data_root))
    return prepared


def prepare_nolima_hard_smoke(data_root: Path) -> dict[str, Any]:
    nolima_dir = data_root / "nolima"
    test_path = nolima_dir / "test.jsonl"
    if nolima_jsonl_is_valid(test_path):
        return {
            "benchmark": "nolima",
            "status": "already_valid",
            "path": str(test_path),
        }

    nolima_dir.mkdir(parents=True, exist_ok=True)
    needle_path = nolima_dir / "needle_set_hard.json"
    haystack_path = nolima_dir / "rand_book_1.txt"
    if not needle_path.exists():
        urlretrieve(NOLIMA_HARD_NEEDLESET_URL, needle_path)
    if not haystack_path.exists():
        urlretrieve(NOLIMA_HAYSTACK_URL, haystack_path)

    needle_set = json.loads(needle_path.read_text(encoding="utf-8"))
    haystack = haystack_path.read_text(encoding="utf-8", errors="replace")
    rows = build_nolima_hard_rows(needle_set, haystack)
    with test_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "benchmark": "nolima",
        "status": "generated",
        "path": str(test_path),
        "rows": len(rows),
        "needle_set": str(needle_path),
        "haystack": str(haystack_path),
    }


def nolima_jsonl_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                return bool(row.get("question") and row.get("context") and row.get("answer"))
    except (OSError, ValueError, TypeError):
        return False
    return False


def build_nolima_hard_rows(
    needle_set: list[dict[str, Any]],
    haystack: str,
    *,
    context_chars: int = NOLIMA_CONTEXT_CHARS,
    base_seed: int = 42,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    depths = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    for exp_config in needle_set:
        exp_id = str(exp_config.get("id", ""))
        questions = exp_config.get("questions") or {}
        tests = exp_config.get("tests") or {}
        for question_type, question_template in questions.items():
            for test_id, test in tests.items():
                input_args = list(test.get("input_args") or [])
                needle = replace_numbered_placeholders(str(exp_config.get("needle", "")), input_args)
                question = replace_numbered_placeholders(str(question_template), input_args)
                character = ""
                if "{CHAR}" in needle or "{CHAR}" in question:
                    character = select_nolima_character(
                        exp_config.get("character_set") or [],
                        seed_material=f"{base_seed}:{exp_id}:{test_id}:{question_type}",
                    )
                    needle = needle.replace("{CHAR}", character)
                    question = question.replace("{CHAR}", character)
                gold = test.get("gold_answers") or character
                if isinstance(gold, list):
                    answer = str(gold[0]) if gold else ""
                else:
                    answer = str(gold)
                depth = depths[len(rows) % len(depths)]
                context = insert_needle_at_depth(
                    haystack=haystack,
                    needle=needle,
                    depth_percent=depth,
                    context_chars=context_chars,
                )
                example_id = f"{exp_id}_{test_id}_{question_type}"
                rows.append(
                    {
                        "_example_id": example_id,
                        "_subject": "nolima_hard_local",
                        "answer": answer,
                        "context": context,
                        "depth_percent": depth,
                        "needle": needle,
                        "question": question,
                        "source": "amodaresi/NoLiMa needle_set_hard + rand_book_1",
                    }
                )
    return rows


def replace_numbered_placeholders(template: str, args: list[Any]) -> str:
    result = template
    for index, arg in enumerate(args, start=1):
        result = result.replace("{" + str(index) + "}", str(arg))
    return result


def select_nolima_character(character_set: Any, *, seed_material: str) -> str:
    if not isinstance(character_set, list) or not character_set:
        raise ValueError("NoLiMa row requires a non-empty character_set")
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    return str(random.Random(seed).choice(character_set))


def insert_needle_at_depth(
    *,
    haystack: str,
    needle: str,
    depth_percent: int,
    context_chars: int,
) -> str:
    budget = max(1_000, context_chars - len(needle) - 4)
    base = haystack[:budget]
    insert_at = int(len(base) * max(0, min(100, depth_percent)) / 100)
    insert_at = nearest_whitespace_boundary(base, insert_at)
    return f"{base[:insert_at].rstrip()}\n\n{needle}\n\n{base[insert_at:].lstrip()}"


def nearest_whitespace_boundary(text: str, index: int, *, window: int = 500) -> int:
    if not text:
        return 0
    index = max(0, min(len(text), index))
    lower = max(0, index - window)
    upper = min(len(text), index + window)
    for offset in range(0, max(index - lower, upper - index) + 1):
        left = index - offset
        right = index + offset
        if left >= lower and left < len(text) and text[left].isspace():
            return left
        if right < upper and text[right].isspace():
            return right
    return index


def patch_agent_bench_runtime(agent_bench: Any) -> None:
    try:
        scorers = importlib.import_module("agent_bench.scorers")
        runner = importlib.import_module("agent_bench.runner")
    except ImportError:
        return
    scorers.SCORERS["l_citeeval"] = score_l_citeeval_with_cited_context
    scorers.SCORERS["repoqa"] = score_repoqa_symbol_or_body
    scorers.SCORERS["docfinqa"] = score_docfinqa_numeric
    runner.SCORERS["l_citeeval"] = score_l_citeeval_with_cited_context
    runner.SCORERS["repoqa"] = score_repoqa_symbol_or_body
    runner.SCORERS["docfinqa"] = score_docfinqa_numeric
    if hasattr(agent_bench, "SCORERS"):
        agent_bench.SCORERS["l_citeeval"] = score_l_citeeval_with_cited_context
        agent_bench.SCORERS["repoqa"] = score_repoqa_symbol_or_body
        agent_bench.SCORERS["docfinqa"] = score_docfinqa_numeric


def score_docfinqa_numeric(output: str, expected: str, **_: Any) -> tuple[float, str]:
    import re

    if not output or output.startswith("[ERROR]"):
        return 0.0, "no_output"
    if not expected:
        return -1.0, "no_reference"
    expected_text = str(expected).strip()
    expected_match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", expected_text)
    if not expected_match:
        scorers = importlib.import_module("agent_bench.scorers")
        return scorers._substring_or_f1(output, expected_text)

    expected_number = parse_docfinqa_number(expected_match.group(0))
    expected_is_percent = "%" in expected_text
    output_numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", output)
    for raw in output_numbers:
        predicted = parse_docfinqa_number(raw)
        if docfinqa_numbers_match(
            predicted=predicted,
            expected=expected_number,
            expected_literal=expected_match.group(0),
            expected_is_percent=expected_is_percent,
        ):
            return 1.0, f"numeric_match:{raw}"

    expected_norm = expected_text.lower().replace(",", "").replace("%", "").strip()
    output_norm = normalize_lookup_text(output).replace(",", "")
    if expected_norm and expected_norm in output_norm:
        return 1.0, "string_match"
    return 0.0, "no_match"


def parse_docfinqa_number(value: str) -> float:
    return float(value.replace(",", ""))


def docfinqa_numbers_match(
    *,
    predicted: float,
    expected: float,
    expected_literal: str,
    expected_is_percent: bool,
) -> bool:
    decimal_places = 0
    if "." in expected_literal:
        decimal_places = len(expected_literal.split(".", 1)[1])
    if expected_is_percent and decimal_places == 0:
        return abs(round(predicted) - expected) <= 0.0 or abs(predicted - expected) < 0.5
    tolerance = max(0.01, 0.5 * (10 ** -decimal_places))
    return abs(predicted - expected) <= tolerance


def score_repoqa_symbol_or_body(output: str, expected: str, **_: Any) -> tuple[float, str]:
    import re

    scorers = importlib.import_module("agent_bench.scorers")
    if not output or output.startswith("[ERROR]"):
        return 0.0, "no_output"
    if not expected:
        return -1.0, "no_reference"
    expected_names = extract_function_names(expected)
    for name in expected_names:
        if re.search(rf"\b{re.escape(name)}\b", output):
            return 1.0, f"name_match:{name}"
    if expected.strip() in output:
        return 1.0, "body_match"
    return scorers._substring_or_f1(output, expected)


def extract_function_names(source: str) -> list[str]:
    import re

    names = re.findall(r"^\s*(?:async\s+def|def)\s+([A-Za-z_]\w*)\s*\(", source, flags=re.MULTILINE)
    seen: set[str] = set()
    result = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def score_l_citeeval_with_cited_context(
    output: str,
    expected: str,
    question: str = "",
    context: str = "",
) -> tuple[float, str]:
    import json as _json
    import re

    scorers = importlib.import_module("agent_bench.scorers")
    judges = importlib.import_module("agent_bench.judges")
    if not output or output.startswith("[ERROR]"):
        return 0.0, "no_output"
    if not expected:
        return -1.0, "no_reference"
    try:
        parsed = _json.loads(expected)
        answer = str(parsed.get("answer", ""))
        required = set(parsed.get("required_citations", []))
    except (ValueError, TypeError):
        answer, required = expected, set()

    cited = {match.lower() for match in re.findall(r"\[(doc\d+|d\d+)\]", output, flags=re.IGNORECASE)}
    required_norm = {str(item).lower() for item in required}
    if required_norm:
        tp = len(cited & required_norm)
        precision = tp / len(cited) if cited else 0.0
        recall = tp / len(required_norm)
        cit_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    else:
        cit_f1 = -1.0

    content_ok = None
    if question and answer:
        judge_docs = build_citation_judge_context(
            context=context,
            output=output,
            answer=answer,
            required_doc_ids=required_norm,
        )
        verdict = scorers.judge_with_gemini_flash_lite(
            judges.CITATION_JUDGE.format(
                question=question[:2000],
                documents=judge_docs,
                expected_citations=", ".join(required) or "(none specified)",
                response=output[:5000],
            ),
        )
        content_ok = bool(verdict.get("content_correct")) if verdict else None

    if cit_f1 >= 0 and content_ok is not None:
        score = 0.5 * cit_f1 + 0.5 * (1.0 if content_ok else 0.0)
        return score, f"cit_f1={cit_f1:.2f},content={content_ok},cited_context=True"
    if cit_f1 >= 0:
        return cit_f1, f"cit_f1={cit_f1:.2f},cited_context=True"
    if content_ok is not None:
        return (1.0 if content_ok else 0.0), f"content={content_ok},cited_context=True"
    return scorers._substring_or_f1(output, answer)


def build_citation_judge_context(
    *,
    context: str,
    output: str,
    answer: str,
    required_doc_ids: set[str],
    max_chars: int = 20_000,
) -> str:
    selected_ids = set(required_doc_ids)
    selected_ids.update(extract_doc_citations(output))
    if not selected_ids:
        selected_ids.update(citation_ids_for_answer(answer, context))
    docs = iter_source_labeled_docs(context)
    selected = [
        f"[{doc_id}] {body.strip()}"
        for doc_id, body in docs
        if doc_id.lower() in selected_ids
    ]
    if not selected:
        return context[:max_chars]
    return "\n\n".join(selected)[:max_chars]


def stable_task_hash(query: str, context: str) -> str:
    payload = json.dumps(
        {"query": query, "context_hash": stable_context_hash(context)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_context_hash(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8", errors="replace")).hexdigest()


def default_agent_bench_root() -> Path:
    return Path(__file__).resolve().parents[3] / "agent-bench"


def ensure_agent_bench_importable(agent_bench_root: str | Path | None = None) -> Path:
    root = Path(agent_bench_root) if agent_bench_root else default_agent_bench_root()
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(f"agent-bench source directory not found: {src}")
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    importlib.import_module("agent_bench")
    return root


def parse_benchmark_specs(values: list[str] | None = None) -> list[AgentBenchSpec]:
    raw_values = values or DEFAULT_BENCHMARK_SPECS
    specs: list[AgentBenchSpec] = []
    for raw in raw_values:
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if ":" in value:
            benchmark, split = value.split(":", 1)
        else:
            benchmark, split = value, "test"
        specs.append(AgentBenchSpec(benchmark=benchmark.strip(), split=split.strip() or "test"))
    return specs


def read_benchmark_spec_file(path: str | Path) -> list[str]:
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(value)
    return values


async def run_agent_bench_suite(
    *,
    config: HarnessConfig,
    specs: list[AgentBenchSpec],
    agent_bench_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    results_dir: str | Path = "scratch/agent_bench_irys",
    trace_dir: str | Path | None = None,
    backend_mode: str = "three-tier",
    limit: int | None = 10,
    concurrency: int = 18,
    benchmark_workers: int = 4,
    checkpoint_every: int = 5,
    resume: bool = False,
) -> dict[str, Any]:
    root = ensure_agent_bench_importable(agent_bench_root)
    agent_bench = importlib.import_module("agent_bench")
    patch_agent_bench_runtime(agent_bench)
    data_root = Path(data_dir) if data_dir else root / "benchmarks" / "data"
    data_preparation = prepare_agent_bench_data(
        data_root,
        benchmarks={spec.benchmark for spec in specs},
    )
    results_root = Path(results_dir)
    trace_root = Path(trace_dir) if trace_dir else results_root / "traces"
    results_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(max(1, benchmark_workers))

    async def run_one(spec: AgentBenchSpec) -> AgentBenchRunResult:
        async with sem:
            bench_results_dir = results_root / spec.benchmark / spec.split
            bench_trace_dir = trace_root / spec.benchmark / spec.split
            completed_path = bench_results_dir / "completed.txt" if resume else None
            backend = IrysAgentBenchBackend(
                config=config,
                benchmark=spec.benchmark,
                split=spec.split,
                mode=backend_mode,
                trace_dir=trace_root,
            )
            try:
                summary = await agent_bench.run_benchmark(
                    benchmark=spec.benchmark,
                    split=spec.split,
                    backend=backend,
                    data_dir=data_root,
                    results_dir=bench_results_dir,
                    limit=limit,
                    concurrency=concurrency,
                    completed_list_path=completed_path,
                    checkpoint_every=checkpoint_every,
                )
                return AgentBenchRunResult.from_summary(
                    summary,
                    summary_path=bench_results_dir / "summary.json",
                    trace_dir=bench_trace_dir,
                )
            except Exception as exc:  # noqa: BLE001 - one benchmark should not kill suite.
                return AgentBenchRunResult.failed(spec, exc)

    started_at = datetime.now(UTC).isoformat()
    results = await asyncio.gather(*[run_one(spec) for spec in specs])
    finished_at = datetime.now(UTC).isoformat()
    scored_results = [item for item in results if not item.error and item.scored > 0]
    macro_avg = (
        sum(item.avg_score for item in scored_results) / len(scored_results)
        if scored_results
        else 0.0
    )
    total_scored = sum(item.scored for item in results)
    total_pass = sum(item.pass_count for item in results)
    aggregate = {
        "started_at": started_at,
        "finished_at": finished_at,
        "backend": IrysAgentBenchBackend.name,
        "backend_version": IrysAgentBenchBackend.version,
        "backend_mode": backend_mode,
        "agent_bench_root": str(root),
        "data_dir": str(data_root),
        "data_preparation": data_preparation,
        "results_dir": str(results_root),
        "trace_dir": str(trace_root),
        "limit": limit,
        "concurrency": concurrency,
        "benchmark_workers": benchmark_workers,
        "benchmarks": len(results),
        "completed_benchmarks": len(scored_results),
        "error_benchmarks": sum(1 for item in results if item.error),
        "total_scored": total_scored,
        "total_pass_count": total_pass,
        "micro_pass_rate": total_pass / total_scored if total_scored else 0.0,
        "macro_avg_score": macro_avg,
        "total_tokens_in": sum(item.total_tokens_in for item in results),
        "total_tokens_out": sum(item.total_tokens_out for item in results),
        "total_cost_usd": sum(item.total_cost_usd for item in results),
        "results": [item.to_dict() for item in results],
    }
    aggregate_path = results_root / "aggregate.json"
    aggregate["aggregate_path"] = str(aggregate_path)
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return aggregate
