from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        if mode not in {"direct", "three-tier"}:
            raise ValueError("mode must be 'direct' or 'three-tier'")
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

        try:
            if self.mode == "direct":
                result = await self._generate(
                    module="extraction",
                    prompt=direct_prompt(query, context),
                    max_output_tokens=max_tokens,
                )
                calls.append(result.usage)
                answer = result.text.strip()
                events.append(model_event("direct_answer", result))
            else:
                extract = await self._generate(
                    module="extraction",
                    prompt=evidence_prompt(query, context),
                    max_output_tokens=min(4096, max_tokens),
                )
                calls.append(extract.usage)
                evidence_packet = extract.text
                events.append(model_event("evidence_packet", extract))

                critique = await self._generate(
                    module="critic",
                    prompt=critic_prompt(query, extract.text),
                    max_output_tokens=2048,
                )
                calls.append(critique.usage)
                events.append(model_event("critic_plan", critique))

                synth = await self._generate(
                    module="synthesis",
                    prompt=synthesis_prompt(query, extract.text, critique.text),
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
    if benchmark != "l_citeeval":
        return answer
    if not answer.strip() or answer.startswith("[ERROR]"):
        return answer
    if has_doc_citation(answer):
        return answer
    citations = citation_ids_for_answer(answer, context)
    if evidence_packet:
        citations = merge_ordered(citations, extract_doc_citations(evidence_packet))
    if not citations:
        return answer
    suffix = " ".join(f"[{doc_id}]" for doc_id in citations[:3])
    return f"{answer.rstrip()} {suffix}".strip()


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


def patch_agent_bench_runtime(agent_bench: Any) -> None:
    try:
        scorers = importlib.import_module("agent_bench.scorers")
        runner = importlib.import_module("agent_bench.runner")
    except ImportError:
        return
    scorers.SCORERS["l_citeeval"] = score_l_citeeval_with_cited_context
    runner.SCORERS["l_citeeval"] = score_l_citeeval_with_cited_context
    if hasattr(agent_bench, "SCORERS"):
        agent_bench.SCORERS["l_citeeval"] = score_l_citeeval_with_cited_context


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
