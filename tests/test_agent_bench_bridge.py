from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from irys_harness.agent_bench_bridge import (
    IrysAgentBenchBackend,
    build_citation_judge_context,
    evidence_prompt_for_benchmark,
    extract_answer_candidate,
    extract_function_names,
    parse_benchmark_specs,
    render_benchmark_answer,
    score_repoqa_symbol_or_body,
    stable_task_hash,
)
from irys_harness.config import ModelConfig, ModelTier, parse_config
from irys_harness.metrics import ModelCallRecord
from irys_harness.models.gemini import ModelResult


def bridge_test_config():
    return parse_config(
        {
            "models": {
                "cheap_worker": {"model": "cheap"},
                "mid_orchestrator": {"model": "mid"},
                "strong_synthesizer": {"model": "strong"},
            },
            "module_tiers": {
                "extraction": "cheap_worker",
                "critic": "mid_orchestrator",
                "synthesis": "strong_synthesizer",
            },
        }
    )


class FakeRouter:
    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[str] = []

    def generate(self, *, module: str, prompt: str, temperature: float, max_output_tokens: int):
        self.calls.append(module)
        model_config = self.config.model_for_module(module)
        text_by_module = {
            "extraction": "ANSWER_CANDIDATE: B\nEVIDENCE: relevant passage",
            "critic": "SUFFICIENCY: sufficient\nFINAL_FORMAT: answer only",
            "synthesis": "B",
        }
        usage = ModelCallRecord.from_usage(
            module=module,
            model_config=ModelConfig(
                tier=model_config.tier,
                model=model_config.model,
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
            ),
            input_tokens=max(1, len(prompt.split())),
            output_tokens=3,
            latency_seconds=0.01,
        )
        return ModelResult(text=text_by_module[module], usage=usage)


class AgentBenchBridgeTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_benchmark_specs_defaults_split_to_test(self) -> None:
        specs = parse_benchmark_specs(["longbench_v2:train", "financebench"])
        self.assertEqual(specs[0].benchmark, "longbench_v2")
        self.assertEqual(specs[0].split, "train")
        self.assertEqual(specs[1].benchmark, "financebench")
        self.assertEqual(specs[1].split, "test")

    def test_stable_task_hash_ignores_context_text_after_hashing(self) -> None:
        first = stable_task_hash("question", "context")
        second = stable_task_hash("question", "context")
        changed = stable_task_hash("question", "different")
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_l_citeeval_renderer_adds_source_label_from_context(self) -> None:
        context = "[doc1] Erich Haenisch died in 1966.\n\n[doc2] William Pooley died in 1629."
        rendered = render_benchmark_answer(
            benchmark="l_citeeval",
            answer="William Pooley",
            context=context,
        )
        self.assertEqual(rendered, "William Pooley [doc2]")

    def test_l_citeeval_renderer_handles_longer_answer_sentence(self) -> None:
        context = "[doc1] Erich Haenisch died in 1966.\n\n[doc2] Sir William Pooley died in 1629."
        rendered = render_benchmark_answer(
            benchmark="l_citeeval",
            answer="Sir William Pooley died first, on 5 August 1629.",
            context=context,
        )
        self.assertEqual(rendered, "Sir William Pooley died first, on 5 August 1629. [doc2]")

    def test_l_citeeval_renderer_cites_multiple_named_entities(self) -> None:
        context = "[doc1] Erich Haenisch died in 1966.\n\n[doc2] Sir William Pooley died in 1629."
        rendered = render_benchmark_answer(
            benchmark="l_citeeval",
            answer="Sir William Pooley died first. Erich Haenisch died later.",
            context=context,
        )
        self.assertEqual(rendered, "Sir William Pooley died first. Erich Haenisch died later. [doc2] [doc1]")

    def test_l_citeeval_renderer_replaces_bad_synthesis_citations_from_evidence(self) -> None:
        context = "[doc17] Puka Punchu is in Peru.\n\n[doc56] Yaritani is in Peru."
        rendered = render_benchmark_answer(
            benchmark="l_citeeval",
            answer="Yes. [doc16] [doc32]",
            context=context,
            evidence_packet="ANSWER_CANDIDATE: Yes.\nEVIDENCE:\n* [doc17] Puka Punchu is in Peru.\n* [doc56] Yaritani is in Peru.",
        )
        self.assertEqual(rendered, "Yes. [doc17] [doc56]")

    def test_repoqa_renderer_prefers_exact_answer_candidate_symbol(self) -> None:
        rendered = render_benchmark_answer(
            benchmark="repoqa",
            answer="`dequantize`",
            context="",
            evidence_packet="ANSWER_CANDIDATE: `get_dequantize_func`\nEVIDENCE: signature supports it",
        )
        self.assertEqual(rendered, "get_dequantize_func")

    def test_repoqa_scorer_accepts_function_name_when_expected_is_body(self) -> None:
        expected = "    def _dequantize(self, param):\n        return param\n"
        score, detail = score_repoqa_symbol_or_body("_dequantize", expected)
        self.assertEqual(score, 1.0)
        self.assertEqual(detail, "name_match:_dequantize")

    def test_extract_function_names_from_expected_body(self) -> None:
        names = extract_function_names("def first():\n    pass\n\n    async def second(self):\n        pass")
        self.assertEqual(names, ["first", "second"])

    def test_mrcr_renderer_uses_candidate_when_synthesis_returns_insufficient(self) -> None:
        rendered = render_benchmark_answer(
            benchmark="mrcr",
            answer="INSUFFICIENT",
            context="",
            evidence_packet="ANSWER_CANDIDATE: abcExact prior answer\nFORMAT_REQUIREMENT: exact response",
        )
        self.assertEqual(rendered, "abcExact prior answer")

    def test_extract_answer_candidate_stops_before_structured_sections(self) -> None:
        candidate = extract_answer_candidate(
            "ANSWER_CANDIDATE: target answer\nFORMAT_REQUIREMENT: exact\nEVIDENCE: source"
        )
        self.assertEqual(candidate, "target answer")

    def test_mrcr_worker_prompt_demands_exact_candidate(self) -> None:
        prompt = evidence_prompt_for_benchmark("mrcr", "Prepend X to the 2nd poem.", "transcript")
        self.assertIn("exact full benchmark response", prompt)
        self.assertIn("copy the requested response", prompt)

    def test_citation_judge_context_selects_cited_doc_not_prefix_window(self) -> None:
        context = "[doc1] irrelevant\n\n[doc49] William Pooley died in 1629."
        packet = build_citation_judge_context(
            context=context,
            output="William Pooley [doc49]",
            answer="William Pooley",
            required_doc_ids=set(),
        )
        self.assertIn("[doc49]", packet)
        self.assertNotIn("[doc1]", packet)

    async def test_direct_backend_writes_private_trace(self) -> None:
        config = bridge_test_config()
        router = FakeRouter(config)
        with tempfile.TemporaryDirectory() as temp:
            backend = IrysAgentBenchBackend(
                config=config,
                benchmark="facts_grounding",
                split="public",
                mode="direct",
                trace_dir=temp,
                router=router,  # type: ignore[arg-type]
            )
            result = await backend.run(query="What is true?", context="B is true.")
            self.assertIn("ANSWER_CANDIDATE", result.answer)
            self.assertEqual(router.calls, ["extraction"])
            trace_files = list(Path(temp).rglob("*.json"))
            self.assertEqual(len(trace_files), 1)
            self.assertGreater(result.tokens_in, 0)
            self.assertGreater(result.cost_usd, 0)

    async def test_three_tier_backend_routes_across_all_tiers(self) -> None:
        config = bridge_test_config()
        router = FakeRouter(config)
        with tempfile.TemporaryDirectory() as temp:
            backend = IrysAgentBenchBackend(
                config=config,
                benchmark="longbench_v2",
                split="train",
                mode="three-tier",
                trace_dir=temp,
                router=router,  # type: ignore[arg-type]
            )
            result = await backend.run(query="Pick one.", context="B is supported.")
            self.assertEqual(result.answer, "B")
            self.assertEqual(router.calls, ["extraction", "critic", "synthesis"])
            trace_text = next(Path(temp).rglob("*.json")).read_text(encoding="utf-8")
            self.assertIn('"cheap_worker"', trace_text)
            self.assertIn('"mid_orchestrator"', trace_text)
            self.assertIn('"strong_synthesizer"', trace_text)

    async def test_adaptive_backend_uses_full_context_direct_route_when_better(self) -> None:
        config = bridge_test_config()
        router = FakeRouter(config)
        with tempfile.TemporaryDirectory() as temp:
            backend = IrysAgentBenchBackend(
                config=config,
                benchmark="mrcr",
                split="2needle",
                mode="adaptive",
                trace_dir=temp,
                router=router,  # type: ignore[arg-type]
            )
            result = await backend.run(query="Repeat the second answer.", context="large transcript")
            self.assertIn("ANSWER_CANDIDATE", result.answer)
            self.assertEqual(router.calls, ["extraction"])
            trace_text = next(Path(temp).rglob("*.json")).read_text(encoding="utf-8")
            self.assertIn('"selected_pipeline": "direct"', trace_text)
            self.assertIn('"route_decision"', trace_text)

    async def test_adaptive_backend_keeps_three_tier_route_for_structured_extraction(self) -> None:
        config = bridge_test_config()
        router = FakeRouter(config)
        with tempfile.TemporaryDirectory() as temp:
            backend = IrysAgentBenchBackend(
                config=config,
                benchmark="cuad",
                split="train",
                mode="adaptive",
                trace_dir=temp,
                router=router,  # type: ignore[arg-type]
            )
            result = await backend.run(query="Extract the clause.", context="contract text")
            self.assertEqual(result.answer, "B")
            self.assertEqual(router.calls, ["extraction", "critic", "synthesis"])
            trace_text = next(Path(temp).rglob("*.json")).read_text(encoding="utf-8")
            self.assertIn('"selected_pipeline": "three-tier"', trace_text)


if __name__ == "__main__":
    unittest.main()
