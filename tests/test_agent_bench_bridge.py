from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from irys_harness.agent_bench_bridge import (
    IrysAgentBenchBackend,
    build_citation_judge_context,
    parse_benchmark_specs,
    render_benchmark_answer,
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


if __name__ == "__main__":
    unittest.main()
