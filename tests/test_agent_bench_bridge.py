from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from irys_harness.agent_bench_bridge import (
    IrysAgentBenchBackend,
    build_nolima_hard_rows,
    build_citation_judge_context,
    evidence_prompt_for_benchmark,
    extract_docfinqa_relevant_lines,
    extract_docfinqa_computed_answer,
    extract_nolima_candidate_facts,
    extract_answer_candidate,
    extract_function_names,
    extract_mrcr_candidate_responses,
    insert_needle_at_depth,
    nolima_jsonl_is_valid,
    parse_mrcr_request,
    parse_benchmark_specs,
    prepare_prompt_context_for_benchmark,
    render_benchmark_answer,
    score_docfinqa_numeric,
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

    def test_mrcr_request_parser_extracts_prefix_ordinal_instruction(self) -> None:
        parsed = parse_mrcr_request(
            "Prepend ABC123 to the 2nd (1 indexed) song about agreements. "
            "Do not include any other text in your response."
        )
        self.assertEqual(parsed["prefix"], "ABC123")
        self.assertEqual(parsed["ordinal"], "2")
        self.assertEqual(parsed["instruction"], "song about agreements")

    def test_mrcr_digest_selects_requested_instance_response(self) -> None:
        context = (
            "[user] write a song about agreements\n\n"
            "[assistant] first song\n\n"
            "[user] write a poem about apples\n\n"
            "[assistant] apple poem\n\n"
            "[user] write a song about agreements\n\n"
            "[assistant] second song"
        )
        candidates = extract_mrcr_candidate_responses(context, "song about agreements")
        self.assertEqual([assistant for _, assistant in candidates], ["first song", "second song"])
        prepared, event = prepare_prompt_context_for_benchmark(
            "mrcr",
            context,
            query=(
                "Prepend XYZ to the 2nd (1 indexed) song about agreements. "
                "Do not include any other text in your response."
            ),
        )
        self.assertIn("MATCHED_ASSISTANT_RESPONSE:\nsecond song", prepared)
        self.assertIsNotNone(event)
        self.assertEqual(event["method"], "mrcr_exact_instance_digest")
        self.assertEqual(event["deterministic_answer"], "XYZsecond song")

    def test_nolima_worker_prompt_demands_latent_short_answer(self) -> None:
        prompt = evidence_prompt_for_benchmark("nolima", "Which character has been to France?", "book")
        self.assertIn("little lexical", prompt)
        self.assertIn("ANSWER_CANDIDATE: short final entity or character name", prompt)

    def test_nolima_renderer_prefers_short_candidate(self) -> None:
        rendered = render_benchmark_answer(
            benchmark="nolima",
            answer="The answer is Mandy.",
            context="",
            evidence_packet="ANSWER_CANDIDATE: Mandy.\nEVIDENCE: sentence",
        )
        self.assertEqual(rendered, "Mandy")

    def test_insert_needle_at_depth_preserves_needle(self) -> None:
        context = insert_needle_at_depth(
            haystack="alpha beta gamma delta " * 100,
            needle="Katie visited the target place.",
            depth_percent=50,
            context_chars=1500,
        )
        self.assertIn("Katie visited the target place.", context)
        self.assertLessEqual(len(context), 1550)

    def test_build_nolima_hard_rows_joins_question_needle_and_answer(self) -> None:
        needle_set = [
            {
                "id": "0409Inv",
                "needle": "There was an engineer living in {1}, named {CHAR}.",
                "character_set": ["Yuki", "Mandy"],
                "questions": {"onehop": "Which character has been to {2}?"},
                "tests": {"T10_C02": {"input_args": ["Calvinia", "South Africa"]}},
            }
        ]
        rows = build_nolima_hard_rows(needle_set, "book text " * 1000, context_chars=1000)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question"], "Which character has been to South Africa?")
        self.assertIn(rows[0]["answer"], {"Yuki", "Mandy"})
        self.assertIn(rows[0]["answer"], rows[0]["context"])
        self.assertIn("Calvinia", rows[0]["context"])

    def test_nolima_jsonl_validity_rejects_text_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.jsonl"
            path.write_text('{"text":"haystack only"}\n', encoding="utf-8")
            self.assertFalse(nolima_jsonl_is_valid(path))
            path.write_text(
                '{"question":"q","context":"ctx","answer":"Yuki"}\n',
                encoding="utf-8",
            )
            self.assertTrue(nolima_jsonl_is_valid(path))

    def test_nolima_candidate_digest_extracts_latent_fact_sentence(self) -> None:
        context = (
            "A distractor sentence about a garden.\n"
            "In 2013, after waiting in line for hours, Katie finally saw the original "
            "'Garden of Earthly Delights' painting up close.\n"
            "Another distractor."
        )
        candidates = extract_nolima_candidate_facts(context)
        self.assertEqual(len(candidates), 1)
        self.assertIn("Katie", candidates[0])

    def test_nolima_candidate_digest_ranks_inverted_needle(self) -> None:
        context = (
            "And if you do that, the whole game is up: your family lives are at risk.\n"
            "In 2013, the original 'Impression, Sunrise' painting was seen up close by Caleb, "
            "finally, after waiting in line for hours."
        )
        candidates = extract_nolima_candidate_facts(context)
        self.assertIn("Caleb", candidates[0])

    def test_nolima_prompt_context_logs_candidate_digest(self) -> None:
        prepared, event = prepare_prompt_context_for_benchmark(
            "nolima",
            "There was an engineer living in Firminy, named Yuki.",
        )
        self.assertIn("Candidate short factual sentences", prepared)
        self.assertIsNotNone(event)
        self.assertEqual(event["method"], "nolima_candidate_fact_digest")

    def test_docfinqa_digest_selects_query_table_rows(self) -> None:
        context = "\n".join(
            [
                "A generic introductory line.",
                "| Item | 2012 | 2011 |",
                "| :--- | :--- | :--- |",
                "| Deferred acquisition payments | $1.2 | $34.8 |",
                "Another generic line with 2012 but no useful value.",
            ]
        )
        lines = extract_docfinqa_relevant_lines(
            query="what percentage decrease occurred from 2011-2012 for deferred acquisition payments?",
            context=context,
        )
        joined = "\n".join(line for _, _, line in lines)
        self.assertIn("Deferred acquisition payments", joined)
        self.assertIn("$34.8", joined)

    def test_docfinqa_prompt_context_logs_numeric_digest(self) -> None:
        prepared, event = prepare_prompt_context_for_benchmark(
            "docfinqa",
            "| Total operating expenses | 41,932 | 38,391 |\n",
            query="what was the total operating expenses in 2018 in millions",
        )
        self.assertIn("Query-focused financial source digest", prepared)
        self.assertIsNotNone(event)
        self.assertEqual(event["method"], "docfinqa_query_numeric_digest")

    def test_docfinqa_renderer_prefers_numeric_candidate(self) -> None:
        rendered = render_benchmark_answer(
            benchmark="docfinqa",
            answer="The answer is approximately 53.2%.",
            context="",
            evidence_packet="ANSWER_CANDIDATE: 53.2%\nEVIDENCE: source",
        )
        self.assertEqual(rendered, "53.2%")

    def test_docfinqa_scorer_handles_percent_expected(self) -> None:
        score, detail = score_docfinqa_numeric("53.2%", "53%")
        self.assertEqual(score, 1.0)
        self.assertIn("numeric_match", detail)

    def test_docfinqa_renderer_prefers_corrected_computation(self) -> None:
        rendered = render_benchmark_answer(
            benchmark="docfinqa",
            answer="88.5%",
            context="",
            evidence_packet=(
                "ANSWER_CANDIDATE: 88.5%\n"
                "COMPUTATIONS:\n"
                "($14,001 / $26,302) * 100 = 53.23%\n"
                "RISKS: candidate was corrected."
            ),
        )
        self.assertEqual(rendered, "53.23%")

    def test_docfinqa_renderer_extracts_direct_calculation_result(self) -> None:
        rendered = render_benchmark_answer(
            benchmark="docfinqa",
            answer=(
                "To calculate the percent of net earnings for 2019:\n\n"
                "**Calculation:**\n"
                "($1,786.2 / $2,807.0) * 100 = 63.6337...%\n\n"
                "In 2019, the percent was approximately 63.6%."
            ),
            context="",
        )
        self.assertEqual(rendered, "63.6337%")

    def test_extract_docfinqa_computed_answer_from_millions(self) -> None:
        value = extract_docfinqa_computed_answer(
            "ANSWER_CANDIDATE: 51.28\nCOMPUTATIONS:\n"
            "1,327,657 * 42.61 / 1,000,000 = $56.57 million\nRISKS: rounding."
        )
        self.assertEqual(value, "56.57")

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

    async def test_adaptive_backend_keeps_lciteeval_on_three_tier_route(self) -> None:
        config = bridge_test_config()
        router = FakeRouter(config)
        with tempfile.TemporaryDirectory() as temp:
            backend = IrysAgentBenchBackend(
                config=config,
                benchmark="l_citeeval",
                split="test",
                mode="adaptive",
                trace_dir=temp,
                router=router,  # type: ignore[arg-type]
            )
            result = await backend.run(query="Which entity is supported? Cite sources.", context="[doc1] B is supported.")
            self.assertEqual(result.answer, "B")
            self.assertEqual(router.calls, ["extraction", "critic", "synthesis"])
            trace_text = next(Path(temp).rglob("*.json")).read_text(encoding="utf-8")
            self.assertIn('"selected_pipeline": "three-tier"', trace_text)

    async def test_adaptive_backend_keeps_docfinqa_on_three_tier_numeric_path(self) -> None:
        config = bridge_test_config()
        router = FakeRouter(config)
        with tempfile.TemporaryDirectory() as temp:
            backend = IrysAgentBenchBackend(
                config=config,
                benchmark="docfinqa",
                split="train",
                mode="adaptive",
                trace_dir=temp,
                router=router,  # type: ignore[arg-type]
            )
            result = await backend.run(
                query="what was the total operating expenses in 2018 in millions",
                context="| Total operating expenses | 41,932 | 38,391 |",
            )
            self.assertEqual(result.answer, "B")
            self.assertEqual(router.calls, ["extraction", "critic", "synthesis"])
            trace_text = next(Path(temp).rglob("*.json")).read_text(encoding="utf-8")
            self.assertIn('"selected_pipeline": "three-tier"', trace_text)
            self.assertIn('"docfinqa_query_numeric_digest"', trace_text)


if __name__ == "__main__":
    unittest.main()
