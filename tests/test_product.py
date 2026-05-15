from __future__ import annotations

from pathlib import Path
from http.server import ThreadingHTTPServer
import json
from types import SimpleNamespace
import threading
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from irys_harness.cli import build_parser
from irys_harness.config import ModelTier, load_config
from irys_harness.metrics import ModelCallRecord
from irys_harness.state import BenchmarkTask, RunState
from irys_harness.product import (
    DEFAULT_PRODUCT_TOP_K,
    build_answer_source_map,
    build_metric_selection_notes,
    build_held_back_inventory_for_prompt,
    build_product_evidence_items,
    build_product_plan_preview,
    build_product_synthesis_prompt,
    build_product_queries_from_state,
    build_product_worker_analysis_prompt,
    build_source_planner_prompt,
    compact_relevant_snippet,
    compare_product_traces,
    discover_corpus_paths,
    plan_product_corpus_scope,
    retrieve_product_chunks,
    run_product_matter,
    sanitize_matter_id,
    score_corpus_paths,
    select_supplemental_product_paths,
)
from irys_harness.product_ui import (
    INDEX_HTML,
    build_rerun_plan_note,
    build_handler,
    compact_trace_for_ui,
    list_product_traces,
    parse_paths,
    pick_local_paths,
    rerun_from_trace,
    rerun_context_paths,
    rerun_plan_from_trace,
    resolve_trace_path,
    selected_paths_for_rerun_payload,
    summarize_trace_rows,
)


class ProductMatterTests(unittest.TestCase):
    def test_discover_corpus_paths_filters_supported_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = root / "contract.txt"
            wanted.write_text("payment covenant notice cure period", encoding="utf-8")
            ignored = root / "archive.bin"
            ignored.write_bytes(b"ignored")

            paths = discover_corpus_paths([str(root)])

        self.assertEqual([path.name for path in paths], ["contract.txt"])

    def test_discover_corpus_paths_default_handles_large_matter_folder(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(90):
                (root / f"doc-{index:03d}.txt").write_text("matter document", encoding="utf-8")

            paths = discover_corpus_paths([str(root)])

        self.assertEqual(len(paths), 90)

    def test_product_plan_scopes_financial_lookup_to_likely_annual_report(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ten_k = root / "filings" / "sec" / "10-K"
            ten_q = root / "filings" / "sec" / "10-Q"
            forms = root / "filings" / "sec" / "144"
            ten_k.mkdir(parents=True)
            ten_q.mkdir(parents=True)
            forms.mkdir(parents=True)
            target = ten_k / "2025-02-20_0001561550-25-000025.pdf"
            target.write_text("2024 annual report EPS", encoding="utf-8")
            (ten_k / "2024-02-23_0001561550-24-000009.pdf").write_text("2023 annual report", encoding="utf-8")
            (ten_q / "2024-11-08_0001561550-24-000175.pdf").write_text("quarterly report", encoding="utf-8")
            (forms / "2024-12-02_0001561550-24-000199.pdf").write_text("insider sale", encoding="utf-8")
            (ten_k / "INDEX.md").write_text(
                "| Date | Accession | Doc | Description |\n"
                "| 2025-02-20 | `0001561550-25-000025` | ddog-20241231.htm | 10-K |\n"
                "| 2024-02-23 | `0001561550-24-000009` | ddog-20231231.htm | 10-K |\n",
                encoding="utf-8",
            )

            plan = build_product_plan_preview(
                objective="What was EPS in 2024?",
                paths=[str(root)],
            )

            self.assertIn(str(target.resolve()), plan["first_read_paths"])
            self.assertLess(plan["first_read_count"], plan["discovered_count"])
            self.assertIn("annual_report", plan["likely_document_families"])

    def test_product_plan_scopes_governance_question_to_board_materials(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            governance = root / "governance" / "board-minutes"
            contracts = root / "contracts"
            governance.mkdir(parents=True)
            contracts.mkdir()
            target = governance / "2024-06-01_board_minutes.txt"
            target.write_text("Board approved the financing.", encoding="utf-8")
            (contracts / "customer_msa.txt").write_text("ordinary contract", encoding="utf-8")

            plan = build_product_plan_preview(
                objective="Which board action approved the financing?",
                paths=[str(root)],
            )

            self.assertEqual(plan["first_read_paths"], [str(target.resolve())])
            self.assertIn("governance", plan["likely_document_families"])

    def test_product_plan_scopes_research_question_to_papers_without_sec_bias(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            papers = root / "biomedical_papers"
            legal = root / "agreements"
            papers.mkdir()
            legal.mkdir()
            target = papers / "clinical_trial_results_compound_alpha.txt"
            target.write_text("The clinical trial results showed response.", encoding="utf-8")
            (legal / "license_agreement.txt").write_text("license terms", encoding="utf-8")

            plan = build_product_plan_preview(
                objective="Summarize the clinical trial results for compound alpha.",
                paths=[str(root)],
            )

            self.assertEqual(plan["first_read_paths"], [str(target.resolve())])
            self.assertIn("research_paper", plan["likely_document_families"])

    def test_product_plan_treats_vague_issue_discovery_as_high_priority_routing_eval(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            email_chain = root / "Project Acorn - Email Chain.docx"
            resignations = root / "Letters of resignation.pdf"
            background_law = root / "Biz Corp Act courtesy copy pages 1-72.pdf"
            unknown_numbered = root / "[47535673]7.4.1.6.6_N(76758298.1).pdf"
            email_chain.write_text("Email thread discusses the practical dispute at length.", encoding="utf-8")
            resignations.write_text("Case-specific resignation letters.", encoding="utf-8")
            background_law.write_text("Generic corporate statute reference.", encoding="utf-8")
            unknown_numbered.write_text("Unlabeled attachment.", encoding="utf-8")

            plan = build_product_plan_preview(
                objective="What is the main issue here?",
                paths=[str(root)],
            )

            self.assertIn(str(email_chain.resolve()), plan["first_read_paths"])
            self.assertIn(str(resignations.resolve()), plan["first_read_paths"])
            self.assertNotIn(str(background_law.resolve()), plan["first_read_paths"])
            self.assertLess(plan["first_read_count"], plan["discovered_count"])
            self.assertIn("case-specific narrative sources", plan["needed_information"])

    def test_product_plan_keeps_data_dog_style_financial_routing_high_priority(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ten_k = root / "filings" / "sec" / "10-K"
            news = root / "filings" / "ir" / "news-releases"
            forms = root / "filings" / "sec" / "144"
            ten_k.mkdir(parents=True)
            news.mkdir(parents=True)
            forms.mkdir(parents=True)
            target = ten_k / "2025-02-20_0001561550-25-000025.pdf"
            target.write_text("2024 annual report EPS", encoding="utf-8")
            (news / "datadog-announces-fourth-quarter-2024-results.pdf").write_text("press release", encoding="utf-8")
            (forms / "2024-12-02_0001561550-24-000199.pdf").write_text("form 144", encoding="utf-8")
            (ten_k / "INDEX.md").write_text(
                "| 2025-02-20 | `0001561550-25-000025` | ddog-20241231.htm | 10-K |\n",
                encoding="utf-8",
            )

            plan = build_product_plan_preview(
                objective="What was EPS in 2024 in the Data Dog file?",
                paths=[str(root)],
            )

            self.assertIn(str(target.resolve()), plan["first_read_paths"])
            self.assertLess(plan["first_read_count"], plan["discovered_count"])

    def test_product_plan_can_use_cheap_worker_source_planner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "email_chain.txt"
            background = root / "background_rules.txt"
            target.write_text("matter-specific dispute", encoding="utf-8")
            background.write_text("general rules", encoding="utf-8")

            class FakeRouter:
                def __init__(self, config: object) -> None:
                    self.config = config

                def generate(self, **kwargs: object) -> SimpleNamespace:
                    self.kwargs = kwargs
                    return SimpleNamespace(
                        text=json.dumps(
                            {
                                "selected_paths": [str(target.resolve())],
                                "rejected_paths": [str(background.resolve())],
                                "should_read_full_corpus": False,
                                "reason": "The email chain is the matter-specific narrative source.",
                                "needed_information": ["event timeline"],
                                "confidence": "high",
                            }
                        ),
                        usage=ModelCallRecord(
                            module="product_source_planner",
                            tier=ModelTier.CHEAP_WORKER,
                            model="fake-cheap-worker",
                            input_tokens=100,
                            output_tokens=20,
                            estimated_cost=0.0001,
                        ),
                    )

            with patch("irys_harness.product.GeminiModelRouter", FakeRouter):
                plan = build_product_plan_preview(
                    objective="What is the main issue here?",
                    paths=[str(root)],
                    config=load_config(),
                    use_llm_planning=True,
                )

            self.assertEqual(plan["first_read_paths"], [str(target.resolve())])
            self.assertEqual(plan["source_planner"]["status"], "used")
            self.assertEqual(plan["source_planner"]["selected_count"], 1)
            self.assertIn("Worker source planner selected", plan["document_strategy"])
            self.assertIn("event timeline", plan["needed_information"])
            self.assertEqual(plan["planner_metrics"]["total_tokens"], 120)
            self.assertAlmostEqual(plan["planner_metrics"]["estimated_cost"], 0.0001)

    def test_cheap_worker_source_planner_can_select_relevant_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            emails = root / "matter_emails"
            background = root / "background_rules"
            emails.mkdir()
            background.mkdir()
            first_email = emails / "thread_001.txt"
            second_email = emails / "thread_002.txt"
            rulebook = background / "generic_policy.txt"
            first_email.write_text("The dispute is discussed here.", encoding="utf-8")
            second_email.write_text("Follow-up email with issue detail.", encoding="utf-8")
            rulebook.write_text("Generic background rule.", encoding="utf-8")

            class FakeRouter:
                def __init__(self, config: object) -> None:
                    self.config = config

                def generate(self, **kwargs: object) -> SimpleNamespace:
                    return SimpleNamespace(
                        text=json.dumps(
                            {
                                "selected_paths": [],
                                "selected_directories": [str(emails.resolve())],
                                "rejected_paths": [str(rulebook.resolve())],
                                "should_read_full_corpus": False,
                                "reason": "The emails folder is the case-specific narrative source.",
                                "needed_information": ["event timeline"],
                                "confidence": "high",
                            }
                        ),
                        usage=ModelCallRecord(
                            module="product_source_planner",
                            tier=ModelTier.CHEAP_WORKER,
                            model="fake-cheap-worker",
                            input_tokens=100,
                            output_tokens=30,
                            estimated_cost=0.0002,
                        ),
                    )

            with patch("irys_harness.product.GeminiModelRouter", FakeRouter):
                plan = build_product_plan_preview(
                    objective="What is the main issue here?",
                    paths=[str(root)],
                    config=load_config(),
                    use_llm_planning=True,
                )

            self.assertEqual(
                plan["first_read_paths"],
                [str(first_email.resolve()), str(second_email.resolve())],
            )
            self.assertEqual(plan["source_planner"]["status"], "used")
            self.assertEqual(plan["source_planner"]["selected_count"], 2)
            self.assertEqual(plan["source_planner"]["selected_directories"], [str(emails.resolve())])
            self.assertNotIn(str(rulebook.resolve()), plan["first_read_paths"])

    def test_product_queries_include_source_planner_needed_information(self) -> None:
        task = BenchmarkTask(
            benchmark="product_matter",
            task_id="matter",
            question="What is the main issue here?",
            context_files=[],
            answer_schema={},
            metadata={
                "corpus_scope_decision": {
                    "selected_paths": [str(Path("email_chain.txt").resolve())],
                    "signals": {
                        "source_planner": {
                            "reason": "The email chain is the matter-specific narrative source.",
                            "needed_information": ["event timeline", "parties and disputed action"],
                        }
                    },
                }
            },
        )
        state = RunState(task=task, config=load_config(), documents=[], chunks=[])

        queries = build_product_queries_from_state(state)

        self.assertTrue(any("event timeline" in query for query in queries))
        self.assertTrue(any("matter-specific narrative source" in query for query in queries))

    def test_packet_review_prompt_exposes_held_back_inventory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = (root / "generic_rules.txt").resolve()
            held_back = (root / "email_chain.txt").resolve()
            task = BenchmarkTask(
                benchmark="product_matter",
                task_id="matter",
                question="What is the main issue here?",
                context_files=[str(active)],
                answer_schema={},
                metadata={
                    "corpus_scope_decision": {
                        "discovered_paths": [str(active), str(held_back)],
                        "selected_paths": [str(active)],
                        "scored_paths": [
                            {"path": str(held_back), "score": 27, "reasons": ["likely case-specific narrative document"]}
                        ],
                    }
                },
            )
            state = RunState(
                task=task,
                config=load_config(),
                documents=[{"doc_id": "doc_0001", "filename": active.name, "path": str(active), "text_chars": 100}],
                chunks=[],
            )

            inventory = build_held_back_inventory_for_prompt(state)

            self.assertIn("email_chain.txt", inventory)
            self.assertIn("likely case-specific narrative document", inventory)

    def test_cheap_worker_source_planner_stages_large_indexed_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            releases = root / "news_releases"
            releases.mkdir()
            index = releases / "INDEX.md"
            index.write_text("Release index with dates and titles.", encoding="utf-8")
            for item in range(75):
                (releases / f"release-{item:03d}.txt").write_text("announcement body", encoding="utf-8")
            rules = root / "background_rules"
            rules.mkdir()
            rulebook = rules / "generic_policy.txt"
            rulebook.write_text("Generic background rule.", encoding="utf-8")

            class FakeRouter:
                def __init__(self, config: object) -> None:
                    self.config = config

                def generate(self, **kwargs: object) -> SimpleNamespace:
                    return SimpleNamespace(
                        text=json.dumps(
                            {
                                "selected_paths": [],
                                "selected_directories": [str(releases.resolve())],
                                "rejected_paths": [str(rulebook.resolve())],
                                "should_read_full_corpus": False,
                                "reason": "Compare announcements through the release directory.",
                                "needed_information": ["announcement dates", "announcement themes"],
                                "confidence": "medium",
                            }
                        ),
                        usage=ModelCallRecord(
                            module="product_source_planner",
                            tier=ModelTier.CHEAP_WORKER,
                            model="fake-cheap-worker",
                            input_tokens=120,
                            output_tokens=35,
                            estimated_cost=0.0002,
                        ),
                    )

            with patch("irys_harness.product.GeminiModelRouter", FakeRouter):
                plan = build_product_plan_preview(
                    objective="Compare the 2023 and 2024 announcements. How did priorities change?",
                    paths=[str(root)],
                    config=load_config(),
                    use_llm_planning=True,
                )

            self.assertEqual(plan["first_read_paths"], [str(index.resolve())])
            self.assertEqual(plan["source_planner"]["status"], "used")
            self.assertEqual(plan["source_planner"]["selected_count"], 1)
            self.assertEqual(plan["source_planner"]["staged_directories"][0]["document_count"], 76)
            self.assertIn("manifest", plan["source_planner"]["staged_directories"][0]["strategy"])

    def test_product_plan_fallback_scopes_announcement_comparison(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            releases = root / "ir" / "news-releases"
            releases.mkdir(parents=True)
            index = releases / "INDEX.md"
            index.write_text("Release index with dated announcement titles.", encoding="utf-8")
            for item in range(75):
                (releases / f"release-{item:03d}.txt").write_text("announcement body", encoding="utf-8")
            background = root / "policies"
            background.mkdir()
            for item in range(20):
                (background / f"policy-{item:03d}.txt").write_text("generic policy", encoding="utf-8")

            plan = build_product_plan_preview(
                objective="Compare the announcements in 2023 and 2024. How did priorities change?",
                paths=[str(root)],
            )

            self.assertLess(plan["first_read_count"], plan["discovered_count"])
            self.assertIn(str(index.resolve()), plan["first_read_paths"])
            self.assertTrue(all("news-releases" in path for path in plan["first_read_paths"]))

    def test_product_plan_prefers_source_family_index_over_periodic_filings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            news = root / "filings" / "ir" / "news-releases"
            sec_10k = root / "filings" / "sec" / "10-K"
            sec_10q = root / "filings" / "sec" / "10-Q"
            news.mkdir(parents=True)
            sec_10k.mkdir(parents=True)
            sec_10q.mkdir(parents=True)
            news_index = news / "INDEX.md"
            news_index.write_text(
                "- 2023 Datadog announces product launch\n- 2024 Datadog announces AI platform launch",
                encoding="utf-8",
            )
            for year in (2023, 2024):
                (news / f"datadog-announces-product-priorities-{year}.txt").write_text(
                    f"{year} announcement about product priorities.",
                    encoding="utf-8",
                )
            sec_index = sec_10k / "INDEX.md"
            sec_index.write_text(
                "| Filing Date | Accession | Document | Form |\n"
                "| 2024-02-23 | `0001561550-24-000009` | ddog-20231231.htm | 10-K |\n",
                encoding="utf-8",
            )
            (sec_10k / "2024-02-23_0001561550-24-000009.txt").write_text(
                "annual report summary",
                encoding="utf-8",
            )
            (sec_10q / "2024-05-08_0001561550-24-000051.txt").write_text(
                "quarterly report summary",
                encoding="utf-8",
            )

            plan = build_product_plan_preview(
                objective="Compare the announcements Datadog made in 2023 to 2024. How did priorities change?",
                paths=[str(root)],
            )

            self.assertEqual(plan["first_read_paths"], [str(news_index.resolve())])
            self.assertIn(
                "source-family index can guide broad comparison before reading every file",
                plan["top_candidates"][0]["reasons"],
            )
            self.assertTrue(
                all("sec" not in {part.lower() for part in Path(path).parts} for path in plan["first_read_paths"])
            )

    def test_source_planner_prompt_uses_bounded_inventory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(300):
                path = root / "archive" / f"irrelevant-{index:03d}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
                paths.append(path.resolve())
            target = root / "filings" / "datadog-announces-fourth-quarter-and-fiscal-year-2024-financial.pdf"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("eps", encoding="utf-8")
            paths.append(target.resolve())
            scored = score_corpus_paths("tell me about eps in 2024", paths)

            prompt = build_source_planner_prompt(
                objective="tell me about eps in 2024",
                plan_note=None,
                paths=paths,
                scored_paths=scored,
            )

            self.assertIn(target.name, prompt)
            self.assertIn('"selected_directories"', prompt)
            self.assertIn('"directory_summary"', prompt)
            self.assertIn('"omitted_candidate_count"', prompt)
            self.assertIn("source-family comparisons", prompt)
            self.assertNotIn("irrelevant-299.txt", prompt)

    def test_sec_fiscal_year_index_beats_filing_date_for_annual_report(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "filings" / "sec" / "10-K"
            root.mkdir(parents=True)
            index = root / "INDEX.md"
            index.write_text(
                "| Filing Date | Accession | Document | Form |\n"
                "| 2025-02-20 | `0001561550-25-000025` | ddog-20241231.htm | 10-K |\n"
                "| 2024-02-23 | `0001561550-24-000009` | ddog-20231231.htm | 10-K |\n",
                encoding="utf-8",
            )
            fiscal_2024 = root / "2025-02-20_0001561550-25-000025.pdf"
            fiscal_2023 = root / "2024-02-23_0001561550-24-000009.pdf"
            fiscal_2024.write_text("2024 annual report", encoding="utf-8")
            fiscal_2023.write_text("2023 annual report", encoding="utf-8")

            rows = score_corpus_paths("tell me about the eps in 2024", [index, fiscal_2024, fiscal_2023])

            self.assertEqual(Path(rows[0]["path"]).name, fiscal_2024.name)
            self.assertIn("index maps accession to fiscal/report year 2024", rows[0]["reasons"])

    def test_product_scope_respects_user_approved_plan_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")

            decision = plan_product_corpus_scope(
                objective="Read the corpus.",
                paths=[first, second],
                selected_paths=[str(second)],
            )

            self.assertEqual(decision.selected_paths, [second.resolve()])
            self.assertIn("User-approved", decision.reason)

    def test_run_product_matter_writes_trace_over_user_corpus(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text(
                "The borrower must deliver quarterly financial statements within 45 days. "
                "A 10 day cure period applies after written notice.",
                encoding="utf-8",
            )
            trace_dir = root / "traces"
            result = run_product_matter(
                objective="What notice and cure period applies to late financial statements?",
                paths=[str(doc)],
                matter_id="Acme Matter",
                config=load_config(),
                trace_dir=trace_dir,
                live_synthesis=False,
                verbose=False,
            )

            trace = result.state.to_trace()
            self.assertTrue(result.trace_path.exists())
            self.assertEqual(trace["benchmark"], "product_matter")
            self.assertEqual(trace["task_id"], "Acme-Matter")
            self.assertEqual(trace["documents"][0]["source_posture"], "user_provided_corpus")
            self.assertTrue(trace["retrieval_iterations"][0]["retrieved_chunks"])
            self.assertIn("10 day cure period", trace["rendered_answer"])
            self.assertEqual(trace["diagnosis"]["status"], "ready_for_review")
            self.assertEqual(trace["diagnosis"]["evidence_count"], 1)
            self.assertGreaterEqual(trace["diagnosis"]["answer_source_map_count"], 1)
            self.assertTrue(Path(trace["artifacts"][0]["path"]).exists())
            self.assertTrue(trace["artifacts"][0]["diagnostic"])
            self.assertTrue(result.to_dict()["artifacts"][0]["filename"].startswith("answer-run_"))
            event_messages = [event["message"] for event in trace["events"]]
            self.assertIn("Found supported documents", event_messages)
            self.assertIn("Choosing first-read documents", event_messages)
            saved_trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
            saved_labels = [event["label"] for event in saved_trace["events"]]
            self.assertIn("SAVE", saved_labels)
            self.assertIn("DONE", saved_labels)
            self.assertEqual(saved_labels[-1], "DONE")

    def test_run_product_matter_preserves_repeated_message_traces(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "source.txt"
            doc.write_text("The answer is preserved across repeated runs.", encoding="utf-8")
            trace_dir = root / "traces"
            kwargs = {
                "objective": "What is preserved?",
                "paths": [str(doc)],
                "matter_id": "Repeated Matter",
                "config": load_config(),
                "trace_dir": trace_dir,
                "live_synthesis": False,
                "verbose": False,
            }

            first = run_product_matter(**kwargs)
            second = run_product_matter(**kwargs)

            self.assertTrue(first.trace_path.exists())
            self.assertTrue(second.trace_path.exists())
            self.assertNotEqual(first.trace_path, second.trace_path)
            self.assertEqual(first.trace_path.name, "Repeated-Matter.json")
            self.assertIn(second.state.run_id, second.trace_path.name)

    def test_product_retrieval_keeps_source_diversity_for_selected_documents(self) -> None:
        documents = [
            {"doc_id": "doc_0001", "filename": "large-stale-report.txt"},
            {"doc_id": "doc_0002", "filename": "short-target-release.txt"},
        ]
        chunks = [
            {
                "doc_id": "doc_0001",
                "chunk_id": f"doc_0001_chunk_{index:04d}",
                "text": "2024 financial report general policy " * 80,
            }
            for index in range(8)
        ]
        chunks.append(
            {
                "doc_id": "doc_0002",
                "chunk_id": "doc_0002_chunk_0000",
                "text": "Fiscal Year 2024 Financial Highlights: GAAP net income per diluted share was $0.52.",
            }
        )

        retrieved = retrieve_product_chunks(
            chunks,
            ["2024 EPS", "earnings per share diluted basic"],
            documents,
            top_k=3,
        )

        self.assertIn("doc_0002", {item.doc_id for item in retrieved})

    def test_product_retrieval_hydrates_neighbor_context_for_split_tables(self) -> None:
        documents = [{"doc_id": "doc_0001", "filename": "annual-report.txt"}]
        chunks = [
            {
                "doc_id": "doc_0001",
                "chunk_id": "doc_0001_chunk_0000",
                "index": 0,
                "text": "Year ended December 31, 2024. Net income per share - basic $0.55.",
                "prev_chunk": None,
                "next_chunk": "doc_0001_chunk_0001",
            },
            {
                "doc_id": "doc_0001",
                "chunk_id": "doc_0001_chunk_0001",
                "index": 1,
                "text": "Net income per share - diluted $0.52. Weighted average shares 358,636.",
                "prev_chunk": "doc_0001_chunk_0000",
                "next_chunk": None,
            },
        ]

        retrieved = retrieve_product_chunks(chunks, ["diluted earnings per share 2024"], documents, top_k=1)
        evidence = build_product_evidence_items(retrieved, queries=["diluted earnings per share 2024"])

        self.assertIn("basic $0.55", evidence[0]["raw_support"])
        self.assertIn("diluted $0.52", evidence[0]["raw_support"])

    def test_product_retrieval_boosts_exact_metric_phrases(self) -> None:
        documents = [
            {"doc_id": "doc_0001", "filename": "tax-note.txt"},
            {"doc_id": "doc_0002", "filename": "eps-table.txt"},
        ]
        chunks = [
            {
                "doc_id": "doc_0001",
                "chunk_id": "doc_0001_chunk_0000",
                "text": "2024 income tax expense deferred tax assets uncertain tax positions " * 80,
            },
            {
                "doc_id": "doc_0002",
                "chunk_id": "doc_0002_chunk_0000",
                "text": (
                    "For the fiscal year ended December 31, 2024, net income per share - basic was $0.55 "
                    "and net income per share - diluted was $0.52. Weighted-average shares were used."
                ),
            },
        ]

        retrieved = retrieve_product_chunks(
            chunks,
            [
                "Tell me about EPS in 2024",
                "earnings per share diluted basic weighted average shares",
                "net income loss per share diluted basic fiscal year financial highlights",
            ],
            documents,
            top_k=1,
        )

        self.assertEqual(retrieved[0].doc_id, "doc_0002")
        self.assertIn("diluted was $0.52", retrieved[0].text)

    def test_product_retrieval_boosts_announcement_inventory_for_priority_questions(self) -> None:
        documents = [
            {"doc_id": "doc_0001", "filename": "quarterly-filing.txt"},
            {"doc_id": "doc_0002", "filename": "INDEX.md"},
        ]
        chunks = [
            {
                "doc_id": "doc_0001",
                "chunk_id": "doc_0001_chunk_0000",
                "text": "2023 2024 company financial statements convertible senior notes accounting policy " * 60,
            },
            {
                "doc_id": "doc_0002",
                "chunk_id": "doc_0002_chunk_0000",
                "text": (
                    "# Datadog press releases\n"
                    "- datadog-announces-bits-ai-assistant-help-engineers-quickly.pdf\n"
                    "- datadog-and-microsoft-announce-strategic-partnership.pdf\n"
                    "- datadog-acquires-eppo-expand-its-ai-product-analytics.pdf\n"
                    "- datadog-adds-identity-vulnerability-and-app-level-findings.pdf\n"
                ),
            },
        ]

        retrieved = retrieve_product_chunks(
            chunks,
            ["Compare 2023 and 2024 announcements. How did company priorities change?"],
            documents,
            top_k=1,
        )

        self.assertEqual(retrieved[0].doc_id, "doc_0002")
        self.assertIn("press releases", retrieved[0].text)

    def test_product_retrieval_keeps_announcement_sources_over_large_summary(self) -> None:
        documents = [
            {"doc_id": "doc_0001", "filename": "annual-report-2023.txt"},
            {"doc_id": "doc_0002", "filename": "first-quarter-2023-financial-results.txt"},
            {"doc_id": "doc_0003", "filename": "first-quarter-2024-financial-results.txt"},
            {"doc_id": "doc_0004", "filename": "security-ai-product-announcement-2024.txt"},
        ]
        chunks = [
            {
                "doc_id": "doc_0001",
                "chunk_id": f"doc_0001_chunk_{index:04d}",
                "text": (
                    "Annual report platform product security observability accounting policy "
                    "convertible senior notes financial statements "
                )
                * 35,
            }
            for index in range(12)
        ]
        chunks.extend(
            [
                {
                    "doc_id": "doc_0002",
                    "chunk_id": "doc_0002_chunk_0000",
                    "text": (
                        "Datadog Announces First Quarter 2023 Financial Results. "
                        "Business highlights included launched Data Streams Monitoring, "
                        "Application Vulnerability Management, and cloud security reports."
                    ),
                },
                {
                    "doc_id": "doc_0003",
                    "chunk_id": "doc_0003_chunk_0000",
                    "text": (
                        "Datadog Announces First Quarter 2024 Financial Results. "
                        "Business highlights included general availability of LLM Observability, "
                        "Bits AI, and expanded security automation."
                    ),
                },
                {
                    "doc_id": "doc_0004",
                    "chunk_id": "doc_0004_chunk_0000",
                    "text": (
                        "Datadog announced a 2024 product expansion for AI security, "
                        "incident management, platform integration, and observability workflows."
                    ),
                },
            ]
        )

        retrieved = retrieve_product_chunks(
            chunks,
            [
                "compare the announcements data dog made in 2023 to 2024. how did company priorities change?",
                "press release news release announcement announces launches expands acquires partnership product platform priorities",
            ],
            documents,
            top_k=3,
        )

        doc_ids = {item.doc_id for item in retrieved}
        self.assertIn("doc_0002", doc_ids)
        self.assertIn("doc_0003", doc_ids)
        self.assertNotEqual([item.doc_id for item in retrieved], ["doc_0001", "doc_0001", "doc_0001"])

    def test_packet_review_can_select_supplemental_held_back_documents(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "generic_rules.txt"
            target = root / "urgent_warranty_email_chain.txt"
            active.write_text("General background rules only.", encoding="utf-8")
            target.write_text("The key issue is warranty fraud discussed in the email chain.", encoding="utf-8")

            selected, profile, calls = select_supplemental_product_paths(
                objective="What is the main issue here?",
                discovered_paths=[active, target],
                active_paths=[active],
                packet_review={
                    "missing_information": ["matter-specific issue narrative"],
                    "revised_queries": ["warranty fraud email chain"],
                    "coverage_risks": ["current packet only has generic background"],
                },
                config=load_config(),
                use_llm_planning=False,
                limit=3,
            )

            self.assertEqual(calls, [])
            self.assertIn(target.resolve(), selected)
            self.assertEqual(profile["status"], "path_scoring")

    def test_product_run_loads_supplemental_documents_after_packet_review(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            generic = root / "generic_rules.txt"
            target = root / "urgent_warranty_email_chain.txt"
            generic.write_text("General background rules only.", encoding="utf-8")
            target.write_text("The key issue is warranty fraud discussed in the email chain.", encoding="utf-8")

            def fake_review(**kwargs: object) -> dict[str, object]:
                return {
                    "status": "used",
                    "sufficient": False,
                    "continue_retrieval": True,
                    "assessment": "Need the matter-specific email chain before drafting.",
                    "missing_information": ["matter-specific issue narrative"],
                    "revised_queries": ["warranty fraud email chain"],
                    "coverage_risks": ["generic rules are not enough"],
                }

            class FakeRouter:
                def __init__(self, config: object) -> None:
                    self.config = config

                def generate(self, **kwargs: object) -> SimpleNamespace:
                    module = str(kwargs.get("module") or "")
                    text = "MATERIAL_FACTS: warranty fraud is the key issue." if module == "extraction" else "The main issue is warranty fraud."
                    return SimpleNamespace(
                        text=text,
                        usage=ModelCallRecord(
                            module=module,
                            tier=ModelTier.CHEAP_WORKER if module == "extraction" else ModelTier.STRONG_SYNTHESIZER,
                            model="fake-model",
                            input_tokens=10,
                            output_tokens=5,
                            estimated_cost=0.001,
                        ),
                    )

            with patch("irys_harness.product.review_product_packet_with_cheap_worker", fake_review):
                with patch("irys_harness.product.GeminiModelRouter", FakeRouter):
                    result = run_product_matter(
                        objective="What is the main issue here?",
                        paths=[str(root)],
                        selected_paths=[str(generic)],
                        matter_id="Supplemental Matter",
                        config=load_config(),
                        trace_dir=root / "traces",
                        live_synthesis=True,
                        verbose=False,
                    )
            trace = result.state.to_trace()

            self.assertEqual(len(trace["documents"]), 2)
            self.assertIn(str(target.resolve()), trace["task"]["context_files"])
            self.assertIn(str(target.resolve()), trace["task"]["metadata"]["supplemental_context_files"])
            self.assertTrue(trace["retrieval_iterations"][-1]["supplemental_context_files"])
            support = "\n".join(item["raw_support"] for item in trace["final_packet"]["verified_evidence"])
            self.assertIn("warranty fraud", support)

    def test_run_product_matter_flags_no_matching_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "source.txt"
            doc.write_text("ordinary payment terms only", encoding="utf-8")

            result = run_product_matter(
                objective="zebra orbital indemnity",
                paths=[str(doc)],
                matter_id="No Evidence",
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            trace = result.state.to_trace()

            self.assertEqual(trace["diagnosis"]["status"], "needs_attention")
            self.assertEqual(trace["diagnosis"]["evidence_count"], 0)
            self.assertIn("No chunks matched the objective", trace["final_packet"]["unresolved"][0])

    def test_run_product_matter_does_not_truncate_local_documents_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = "UNTRUNCATED_MARKER_NOTICE_CURE"
            doc = root / "large_agreement.txt"
            doc.write_text(("A" * 210_000) + f" {marker}", encoding="utf-8")

            result = run_product_matter(
                objective=marker,
                paths=[str(doc)],
                matter_id="Large Matter",
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            trace = result.state.to_trace()

            self.assertGreater(trace["documents"][0]["text_chars"], 200_000)
            self.assertIn(marker, trace["rendered_answer"])

    def test_run_product_matter_pins_sources_into_synthesis_packet(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "loan_notice.txt"
            pinned = root / "board_minutes.txt"
            primary.write_text(
                "The borrower missed the May interest payment. The loan gives a five day cure period.",
                encoding="utf-8",
            )
            pinned_marker = "PINNED_CONTEXT_MARKER board approved enforcement of the guaranty."
            pinned.write_text(pinned_marker, encoding="utf-8")

            result = run_product_matter(
                objective="What cure period applies to the missed May payment?",
                paths=[str(root)],
                matter_id="Pinned Source",
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                selected_paths=[str(primary)],
                pinned_paths=[str(pinned)],
                verbose=False,
            )
            trace = result.state.to_trace()
            pinned_resolved = str(pinned.resolve())

            self.assertIn(pinned_resolved, trace["task"]["context_files"])
            self.assertEqual(trace["task"]["metadata"]["pinned_context_files"], [pinned_resolved])
            self.assertEqual(trace["diagnosis"]["pinned_source_count"], 1)
            pinned_sources = trace["final_packet"]["pinned_sources"]
            self.assertEqual(pinned_sources[0]["path"], pinned_resolved)
            self.assertIn(pinned_marker, pinned_sources[0]["excerpts"][0]["text"])
            self.assertIn(pinned_marker, build_product_synthesis_prompt(result.state))

    def test_sanitize_matter_id_is_path_safe(self) -> None:
        self.assertEqual(sanitize_matter_id("Client / Matter: 01"), "Client-Matter-01")

    def test_resolve_trace_path_stays_under_trace_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            trace_dir.mkdir()
            trace = trace_dir / "product_matter" / "matter.json"
            trace.parent.mkdir()
            trace.write_text("{}", encoding="utf-8")

            self.assertEqual(resolve_trace_path(str(trace), trace_dir), trace.resolve())
            with self.assertRaises(ValueError):
                resolve_trace_path(str(root / "outside.json"), trace_dir)

    def test_compact_trace_for_ui_keeps_browser_payload_bounded(self) -> None:
        trace = {
            "task": {"question": "Q"},
            "chunks": [{"doc_id": "doc_1", "chunk_id": "c1", "text": "x" * 20_000}],
            "final_packet": {"verified_evidence": [{"raw_support": "y" * 20_000}]},
            "extraction_records": [{"analysis": "z" * 20_000}],
        }

        compact = compact_trace_for_ui(trace)

        self.assertTrue(compact["_ui_compaction"]["compacted"])
        self.assertLess(len(compact["chunks"][0]["text_preview"]), 1000)
        self.assertIn("full text is saved in the trace file", compact["chunks"][0]["text_preview"])
        self.assertLess(len(compact["final_packet"]["verified_evidence"][0]["raw_support"]), 6000)

    def test_product_ui_renders_answer_markdown_online(self) -> None:
        self.assertIn('class="answer" id="answer"', INDEX_HTML)
        self.assertIn("function renderMarkdown", INDEX_HTML)
        self.assertNotIn('<pre id="answer"></pre>', INDEX_HTML)
        self.assertIn('id="chat"', INDEX_HTML)
        self.assertIn("conversation_history", INDEX_HTML)
        self.assertIn('id="chatHistory"', INDEX_HTML)
        self.assertIn("function renderChatHistory", INDEX_HTML)
        self.assertIn('id="traceList"', INDEX_HTML)
        self.assertIn("/api/traces", INDEX_HTML)
        self.assertIn('id="matterCost"', INDEX_HTML)
        self.assertIn('id="matterMessages"', INDEX_HTML)
        self.assertIn("function pathPayload", INDEX_HTML)
        self.assertIn("Paste local file or folder paths", INDEX_HTML)
        self.assertIn("No artificial file-count or per-document character cap", INDEX_HTML)
        self.assertIn("Workflow", INDEX_HTML)
        self.assertIn("Smart source planning reviews the file inventory", INDEX_HTML)
        self.assertIn("Draft final answer is on by default", INDEX_HTML)
        self.assertIn('id="live" type="checkbox" checked', INDEX_HTML)
        self.assertIn("Evidence passages controls how many retrieved source passages", INDEX_HTML)
        self.assertIn('id="topk" type="number" min="1" max="200" value="36"', INDEX_HTML)
        self.assertIn("default favors source coverage over minimal token use", INDEX_HTML)
        self.assertIn('id="usePlanner"', INDEX_HTML)
        self.assertIn("/api/pick-path", INDEX_HTML)
        self.assertIn("/api/plan", INDEX_HTML)
        self.assertIn("/api/rerun-plan", INDEX_HTML)
        self.assertIn("Review Source Plan", INDEX_HTML)
        self.assertIn("Plan ready. Review first-read documents, then click Run Approved Plan again.", INDEX_HTML)
        self.assertIn("Detailed Plan", INDEX_HTML)
        self.assertIn('id="planPreview"', INDEX_HTML)
        self.assertIn("command-bar", INDEX_HTML)
        self.assertIn('id="topRun"', INDEX_HTML)
        self.assertIn('id="topApplyNudge"', INDEX_HTML)
        self.assertIn('id="nextPassSummary"', INDEX_HTML)
        self.assertIn("control-dock", INDEX_HTML)
        self.assertIn('id="quickInstruction"', INDEX_HTML)
        self.assertIn("Correction Or Steering", INDEX_HTML)
        self.assertIn("Apply To Source Plan", INDEX_HTML)
        self.assertIn("Preview Next Pass", INDEX_HTML)
        self.assertIn("function updateQuickDock", INDEX_HTML)
        self.assertIn("function appendInstructionToTextarea", INDEX_HTML)
        self.assertIn("Load or finish a run before previewing a corrected next pass", INDEX_HTML)
        self.assertIn("Use the dock to steer the next pass", INDEX_HTML)
        self.assertIn('id="nextPassSetup"', INDEX_HTML)
        self.assertIn("Next Pass Setup", INDEX_HTML)
        self.assertIn("function renderNextPassSetup", INDEX_HTML)
        self.assertIn("function appendSteeringInstruction", INDEX_HTML)
        self.assertIn("steering note ready", INDEX_HTML)
        self.assertIn("function updateCommandStep", INDEX_HTML)
        self.assertIn("function syncCommandButtons", INDEX_HTML)
        self.assertIn("Run Brief", INDEX_HTML)
        self.assertIn('id="runBrief"', INDEX_HTML)
        self.assertIn("Recommended next action", INDEX_HTML)
        self.assertIn("Next Best Action", INDEX_HTML)
        self.assertIn('id="actionBoard"', INDEX_HTML)
        self.assertIn("function renderActionBoard", INDEX_HTML)
        self.assertIn("action-board-action", INDEX_HTML)
        self.assertIn("Question ready", INDEX_HTML)
        self.assertIn("Plan source focus", INDEX_HTML)
        self.assertIn("Current Work", INDEX_HTML)
        self.assertIn("Correction ready", INDEX_HTML)
        self.assertIn("Review Checklist", INDEX_HTML)
        self.assertIn('id="reviewChecklist"', INDEX_HTML)
        self.assertIn("Ready to rely on?", INDEX_HTML)
        self.assertIn("function renderReviewChecklist", INDEX_HTML)
        self.assertIn("function auditCard", INDEX_HTML)
        self.assertIn("Cheap worker share", INDEX_HTML)
        self.assertIn("Start with:", INDEX_HTML)
        self.assertIn('id="candidateReview"', INDEX_HTML)
        self.assertIn('id="applyCheckedCandidates"', INDEX_HTML)
        self.assertIn('id="selectAllCandidates"', INDEX_HTML)
        self.assertIn('id="restoreRecommendedCandidates"', INDEX_HTML)
        self.assertIn('id="firstReadPaths"', INDEX_HTML)
        self.assertIn('id="sourcePlanDetails"', INDEX_HTML)
        self.assertIn("function setSourcePlanOpen", INDEX_HTML)
        self.assertIn('id="planNote"', INDEX_HTML)
        self.assertIn('id="applyPlanCorrection"', INDEX_HTML)
        self.assertIn("Preview Corrected Plan", INDEX_HTML)
        self.assertIn("steering-panel", INDEX_HTML)
        self.assertIn("function renderPlan", INDEX_HTML)
        self.assertIn("function initialPlanPathKey", INDEX_HTML)
        self.assertIn("function rerunPlanPathKey", INDEX_HTML)
        self.assertIn("function requestRerunPlan", INDEX_HTML)
        self.assertIn("function rerunPlanNeedsRefresh", INDEX_HTML)
        self.assertIn('currentPlanMode !== "initial"', INDEX_HTML)
        self.assertIn("Steering plan ready. Review first-read documents, then click Run Corrected Pass again.", INDEX_HTML)
        self.assertIn("function renderRunBrief", INDEX_HTML)
        self.assertIn("function briefCard", INDEX_HTML)
        self.assertIn("function formatPlainText", INDEX_HTML)
        self.assertIn("function recommendedNextAction", INDEX_HTML)
        self.assertIn("function renderTracePlan", INDEX_HTML)
        self.assertIn("Answer target", INDEX_HTML)
        self.assertIn("Answer needs", INDEX_HTML)
        self.assertIn("function latestAnswerContract", INDEX_HTML)
        self.assertIn("function renderCandidateReview", INDEX_HTML)
        self.assertIn("function applyCheckedCandidatePaths", INDEX_HTML)
        self.assertIn("firstReadPathsDirty", INDEX_HTML)
        self.assertIn("selected_paths_locked", INDEX_HTML)
        self.assertIn("function setFirstReadPaths", INDEX_HTML)
        self.assertIn("re-plans which files to read", INDEX_HTML)
        self.assertIn("function formatPlannerSummary", INDEX_HTML)
        self.assertIn("Applying your steering note.", INDEX_HTML)
        self.assertIn("Your instruction:", INDEX_HTML)
        self.assertIn("Source plan:", INDEX_HTML)
        self.assertIn("Error:", INDEX_HTML)
        self.assertIn("Retrieval decision:", INDEX_HTML)
        self.assertIn("Missing information:", INDEX_HTML)
        self.assertIn("Revised search targets:", INDEX_HTML)
        self.assertIn("Coverage risks:", INDEX_HTML)
        self.assertIn("Source coverage:", INDEX_HTML)
        self.assertIn("function formatSourceCoverageSummary", INDEX_HTML)
        self.assertIn("Coverage warning:", INDEX_HTML)
        self.assertIn("function sourceCoverageWarning", INDEX_HTML)
        self.assertIn("Reading first:", INDEX_HTML)
        self.assertIn("Pinned documents:", INDEX_HTML)
        self.assertIn("Held back for now:", INDEX_HTML)
        self.assertIn("function formatSourceSelectionMode", INDEX_HTML)
        self.assertIn("re-planning from your steering note", INDEX_HTML)
        self.assertIn("use_llm_planning", INDEX_HTML)
        self.assertIn("candidate-check", INDEX_HTML)
        self.assertIn('id="chooseFolder"', INDEX_HTML)
        self.assertIn('id="chooseFiles"', INDEX_HTML)
        self.assertIn("function appendPaths", INDEX_HTML)
        self.assertIn('id="liveEvents"', INDEX_HTML)
        self.assertIn("/api/run-async", INDEX_HTML)
        self.assertIn("/api/rerun-async", INDEX_HTML)
        self.assertIn('id="previewNudgePlan"', INDEX_HTML)
        self.assertIn("Preview Steering Plan", INDEX_HTML)
        self.assertIn("Run Corrected Pass", INDEX_HTML)
        self.assertIn("/api/run-status", INDEX_HTML)
        self.assertIn("/api/cancel-run", INDEX_HTML)
        self.assertIn("function pollRunJob", INDEX_HTML)
        self.assertIn("What Irys Is Doing", INDEX_HTML)
        self.assertIn("function renderRecentLiveEvents", INDEX_HTML)
        self.assertIn("earlier update(s) are saved in the trace", INDEX_HTML)
        self.assertIn('id="currentStep"', INDEX_HTML)
        self.assertIn('id="runTimeline"', INDEX_HTML)
        self.assertIn("function renderRunTimeline", INDEX_HTML)
        self.assertIn("function isRunCompleteEvent", INDEX_HTML)
        self.assertIn("Review the answer, sources, and trace.", INDEX_HTML)
        self.assertIn("timeline-step", INDEX_HTML)
        self.assertIn("Select sources", INDEX_HTML)
        self.assertIn("Draft answer", INDEX_HTML)
        self.assertIn('id="stopRun"', INDEX_HTML)
        self.assertIn("Source Review", INDEX_HTML)
        self.assertIn("Documents Held Back", INDEX_HTML)
        self.assertIn("ignore-source-next", INDEX_HTML)
        self.assertIn("Ignore next pass", INDEX_HTML)
        self.assertIn("deeper-source-next", INDEX_HTML)
        self.assertIn("Read deeper next pass", INDEX_HTML)
        self.assertIn("pin-source-next", INDEX_HTML)
        self.assertIn("Pin to synthesis", INDEX_HTML)
        self.assertIn("Pinned For Next Pass", INDEX_HTML)
        self.assertIn('id="pinnedSources"', INDEX_HTML)
        self.assertIn("pinned_paths", INDEX_HTML)
        self.assertIn("pinnedSourcePaths", INDEX_HTML)
        self.assertIn("function addPinnedSourcePath", INDEX_HTML)
        self.assertIn("function renderPinnedSources", INDEX_HTML)
        self.assertIn("unpin-source-next", INDEX_HTML)
        self.assertIn("Pinned sources", INDEX_HTML)
        self.assertIn("function sourceCard", INDEX_HTML)
        self.assertIn("Reviewer kept", INDEX_HTML)
        self.assertIn("Reviewer marked low value", INDEX_HTML)
        self.assertIn("function renderReviewerSourceActions", INDEX_HTML)
        self.assertIn("function sourceIdsToDocuments", INDEX_HTML)
        self.assertIn("Next pass changes queued", INDEX_HTML)
        self.assertIn("function updateCommandForQueuedNextPass", INDEX_HTML)
        self.assertIn("function announceNextPassQueued", INDEX_HTML)
        self.assertIn("function removePathFromFirstRead", INDEX_HTML)
        self.assertIn("excludedSourcePaths", INDEX_HTML)
        self.assertIn("excluded_paths", INDEX_HTML)
        self.assertIn("function addExcludedSourcePath", INDEX_HTML)
        self.assertIn("function removeExcludedSourcePath", INDEX_HTML)
        self.assertIn("unignore-source-next", INDEX_HTML)
        self.assertIn("Updated first-read set", INDEX_HTML)
        self.assertIn("if (!firstReadPathsDirty) applyCheckedCandidatePaths", INDEX_HTML)
        self.assertIn("Question Or Work Product", INDEX_HTML)
        self.assertIn(".objective.compact", INDEX_HTML)
        self.assertIn('classList.add("compact")', INDEX_HTML)
        self.assertIn('classList.remove("compact")', INDEX_HTML)
        self.assertIn("For a follow-up, keep the same Matter ID and Chat ID", INDEX_HTML)
        self.assertIn("Source Passages", INDEX_HTML)
        self.assertIn("Current Run Cost", INDEX_HTML)
        self.assertIn("Matter Runs", INDEX_HTML)
        self.assertIn("Conversation History", INDEX_HTML)
        self.assertIn("Loaded saved run:", INDEX_HTML)
        self.assertIn("Searchable passages:", INDEX_HTML)
        self.assertIn("Passages from this document:", INDEX_HTML)
        self.assertIn("Extract text and source passages", INDEX_HTML)
        self.assertIn('id="heldBackSources"', INDEX_HTML)
        self.assertIn("function renderHeldBackSources", INDEX_HTML)
        self.assertIn("function renderLimitedCards", INDEX_HTML)
        self.assertIn("function limitDisplayText", INDEX_HTML)
        self.assertIn("function compactJsonForDisplay", INDEX_HTML)
        self.assertIn("function formatInlineList", INDEX_HTML)
        self.assertIn("saved in the trace but not rendered here", INDEX_HTML)
        self.assertIn("more characters are saved in the trace but not rendered here", INDEX_HTML)
        self.assertIn("include-held-back", INDEX_HTML)
        self.assertIn("Read next pass", INDEX_HTML)
        self.assertIn("in the next pass", INDEX_HTML)
        self.assertIn("next first-read set", INDEX_HTML)
        self.assertIn("function filenameFromPath", INDEX_HTML)
        self.assertIn("Tokens by tier", INDEX_HTML)
        self.assertIn("Cost by tier", INDEX_HTML)
        self.assertIn("function formatTierMetric", INDEX_HTML)
        self.assertIn("Documents added", INDEX_HTML)
        self.assertIn("Documents removed", INDEX_HTML)
        self.assertIn("New evidence", INDEX_HTML)
        self.assertIn("Evidence no longer used", INDEX_HTML)
        self.assertIn("Cleared open questions", INDEX_HTML)
        self.assertIn("Plain-English takeaway", INDEX_HTML)
        self.assertIn("function comparisonTakeaway", INDEX_HTML)
        self.assertIn("The answer changed. Review the changed answer against Sources Used before accepting it.", INDEX_HTML)
        self.assertIn("function formatSimpleList", INDEX_HTML)
        self.assertIn("function formatEvidenceDelta", INDEX_HTML)
        self.assertIn("Advanced diagnostic data", INDEX_HTML)
        self.assertIn("Advanced run details", INDEX_HTML)
        self.assertNotIn("webkitdirectory", INDEX_HTML)
        self.assertNotIn("function uploadFiles", INDEX_HTML)

    def test_product_run_cli_uses_product_retrieval_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["product-run", "--objective", "What is the issue?", "--path", "matter"])

        self.assertEqual(args.top_k, DEFAULT_PRODUCT_TOP_K)

    def test_pick_local_paths_rejects_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            pick_local_paths(mode="invalid")

    def test_pick_path_endpoint_returns_selected_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            output_dir = root / "outputs"
            handler = build_handler(
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch(
                    "irys_harness.product_ui.pick_local_paths",
                    return_value=[str(root / "selected")],
                ):
                    request = Request(
                        f"http://127.0.0.1:{server.server_port}/api/pick-path",
                        data=json.dumps({"mode": "folder"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=5) as response:  # noqa: S310 - local test server.
                        data = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(data["paths"], [str(root / "selected")])

    def test_async_run_endpoint_streams_live_events_until_completion(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            trace_dir = root / "traces"
            output_dir = root / "outputs"
            handler = build_handler(
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/run-async",
                    data=json.dumps(
                        {
                            "matter_id": "Live Matter",
                            "paths": [str(doc)],
                            "objective": "What cure period applies?",
                            "live_synthesis": False,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:  # noqa: S310 - local test server.
                    started = json.loads(response.read().decode("utf-8"))

                status = started
                for _ in range(20):
                    with urlopen(  # noqa: S310 - local test server.
                        f"http://127.0.0.1:{server.server_port}/api/run-status?job_id={started['job_id']}",
                        timeout=5,
                    ) as response:
                        status = json.loads(response.read().decode("utf-8"))
                    if status["status"] != "running":
                        break
                    time.sleep(0.05)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(status["status"], "completed")
            self.assertIn("5 day cure period", status["result"]["rendered_answer"])
            labels = [event["label"] for event in status["events"]]
            self.assertIn("SCOPE", labels)
            self.assertIn("READ", labels)
            self.assertIn("LOAD", labels)
            self.assertIn("SEARCH", labels)
            self.assertIn("SAVE", labels)
            summaries = [event.get("fields", {}).get("summary", "") for event in status["events"]]
            self.assertTrue(any("Reading document" in summary for summary in summaries))

    def test_async_run_endpoint_streams_readable_failure_event(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            unsupported = root / "archive.bin"
            unsupported.write_bytes(b"not a supported matter file")
            trace_dir = root / "traces"
            output_dir = root / "outputs"
            handler = build_handler(
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/run-async",
                    data=json.dumps(
                        {
                            "matter_id": "Bad Matter",
                            "paths": [str(unsupported)],
                            "objective": "What does this say?",
                            "live_synthesis": False,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:  # noqa: S310 - local test server.
                    started = json.loads(response.read().decode("utf-8"))

                status = started
                for _ in range(20):
                    with urlopen(  # noqa: S310 - local test server.
                        f"http://127.0.0.1:{server.server_port}/api/run-status?job_id={started['job_id']}",
                        timeout=5,
                    ) as response:
                        status = json.loads(response.read().decode("utf-8"))
                    if status["status"] != "running":
                        break
                    time.sleep(0.05)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(status["status"], "failed")
            error_events = [event for event in status["events"] if event["label"] == "ERROR"]
            self.assertTrue(error_events)
            self.assertEqual(error_events[-1]["fields"]["summary"], "The run failed before completion.")
            self.assertIn("at least one supported document path is required", error_events[-1]["fields"]["error"])

    def test_list_product_traces_filters_by_matter_and_chat(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            trace_dir = root / "traces"
            run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Matter A",
                chat_id="main",
                config=load_config(),
                trace_dir=trace_dir,
                live_synthesis=False,
                verbose=False,
            )
            run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Matter A",
                chat_id="side",
                config=load_config(),
                trace_dir=trace_dir,
                live_synthesis=False,
                verbose=False,
            )

            all_rows = list_product_traces(trace_dir, matter_id="Matter A")
            side_rows = list_product_traces(trace_dir, matter_id="Matter A", chat_id="side")

            self.assertEqual(len(all_rows), 2)
            self.assertEqual(len(side_rows), 1)
            self.assertEqual(side_rows[0]["chat_id"], "side")
            self.assertTrue(Path(side_rows[0]["path"]).exists())

    def test_summarize_trace_rows_reports_matter_cost(self) -> None:
        summary = summarize_trace_rows(
            [
                {"chat_id": "main", "estimated_cost": 0.25, "total_tokens": 10},
                {"chat_id": "side", "estimated_cost": 0.75, "total_tokens": 30},
            ]
        )

        self.assertEqual(summary["messages"], 2)
        self.assertEqual(summary["chats"], 2)
        self.assertEqual(summary["total_tokens"], 40)
        self.assertAlmostEqual(summary["estimated_cost"], 1.0)
        self.assertAlmostEqual(summary["average_cost_per_message"], 0.5)

    def test_rerun_from_trace_links_parent_and_nudge(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            trace_dir = root / "traces"
            parent = run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Parent Matter",
                config=load_config(),
                trace_dir=trace_dir,
                live_synthesis=False,
                verbose=False,
            )

            response = rerun_from_trace(
                {
                    "trace_path": str(parent.trace_path),
                    "nudge": "Focus only on payment defaults.",
                    "live_synthesis": False,
                    "top_k": 3,
                },
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=root / "outputs",
            )
            trace = response["trace"]

            self.assertTrue(response["trace_path"])
            self.assertIn("User steering note: Focus only on payment defaults.", trace["task"]["question"])
            self.assertEqual(trace["task"]["metadata"]["parent_trace_path"], str(parent.trace_path.resolve()))
            self.assertEqual(trace["task"]["metadata"]["user_nudge"], "Focus only on payment defaults.")
            self.assertEqual(response["comparison"]["parent_task_id"], "Parent-Matter")
            self.assertTrue(response["comparison"]["objective_changed"])
            history = trace["final_packet"]["conversation_history"]
            self.assertEqual(set(history[-1]), {"user", "assistant"})
            self.assertIn("What cure period applies?", history[-1]["user"])
            self.assertIn("5 day cure period", history[-1]["assistant"])

    def test_rerun_from_trace_can_add_local_path_corpus(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            guaranty = root / "guaranty.txt"
            guaranty.write_text(
                "The guarantor cure period is 2 business days after written notice.",
                encoding="utf-8",
            )
            trace_dir = root / "traces"
            output_dir = root / "outputs"
            parent = run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Corpus Matter",
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
                live_synthesis=False,
                verbose=False,
            )

            response = rerun_from_trace(
                {
                    "trace_path": str(parent.trace_path),
                    "nudge": "Also consider the guaranty.",
                    "paths": [str(guaranty)],
                },
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
            )
            trace = response["trace"]

            self.assertEqual(len(trace["task"]["context_files"]), 2)
            self.assertEqual(response["comparison"]["document_delta"]["kept_count"], 1)
            self.assertEqual(len(response["comparison"]["document_delta"]["added"]), 1)

    def test_parse_paths_accepts_textarea_or_list(self) -> None:
        self.assertEqual(parse_paths(" a.txt \n\n b.txt "), ["a.txt", "b.txt"])
        self.assertEqual(parse_paths([" a.txt ", ""]), ["a.txt"])

    def test_rerun_selected_paths_require_explicit_lock(self) -> None:
        payload = {
            "selected_paths": ["old-first-read.txt"],
            "nudge": "Focus on the later 10-K instead.",
        }

        self.assertIsNone(selected_paths_for_rerun_payload(payload))

        payload["selected_paths_locked"] = True
        self.assertEqual(selected_paths_for_rerun_payload(payload), ["old-first-read.txt"])

    def test_rerun_context_paths_honor_excluded_paths(self) -> None:
        parent = {
            "task": {
                "context_files": ["old-source.txt", "keep-source.txt"],
                "metadata": {"discovered_context_files": ["old-source.txt", "keep-source.txt"]},
            }
        }
        payload = {
            "paths": ["new-source.txt"],
            "excluded_paths": ["old-source.txt"],
            "selected_paths": ["old-source.txt", "keep-source.txt"],
            "selected_paths_locked": True,
        }

        self.assertEqual(rerun_context_paths(parent, payload), ["keep-source.txt", "new-source.txt"])
        self.assertEqual(selected_paths_for_rerun_payload(payload), ["keep-source.txt"])

    def test_rerun_plan_note_keeps_current_steering_note(self) -> None:
        note = build_rerun_plan_note("Prior correction: ignore drafts.", "Focus on the 2024 10-K.")

        self.assertIn("Prior correction: ignore drafts.", note)
        self.assertIn("Current steering note: Focus on the 2024 10-K.", note)

    def test_rerun_plan_from_trace_previews_nudged_source_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            email = root / "email.txt"
            ten_k = root / "2024_10-K.txt"
            email.write_text("Email discusses a renewal dispute.", encoding="utf-8")
            ten_k.write_text("Annual report: diluted EPS was 1.23.", encoding="utf-8")
            trace_dir = root / "traces"
            output_dir = root / "outputs"
            parent = run_product_matter(
                objective="What is the main issue here?",
                paths=[str(email)],
                matter_id="Preview Matter",
                config=load_config(),
                trace_dir=trace_dir,
                output_dir=output_dir,
                live_synthesis=False,
                verbose=False,
            )

            plan = rerun_plan_from_trace(
                {
                    "trace_path": str(parent.trace_path),
                    "nudge": "Focus on the 2024 10-K EPS instead.",
                    "paths": [str(ten_k)],
                    "top_k": 3,
                },
                config=load_config(),
                trace_dir=trace_dir,
            )

            self.assertIn("User steering note: Focus on the 2024 10-K EPS instead.", plan["objective"])
            self.assertIn("Current steering note: Focus on the 2024 10-K EPS instead.", plan["plan_note"])
            self.assertIn(str(ten_k.resolve()), plan["first_read_paths"])

    def test_conversation_history_is_synthesis_only_not_retrieval_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            result = run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Chat Matter",
                chat_id="follow-up",
                conversation_history=[
                    {
                        "user": "Prior question with NEVER_RETRIEVE_HISTORY_MARKER.",
                        "assistant": "Prior final answer only.",
                        "intermediate": "must not survive",
                    }
                ],
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            trace = result.state.to_trace()
            queries = "\n".join(trace["retrieval_iterations"][0]["queries"])
            history = trace["final_packet"]["conversation_history"]
            prompt = build_product_synthesis_prompt(result.state)

            self.assertEqual(trace["task"]["metadata"]["chat_id"], "follow-up")
            self.assertIn("--chat-follow-up", trace["task_id"])
            self.assertEqual(history, [{"user": "Prior question with NEVER_RETRIEVE_HISTORY_MARKER.", "assistant": "Prior final answer only."}])
            self.assertNotIn("NEVER_RETRIEVE_HISTORY_MARKER", queries)
            self.assertFalse(trace["retrieval_iterations"][0]["conversation_history_used"])
            self.assertIn("NEVER_RETRIEVE_HISTORY_MARKER", prompt)

    def test_history_dependent_followup_is_flagged_without_using_history_for_retrieval(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Guaranty enforcement requires a continuing payment default.", encoding="utf-8")
            result = run_product_matter(
                objective="What about that?",
                paths=[str(doc)],
                matter_id="Followup Matter",
                conversation_history=[{"user": "Can lender accelerate?", "assistant": "Only after default."}],
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            trace = result.state.to_trace()

            self.assertIn("may depend on prior chat context", trace["final_packet"]["unresolved"][0])
            self.assertFalse(trace["retrieval_iterations"][0]["conversation_history_used"])

    def test_product_synthesis_prompt_includes_worker_analysis(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "agreement.txt"
            doc.write_text("Payment default has a 5 day cure period after notice.", encoding="utf-8")
            result = run_product_matter(
                objective="What cure period applies?",
                paths=[str(doc)],
                matter_id="Analysis Matter",
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )
            result.state.final_packet["worker_analysis"] = "MATERIAL_FACTS: payment default cure is 5 days."

            prompt = build_product_synthesis_prompt(result.state)

            self.assertIn("Worker analysis:", prompt)
            self.assertIn("MATERIAL_FACTS: payment default cure is 5 days.", prompt)
            self.assertIn("preserve row labels, column labels, periods, classes, and basic/diluted distinctions", prompt)
            self.assertIn("Do not label a three-month, quarterly, Q4, or interim value as fiscal-year or annual", prompt)

    def test_product_worker_analysis_prompt_preserves_numeric_periods(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "earnings.txt"
            doc.write_text("Fourth quarter diluted EPS was $0.13. Fiscal year diluted EPS was $0.52.", encoding="utf-8")
            result = run_product_matter(
                objective="Tell me about EPS in 2024.",
                paths=[str(doc)],
                matter_id="Metric Period Matter",
                config=load_config(),
                trace_dir=root / "traces",
                live_synthesis=False,
                verbose=False,
            )

            prompt = build_product_worker_analysis_prompt(result.state)

            self.assertIn("preserve the source period exactly", prompt)
            self.assertIn("Do not label a three-month, quarterly, Q4, or interim value as fiscal-year or annual", prompt)

    def test_metric_selection_notes_flag_intermediate_reconciliation_rows(self) -> None:
        evidence = [
            {
                "raw_support": (
                    "Non-GAAP net income before non-GAAP tax adjustments. "
                    "Net income per share before non-GAAP tax adjustments - diluted $2.25. "
                    "Net income per share after non-GAAP\n| tax adjustments - basic $1.94. "
                    "Net income per share after non-GAAP\n| tax adjustments - diluted $1.82."
                )
            }
        ]

        notes = build_metric_selection_notes(evidence)

        self.assertGreaterEqual(len(notes), 2)
        self.assertIn("Before-adjustment rows are intermediate reconciliation rows", notes[0])
        self.assertIn("$1.82", "\n".join(notes))

    def test_compact_relevant_snippet_centers_final_adjusted_metric_rows(self) -> None:
        text = (
            "Revenue was strong. " + ("filler " * 80)
            + "Net income per share before non-GAAP tax adjustments - basic $2.40. "
            + "Net income per share before non-GAAP tax adjustments - diluted $2.25. "
            + "Net income per share after non-GAAP tax adjustments - basic $1.94. "
            + "Net income per share after non-GAAP tax adjustments - diluted $1.82. "
            + ("tail " * 80)
        )

        snippet = compact_relevant_snippet(text, queries=["tell me about EPS in 2024"], max_chars=220)

        self.assertIn("$1.82", snippet)
        self.assertIn("after non-GAAP tax adjustments", snippet)

    def test_build_answer_source_map_links_answer_sections_to_evidence(self) -> None:
        evidence = [
            {
                "raw_support": "The payment default cure period is 5 business days after notice.",
                "source": {"doc_id": "doc_0001", "chunk_id": "doc_0001_chunk_0000"},
            }
        ]
        answer = "Payment default has a 5 business day cure period (doc_0001_chunk_0000)."

        source_map = build_answer_source_map(answer, evidence)

        self.assertEqual(len(source_map), 1)
        self.assertEqual(source_map[0]["source_refs"], ["doc_0001_chunk_0000"])
        self.assertEqual(source_map[0]["support"][0]["doc_id"], "doc_0001")

    def test_compare_product_traces_reports_evidence_and_metric_deltas(self) -> None:
        parent = {
            "run_id": "parent",
            "task_id": "matter",
            "task": {"question": "Question"},
            "documents": [{"path": "a.txt"}],
            "final_packet": {
                "verified_evidence": [
                    {
                        "claim": "Old claim",
                        "raw_support": "Old support",
                        "source": {"doc_id": "doc_1", "chunk_id": "chunk_1"},
                    }
                ],
                "unresolved": ["old gap"],
            },
            "rendered_answer": "Old answer",
            "metrics": {"total_tokens": 10, "estimated_cost": 0.25},
        }
        child = {
            "run_id": "child",
            "task_id": "matter-nudge",
            "task": {"question": "Question\n\nUser steering note: focus"},
            "documents": [{"path": "a.txt"}, {"path": "b.txt"}],
            "final_packet": {
                "verified_evidence": [
                    {
                        "claim": "New claim",
                        "raw_support": "New support",
                        "source": {"doc_id": "doc_2", "chunk_id": "chunk_2"},
                    }
                ],
                "unresolved": ["new gap"],
            },
            "rendered_answer": "New answer",
            "metrics": {"total_tokens": 17, "estimated_cost": 0.5},
        }

        comparison = compare_product_traces(parent, child)

        self.assertTrue(comparison["objective_changed"])
        self.assertTrue(comparison["answer_changed"])
        self.assertEqual(comparison["document_delta"]["added"], ["b.txt"])
        self.assertEqual(comparison["evidence_delta"]["added_count"], 1)
        self.assertEqual(comparison["evidence_delta"]["removed_count"], 1)
        self.assertEqual(comparison["unresolved_delta"]["added"], ["new gap"])
        self.assertEqual(comparison["metrics_delta"]["total_tokens"], 7)
        self.assertAlmostEqual(comparison["metrics_delta"]["estimated_cost"], 0.25)


if __name__ == "__main__":
    unittest.main()
