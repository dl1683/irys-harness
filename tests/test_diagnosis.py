from __future__ import annotations

import unittest

from irys_harness.diagnosis import diagnose_harvey_scores


class DiagnosisTests(unittest.TestCase):
    def test_diagnose_harvey_scores_classifies_missing_items(self) -> None:
        diagnosis = diagnose_harvey_scores(
            {
                "task": "area/task",
                "all_pass": False,
                "n_passed": 1,
                "n_criteria": 2,
                "criteria_results": [
                    {
                        "id": "C-1",
                        "title": "Identifies missing exhibit",
                        "verdict": "fail",
                        "reasoning": "The memo does not mention the missing exhibit.",
                    }
                ],
            }
        )
        self.assertIn("retrieval_miss", diagnosis["failure_tags"])
        self.assertEqual(diagnosis["suspected_module"], "retriever_extractor")
        self.assertEqual(diagnosis["suspected_actor"], "cheap_worker")

    def test_diagnose_harvey_scores_classifies_final_enumeration_drop(self) -> None:
        diagnosis = diagnose_harvey_scores(
            {
                "task": "area/task",
                "all_pass": False,
                "n_passed": 26,
                "n_criteria": 27,
                "criteria_results": [
                    {
                        "id": "C-25",
                        "title": "Correctly identifies expert letter writers whose letters are present",
                        "verdict": "fail",
                        "reasoning": (
                            "The memo fails to identify the specific expert letter writers present. "
                            "It does not list Johansson, Tsai, Al-Rashidi, and Moriarty."
                        ),
                    }
                ],
            }
        )
        self.assertIn("synthesis_error", diagnosis["failure_tags"])
        self.assertIn("context_packing_error", diagnosis["failure_tags"])
        self.assertEqual(diagnosis["suspected_module"], "final_packet_synthesizer")
        self.assertEqual(diagnosis["suspected_actor"], "strong_synthesizer")

    def test_diagnose_harvey_scores_classifies_severity_calibration(self) -> None:
        diagnosis = diagnose_harvey_scores(
            {
                "task": "area/task",
                "all_pass": False,
                "n_passed": 42,
                "n_criteria": 43,
                "criteria_results": [
                    {
                        "id": "C-7",
                        "title": "Rates preferred stock authorization mismatch as High severity",
                        "verdict": "fail",
                        "reasoning": "The agent rated the issue as Medium severity, whereas High severity was required.",
                    }
                ],
            }
        )
        self.assertIn("severity_calibration_error", diagnosis["failure_tags"])
        self.assertEqual(diagnosis["suspected_module"], "severity_calibrator")
        self.assertEqual(diagnosis["suspected_actor"], "cheap_worker")

    def test_diagnose_harvey_scores_classifies_wrong_computation(self) -> None:
        diagnosis = diagnose_harvey_scores(
            {
                "task": "area/task",
                "all_pass": False,
                "n_passed": 19,
                "n_criteria": 36,
                "criteria_results": [
                    {
                        "id": "C-12",
                        "title": "Calculates corrected leverage ratio",
                        "verdict": "fail",
                        "reasoning": "The agent calculated a corrected leverage ratio of 4.70x, not 5.54x.",
                    }
                ],
            }
        )
        self.assertIn("wrong_computation", diagnosis["failure_tags"])
        self.assertEqual(diagnosis["suspected_module"], "calculator")
        self.assertEqual(diagnosis["suspected_actor"], "cheap_worker")

    def test_diagnose_harvey_scores_classifies_request_list_coverage_drop(self) -> None:
        diagnosis = diagnose_harvey_scores(
            {
                "task": "area/task",
                "all_pass": False,
                "n_passed": 41,
                "n_criteria": 69,
                "criteria_results": [
                    {
                        "id": "C-24",
                        "title": "Requests all SOC 2 reports",
                        "verdict": "fail",
                        "reasoning": (
                            "The agent's output does not include a specific request for all "
                            "SOC 2 Type II reports in its due diligence request list."
                        ),
                    },
                    {
                        "id": "C-63",
                        "title": "Addresses Atlanta HQ lease assignment consent requirement",
                        "verdict": "fail",
                        "reasoning": (
                            "The output fails to include a specific request for landlord "
                            "consent even though the lease threshold is material."
                        ),
                    },
                ],
            }
        )
        self.assertIn("synthesis_error", diagnosis["failure_tags"])
        self.assertIn("context_packing_error", diagnosis["failure_tags"])
        self.assertEqual(diagnosis["suspected_module"], "final_packet_synthesizer")
        self.assertEqual(diagnosis["suspected_actor"], "strong_synthesizer")

    def test_diagnose_harvey_scores_prioritizes_artifact_coverage_over_numeric_wording(self) -> None:
        diagnosis = diagnose_harvey_scores(
            {
                "task": "area/task",
                "all_pass": False,
                "n_passed": 24,
                "n_criteria": 48,
                "criteria_results": [
                    {
                        "id": "C-19",
                        "title": "Identifies revolving credit facility exceeds debt threshold",
                        "verdict": "fail",
                        "reasoning": (
                            "The agent's output does not address the debt-threshold issue "
                            "in the governance issues report."
                        ),
                    }
                ],
            }
        )
        self.assertIn("context_packing_error", diagnosis["failure_tags"])
        self.assertEqual(diagnosis["suspected_module"], "final_packet_synthesizer")
        self.assertEqual(diagnosis["suspected_actor"], "strong_synthesizer")

    def test_diagnose_harvey_scores_classifies_transaction_coverage_drop(self) -> None:
        diagnosis = diagnose_harvey_scores(
            {
                "task": "area/task",
                "all_pass": False,
                "n_passed": 16,
                "n_criteria": 66,
                "criteria_results": [
                    {
                        "id": "C-8",
                        "title": "Charlotte MSA combined share correctly calculated",
                        "verdict": "fail",
                        "reasoning": (
                            "The agent's output focuses exclusively on the Pinnacle transaction. "
                            "It does not contain any analysis of the LabVantage/Prism transaction "
                            "from the source documents."
                        ),
                    }
                ],
            }
        )
        self.assertIn("distractor_confusion", diagnosis["failure_tags"])
        self.assertIn("context_packing_error", diagnosis["failure_tags"])
        self.assertEqual(diagnosis["suspected_module"], "final_packet_synthesizer")
        self.assertEqual(diagnosis["suspected_actor"], "strong_synthesizer")


if __name__ == "__main__":
    unittest.main()
