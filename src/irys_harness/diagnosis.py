from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .failures import FailureTag


def diagnose_harvey_scores(scores: dict[str, Any]) -> dict[str, Any]:
    criteria = scores.get("criteria_results", [])
    failed = [item for item in criteria if item.get("verdict") == "fail"]
    tags = classify_failed_criteria(failed)
    return {
        "task_id": scores.get("task"),
        "failed": not bool(scores.get("all_pass")),
        "score": scores.get("score"),
        "n_passed": scores.get("n_passed"),
        "n_criteria": scores.get("n_criteria"),
        "rubric_pass_rate": (scores.get("n_passed") / scores.get("n_criteria"))
        if scores.get("n_criteria")
        else None,
        "failure_tags": sorted(tag.value for tag in tags),
        "failed_criteria": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "reasoning": item.get("reasoning"),
            }
            for item in failed
        ],
        "suspected_module": suspected_module(tags),
        "suspected_actor": suspected_actor(tags),
        "recommended_experiment": recommended_experiment(tags),
    }


def diagnose_harvey_scores_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return diagnose_harvey_scores(json.load(handle))


def classify_failed_criteria(failed: list[dict[str, Any]]) -> set[FailureTag]:
    tags: set[FailureTag] = set()
    for item in failed:
        text = f"{item.get('title', '')} {item.get('reasoning', '')}".lower()
        if any(
            phrase in text
            for phrase in [
                "different transaction",
                "focuses exclusively on",
                "focuses on the",
                "does not contain any analysis of",
                "does not discuss the",
                "does not mention the",
                "not relevant to the primary task",
            ]
        ) and any(
            term in text
            for term in [
                "transaction",
                "deal",
                "entity",
                "entities",
                "parties",
                "source documents",
                "work product",
            ]
        ):
            tags.add(FailureTag.DISTRACTOR_CONFUSION)
            tags.add(FailureTag.CONTEXT_PACKING_ERROR)
            tags.add(FailureTag.SYNTHESIS_ERROR)
            continue
        if any(
            phrase in text
            for phrase in [
                "fails to identify the specific",
                "does not list",
                "does not acknowledge",
                "does not explicitly list",
            ]
        ):
            tags.add(FailureTag.SYNTHESIS_ERROR)
            tags.add(FailureTag.CONTEXT_PACKING_ERROR)
            continue
        if "rated" in text and "severity" in text:
            tags.add(FailureTag.SEVERITY_CALIBRATION_ERROR)
            continue
        if any(
            phrase in text
            for phrase in [
                "calculates",
                "calculated",
                "calculation",
                "fails to provide the specific mathematical",
                "fails to perform",
                "wrong denominator",
                "ratio",
                "cap ",
                "threshold",
            ]
        ):
            tags.add(FailureTag.WRONG_COMPUTATION)
        if any(word in text for word in ["missing", "fails to identify", "does not mention", "failed to identify"]):
            tags.add(FailureTag.RETRIEVAL_MISS)
            tags.add(FailureTag.BAD_EXTRACTION)
        if any(word in text for word in ["incorrectly states", "inconsistent", "wrong"]):
            tags.add(FailureTag.UNSUPPORTED_INFERENCE)
        if "summary table" in text or "tabular" in text:
            tags.add(FailureTag.FORMAT_ERROR)
        if "risk of misrepresentation" in text or "credibility" in text:
            tags.add(FailureTag.SYNTHESIS_ERROR)
    if not tags and failed:
        tags.add(FailureTag.UNCLASSIFIED_FAILURE)
    return tags


def suspected_module(tags: set[FailureTag]) -> str | None:
    if FailureTag.DISTRACTOR_CONFUSION in tags:
        return "final_packet_synthesizer"
    if FailureTag.WRONG_COMPUTATION in tags:
        return "calculator"
    if FailureTag.SEVERITY_CALIBRATION_ERROR in tags:
        return "severity_calibrator"
    if FailureTag.CONTEXT_PACKING_ERROR in tags or FailureTag.SYNTHESIS_ERROR in tags:
        return "final_packet_synthesizer"
    if FailureTag.RETRIEVAL_MISS in tags or FailureTag.BAD_EXTRACTION in tags:
        return "retriever_extractor"
    if FailureTag.FORMAT_ERROR in tags:
        return "renderer"
    return None


def suspected_actor(tags: set[FailureTag]) -> str | None:
    if FailureTag.DISTRACTOR_CONFUSION in tags:
        return "strong_synthesizer"
    if FailureTag.WRONG_COMPUTATION in tags:
        return "cheap_worker"
    if FailureTag.SEVERITY_CALIBRATION_ERROR in tags:
        return "cheap_worker"
    if FailureTag.CONTEXT_PACKING_ERROR in tags or FailureTag.SYNTHESIS_ERROR in tags:
        return "strong_synthesizer"
    if FailureTag.RETRIEVAL_MISS in tags or FailureTag.BAD_EXTRACTION in tags:
        return "cheap_worker"
    if FailureTag.FORMAT_ERROR in tags:
        return "renderer"
    return None


def recommended_experiment(tags: set[FailureTag]) -> dict[str, Any] | None:
    module = suspected_module(tags)
    if module is None:
        return None
    if module == "retriever_extractor":
        return {
            "change_type": "retrieval",
            "target": module,
            "hypothesis": "Add issue-oriented query expansion and increase recall before cheap-worker extraction.",
            "validation": ["same_task", "same_practice_area_smoke"],
        }
    if module == "renderer":
        return {
            "change_type": "rendering",
            "target": module,
            "hypothesis": "Force required tabular sections when the deliverable asks for gap analysis or workpapers.",
            "validation": ["same_task"],
        }
    if module == "severity_calibrator":
        return {
            "change_type": "worker_calibration",
            "target": module,
            "hypothesis": "Add task-family severity calibration rules so worker matrices preserve expected risk labels.",
            "validation": ["same_task", "same_practice_area_smoke"],
        }
    if module == "calculator":
        return {
            "change_type": "worker_operator",
            "target": module,
            "hypothesis": "Add formula-level cheap-worker calculation operators for covenant, numeric, and scenario tasks.",
            "validation": ["same_task", "same_practice_area_smoke", "mixed_smoke"],
        }
    if FailureTag.DISTRACTOR_CONFUSION in tags:
        return {
            "change_type": "context_packing",
            "target": module,
            "hypothesis": "Add source/entity coverage controls so the final artifact cannot answer only one transaction, entity, or document family when the rubric expects several.",
            "validation": ["same_task", "same_practice_area_smoke", "mixed_smoke"],
        }
    return {
        "change_type": "synthesis",
        "target": module,
        "hypothesis": "Force preservation of worker-discovered enumerations and rubric-critical sections in the final deliverable.",
        "validation": ["same_task"],
    }
