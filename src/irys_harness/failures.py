from __future__ import annotations

from enum import StrEnum


class FailureTag(StrEnum):
    ADAPTER_ERROR = "adapter_error"
    CONTRACT_ERROR = "contract_error"
    RETRIEVAL_MISS = "retrieval_miss"
    WRONG_CHUNK_SELECTED = "wrong_chunk_selected"
    DISTRACTOR_CONFUSION = "distractor_confusion"
    MISSING_EXCEPTION = "missing_exception"
    BAD_EXTRACTION = "bad_extraction"
    UNSUPPORTED_INFERENCE = "unsupported_inference"
    BAD_VERIFICATION = "bad_verification"
    WRONG_COMPUTATION = "wrong_computation"
    CONTEXT_PACKING_ERROR = "context_packing_error"
    SYNTHESIS_ERROR = "synthesis_error"
    SEVERITY_CALIBRATION_ERROR = "severity_calibration_error"
    FORMAT_ERROR = "format_error"
    CITATION_ERROR = "citation_error"
    SCORER_ADAPTER_ERROR = "scorer_adapter_error"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    TRACE_INCOMPLETE = "trace_incomplete"
    UNCLASSIFIED_FAILURE = "unclassified_failure"


def validate_failure_tags(tags: list[str]) -> list[str]:
    valid = {tag.value for tag in FailureTag}
    unknown = sorted(set(tags).difference(valid))
    if unknown:
        raise ValueError("Unknown failure tags: " + ", ".join(unknown))
    return tags
