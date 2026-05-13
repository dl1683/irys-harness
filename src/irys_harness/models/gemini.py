from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from google import genai

from irys_harness.config import HarnessConfig
from irys_harness.metrics import ModelCallRecord


@dataclass(frozen=True)
class ModelResult:
    text: str
    usage: ModelCallRecord
    raw: Any = None


class GeminiModelRouter:
    def __init__(
        self,
        config: HarnessConfig,
        *,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
    ) -> None:
        self.config = config
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for live Gemini calls")
        self.client = genai.Client(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def generate(
        self,
        *,
        module: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> ModelResult:
        model_config = self.config.model_for_module(module)
        request_config: dict[str, Any] = {"temperature": temperature}
        if max_output_tokens is not None:
            request_config["max_output_tokens"] = max_output_tokens

        started = time.perf_counter()
        response = None
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=model_config.model,
                    contents=prompt,
                    config=request_config,
                )
                break
            except Exception as exc:  # noqa: BLE001 - provider SDK exposes several transient exception types.
                last_error = exc
                if attempt >= self.max_retries or not is_transient_model_error(exc):
                    raise
                time.sleep(self.retry_base_seconds * (2**attempt))
        if response is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Gemini request failed without a response")
        latency = time.perf_counter() - started
        input_tokens, output_tokens = usage_tokens(response)
        usage = ModelCallRecord.from_usage(
            module=module,
            model_config=model_config,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
        )
        return ModelResult(text=response_text(response), usage=usage, raw=response)


def is_transient_model_error(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    transient_markers = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "bad gateway",
        "deadline",
        "rate limit",
        "servererror",
        "service unavailable",
        "temporarily unavailable",
        "temporary error",
        "timeout",
    ]
    return any(marker in text for marker in transient_markers)


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(str(part_text))
    return "\n".join(parts)


def usage_tokens(response: Any) -> tuple[int, int]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return 0, 0
    input_tokens = int(
        getattr(metadata, "prompt_token_count", None)
        or getattr(metadata, "input_token_count", None)
        or 0
    )
    output_tokens = int(
        getattr(metadata, "candidates_token_count", None)
        or getattr(metadata, "output_token_count", None)
        or 0
    )
    return input_tokens, output_tokens
