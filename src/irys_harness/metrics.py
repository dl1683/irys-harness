from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ModelConfig, ModelTier


@dataclass
class ModelCallRecord:
    module: str
    tier: ModelTier
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    estimated_cost: float = 0.0
    retries: int = 0

    @classmethod
    def from_usage(
        cls,
        *,
        module: str,
        model_config: ModelConfig,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float = 0.0,
        retries: int = 0,
    ) -> "ModelCallRecord":
        cost = (
            input_tokens * model_config.input_cost_per_million
            + output_tokens * model_config.output_cost_per_million
        ) / 1_000_000
        return cls(
            module=module,
            tier=model_config.tier,
            model=model_config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
            estimated_cost=cost,
            retries=retries,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "tier": self.tier.value,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "estimated_cost": self.estimated_cost,
            "retries": self.retries,
        }


@dataclass
class QualityMetrics:
    score: float | None = None
    max_score: float | None = None
    passed: bool | None = None
    rubric_passed: int | None = None
    rubric_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "max_score": self.max_score,
            "passed": self.passed,
            "rubric_passed": self.rubric_passed,
            "rubric_total": self.rubric_total,
        }


@dataclass
class RunMetrics:
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    quality: QualityMetrics = field(default_factory=QualityMetrics)

    def add_call(self, call: ModelCallRecord) -> None:
        self.model_calls.append(call)

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.model_calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.model_calls)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float:
        return sum(call.estimated_cost for call in self.model_calls)

    @property
    def latency_seconds(self) -> float:
        return sum(call.latency_seconds for call in self.model_calls)

    def token_share_by_tier(self) -> dict[str, float]:
        total = self.total_tokens
        by_tier = {tier.value: 0 for tier in ModelTier}
        for call in self.model_calls:
            by_tier[call.tier.value] += call.input_tokens + call.output_tokens
        if total == 0:
            return {tier: 0.0 for tier in by_tier}
        return {tier: tokens / total for tier, tokens in by_tier.items()}

    def tokens_by_tier(self) -> dict[str, int]:
        by_tier = {tier.value: 0 for tier in ModelTier}
        for call in self.model_calls:
            by_tier[call.tier.value] += call.input_tokens + call.output_tokens
        return by_tier

    def cost_by_tier(self) -> dict[str, float]:
        by_tier = {tier.value: 0.0 for tier in ModelTier}
        for call in self.model_calls:
            by_tier[call.tier.value] += call.estimated_cost
        return by_tier

    def tokens_by_module(self) -> dict[str, int]:
        by_module: dict[str, int] = {}
        for call in self.model_calls:
            by_module[call.module] = by_module.get(call.module, 0) + call.input_tokens + call.output_tokens
        return by_module

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "latency_seconds": self.latency_seconds,
            "tokens_by_tier": self.tokens_by_tier(),
            "token_share_by_tier": self.token_share_by_tier(),
            "cost_by_tier": self.cost_by_tier(),
            "tokens_by_module": self.tokens_by_module(),
            "model_calls": [call.to_dict() for call in self.model_calls],
            "quality": self.quality.to_dict(),
        }

