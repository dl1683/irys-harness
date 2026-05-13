from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ModelTier(StrEnum):
    CHEAP_WORKER = "cheap_worker"
    MID_ORCHESTRATOR = "mid_orchestrator"
    STRONG_SYNTHESIZER = "strong_synthesizer"


@dataclass(frozen=True)
class ModelConfig:
    tier: ModelTier
    model: str
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    max_iterations: int = 3
    memory_enabled: bool = False
    web_enabled: bool = False


@dataclass(frozen=True)
class HarnessConfig:
    run: RunConfig = field(default_factory=RunConfig)
    judge_model: str = "gemini-3.1-flash-lite-preview"
    models: dict[ModelTier, ModelConfig] = field(default_factory=dict)
    module_tiers: dict[str, ModelTier] = field(default_factory=dict)

    def model_for_module(self, module: str) -> ModelConfig:
        try:
            tier = self.module_tiers[module]
        except KeyError as exc:
            raise ConfigError(f"Module {module!r} has no configured model tier") from exc
        try:
            return self.models[tier]
        except KeyError as exc:
            raise ConfigError(f"Tier {tier!r} has no configured model") from exc

    def validate(self) -> None:
        required = set(ModelTier)
        missing = required.difference(self.models)
        if missing:
            labels = ", ".join(sorted(t.value for t in missing))
            raise ConfigError(f"Missing model tier config: {labels}")

        for name, tier in self.module_tiers.items():
            if tier not in self.models:
                raise ConfigError(f"Module {name!r} references unconfigured tier {tier!r}")

        if self.run.max_iterations < 1:
            raise ConfigError("run.max_iterations must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": {
                "max_iterations": self.run.max_iterations,
                "memory_enabled": self.run.memory_enabled,
                "web_enabled": self.run.web_enabled,
            },
            "judge_model": self.judge_model,
            "models": {
                tier.value: {
                    "model": config.model,
                    "input_cost_per_million": config.input_cost_per_million,
                    "output_cost_per_million": config.output_cost_per_million,
                }
                for tier, config in self.models.items()
            },
            "module_tiers": {module: tier.value for module, tier in self.module_tiers.items()},
        }


class ConfigError(ValueError):
    pass


def load_config(path: str | Path | None = None) -> HarnessConfig:
    config_path = Path(path) if path else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = parse_config(raw)
    config.validate()
    return config


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.json"


def parse_config(raw: dict[str, Any]) -> HarnessConfig:
    run_raw = raw.get("run", {})
    run = RunConfig(
        max_iterations=int(run_raw.get("max_iterations", 3)),
        memory_enabled=bool(run_raw.get("memory_enabled", False)),
        web_enabled=bool(run_raw.get("web_enabled", False)),
    )

    models: dict[ModelTier, ModelConfig] = {}
    for tier_label, item in raw.get("models", {}).items():
        tier = ModelTier(tier_label)
        model_name = str(item["model"])
        env_override = os.getenv(f"IRYS_MODEL_{tier.value.upper()}")
        models[tier] = ModelConfig(
            tier=tier,
            model=env_override or model_name,
            input_cost_per_million=float(item.get("input_cost_per_million", 0.0)),
            output_cost_per_million=float(item.get("output_cost_per_million", 0.0)),
        )

    module_tiers = {
        str(module): ModelTier(tier_label)
        for module, tier_label in raw.get("module_tiers", {}).items()
    }

    judge_model = os.getenv("IRYS_HARNESS_JUDGE_MODEL") or str(
        raw.get("judge_model", "gemini-3.1-flash-lite-preview")
    )

    return HarnessConfig(
        run=run,
        judge_model=judge_model,
        models=models,
        module_tiers=module_tiers,
    )

