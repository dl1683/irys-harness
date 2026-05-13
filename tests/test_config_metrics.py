from __future__ import annotations

import unittest

from irys_harness.config import ModelTier, load_config
from irys_harness.metrics import ModelCallRecord, RunMetrics


class ConfigMetricsTests(unittest.TestCase):
    def test_default_config_has_three_tiers(self) -> None:
        config = load_config()
        self.assertEqual(set(config.models), set(ModelTier))
        self.assertEqual(config.model_for_module("extraction").tier, ModelTier.CHEAP_WORKER)
        self.assertEqual(config.model_for_module("synthesis").tier, ModelTier.STRONG_SYNTHESIZER)

    def test_metrics_report_token_share_by_tier(self) -> None:
        config = load_config()
        metrics = RunMetrics()
        metrics.add_call(
            ModelCallRecord.from_usage(
                module="extraction",
                model_config=config.model_for_module("extraction"),
                input_tokens=900,
                output_tokens=100,
            )
        )
        metrics.add_call(
            ModelCallRecord.from_usage(
                module="synthesis",
                model_config=config.model_for_module("synthesis"),
                input_tokens=50,
                output_tokens=50,
            )
        )
        shares = metrics.token_share_by_tier()
        self.assertAlmostEqual(shares["cheap_worker"], 1000 / 1100)
        self.assertAlmostEqual(shares["strong_synthesizer"], 100 / 1100)


if __name__ == "__main__":
    unittest.main()

