from __future__ import annotations

import tempfile
import unittest

from irys_harness.experiments import close_experiment, open_experiment, read_experiment


class ExperimentTests(unittest.TestCase):
    def test_open_and_close_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = open_experiment(
                baseline_run="baseline",
                hypothesis="Improve routing",
                target="routing",
                experiments_dir=tmp,
            )
            self.assertTrue(path.exists())
            self.assertEqual(read_experiment(path).status, "open")
            closed = close_experiment(
                path,
                experiment_run="experiment",
                accepted=True,
                decision_reason="Token share improved without score loss.",
                comparison={"summary": {"passed_delta": 0}},
            )
            self.assertEqual(closed.status, "accepted")
            self.assertTrue(closed.accepted)
            self.assertEqual(read_experiment(path).experiment_run, "experiment")


if __name__ == "__main__":
    unittest.main()
