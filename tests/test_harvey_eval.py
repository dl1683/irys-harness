from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from irys_harness.benchmarks.harvey_eval import prepare_harvey_eval_package


class HarveyEvalTests(unittest.TestCase):
    def test_prepare_harvey_eval_package_copies_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.docx"
            artifact.write_text("content", encoding="utf-8")
            trace = {
                "benchmark": "harvey_lab_sample",
                "run_id": "run123",
                "task_id": "area/task",
                "artifacts": [{"path": str(artifact)}],
            }
            package = prepare_harvey_eval_package(trace, harvey_root=root / "harvey")
            self.assertEqual(package.run_id, "irys/run123")
            self.assertEqual(len(package.copied_files), 1)
            self.assertTrue(Path(package.copied_files[0]).exists())


if __name__ == "__main__":
    unittest.main()
