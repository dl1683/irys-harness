from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from irys_harness.benchmarks.harvey import (
    HarveyLabAdapter,
    discover_tasks,
    extract_deliverables,
    extract_deliverables_from_instructions,
)


class HarveyAdapterTests(unittest.TestCase):
    def test_extract_deliverables_from_criteria(self) -> None:
        self.assertEqual(
            extract_deliverables(
                {
                    "criteria": [
                        {"deliverables": ["b.docx", "a.docx"]},
                        {"deliverables": ["a.docx"]},
                    ]
                }
            ),
            ["a.docx", "b.docx"],
        )

    def test_extract_deliverables_from_instructions(self) -> None:
        self.assertEqual(
            extract_deliverables_from_instructions("Output: `memo.docx` and workpapers.xlsx."),
            ["memo.docx", "workpapers.xlsx"],
        )

    def test_discover_tasks_and_load_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "area-one" / "task-one"
            docs_dir = task_dir / "documents"
            docs_dir.mkdir(parents=True)
            (docs_dir / "source.docx").write_text("doc", encoding="utf-8")
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "title": "Title",
                        "instructions": "Make the thing. Output: `memo.docx`.",
                        "criteria": [
                            {
                                "id": "C-001",
                                "title": "Deliverable exists",
                                "deliverables": ["memo.docx"],
                                "match_criteria": "PASS if memo exists.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            refs = discover_tasks(root)
            self.assertEqual(refs[0].task_id, "area-one/task-one")
            task = HarveyLabAdapter(root=root).load_task("first")
            self.assertEqual(task.answer_schema["deliverables"], ["memo.docx"])
            self.assertEqual(len(task.context_files), 1)


if __name__ == "__main__":
    unittest.main()
