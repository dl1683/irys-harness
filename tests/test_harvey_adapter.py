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

    def test_load_task_recurses_nested_document_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "area-one" / "nested-doc-task"
            docs_dir = task_dir / "documents"
            (docs_dir / "folder-a").mkdir(parents=True)
            (docs_dir / "folder-b").mkdir(parents=True)
            (docs_dir / "folder-a" / "source-a.docx").write_text("doc a", encoding="utf-8")
            (docs_dir / "folder-b" / "source-b.pdf").write_text("doc b", encoding="utf-8")
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "title": "Nested docs",
                        "instructions": "Make the thing. Output: `memo.docx`.",
                        "criteria": [{"id": "C-001", "title": "Deliverable exists"}],
                    }
                ),
                encoding="utf-8",
            )

            task = HarveyLabAdapter(root=root).load_task("area-one/nested-doc-task")

            self.assertEqual(
                sorted(Path(path).name for path in task.context_files),
                ["source-a.docx", "source-b.pdf"],
            )

    def test_sample_split_uses_balanced_target_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for area in ["area-a", "area-b", "area-c"]:
                for index in range(4):
                    task_dir = root / "tasks" / area / f"task-{index}"
                    task_dir.mkdir(parents=True)
                    (task_dir / "task.json").write_text(
                        json.dumps(
                            {
                                "title": f"{area} {index}",
                                "instructions": "Make the thing. Output: `memo.docx`.",
                                "criteria": [{"id": "C-001", "title": "Deliverable exists"}],
                            }
                        ),
                        encoding="utf-8",
                    )

            refs = HarveyLabAdapter(root=root, sample_size=5).list_tasks("sample")
            self.assertEqual(
                [ref.task_id for ref in refs],
                [
                    "area-a/task-0",
                    "area-b/task-0",
                    "area-c/task-0",
                    "area-a/task-1",
                    "area-b/task-1",
                ],
            )
            self.assertEqual(len(HarveyLabAdapter(root=root).list_tasks("sample500")), 12)


if __name__ == "__main__":
    unittest.main()
