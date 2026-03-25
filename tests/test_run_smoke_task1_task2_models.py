from __future__ import annotations

import unittest
from pathlib import Path

from run_dataset_via_main import build_model_overrides
from run_smoke_task1_task2_models import DEFAULT_MODELS, TASK_SPECS, build_task_commands


class RunSmokeTask1Task2ModelsTests(unittest.TestCase):
    def test_build_model_overrides_covers_all_runtime_tasks(self) -> None:
        self.assertEqual(
            build_model_overrides("gemini"),
            {
                "splitter": "gemini",
                "planner": "gemini",
                "executor": "gemini",
                "answer": "gemini",
                "tool_planner": "gemini",
            },
        )

    def test_task_specs_cover_task1_and_task2_batches(self) -> None:
        self.assertEqual(
            [task_name for task_name, _ in TASK_SPECS],
            [
                "task1_add",
                "task1_add-change",
                "task1_add-del",
                "task1_change",
                "task1_del",
                "task1_del-change",
                "task2_del",
                "task2_del-add",
                "task2_del-change",
            ],
        )

    def test_build_task_commands_uses_session_model_layout_and_resume(self) -> None:
        commands = build_task_commands(
            python_executable="/usr/bin/python3",
            session_root=Path("/tmp/session"),
            models=list(DEFAULT_MODELS[:2]),
            cases_per_task=10,
            qa_index=0,
            max_render_retries=2,
            resume=True,
        )

        self.assertEqual(len(commands), 18)
        first = commands[0]
        self.assertEqual(
            first,
            [
                "/usr/bin/python3",
                str(Path("/Users/xueqianzheng/code/ChartAgent/src/run_smoke_task1_task2_models.py").resolve().parents[1] / "src" / "run_dataset_via_main.py"),
                "--input-dir",
                "dataset/task1/add",
                "--qa-index",
                "0",
                "--max-render-retries",
                "2",
                "--limit",
                "10",
                "--record-root",
                "/tmp/session/claude",
                "--model",
                "claude",
                "--resume",
            ],
        )


if __name__ == "__main__":
    unittest.main()
