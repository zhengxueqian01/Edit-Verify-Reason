from __future__ import annotations

import unittest
from pathlib import Path

from src.run_validate_dataset_records import build_output_path, infer_dataset_dir


class RunValidateDatasetRecordsTests(unittest.TestCase):
    def test_infer_dataset_dir_prefers_task1_directory(self) -> None:
        self.assertEqual(infer_dataset_dir("task1_add-change"), "task1/add-change")

    def test_build_output_path_uses_passed_records_root(self) -> None:
        records_root = Path("/tmp/custom/dataset_records/session")
        pred_root = records_root / "gpt" / "task1_add"

        out_path = build_output_path(pred_root, records_root)

        self.assertEqual(
            out_path,
            Path("/Users/xueqianzheng/code/ChartAgent/output/svg_match/session/gpt/task1_add.json"),
        )


if __name__ == "__main__":
    unittest.main()
