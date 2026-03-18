from __future__ import annotations

import unittest

from run_dataset_via_main import DATASET_ROOT, build_run_dir_name


class RunDatasetResumeLayoutTests(unittest.TestCase):
    def test_build_run_dir_name_is_stable_without_timestamp(self) -> None:
        input_dir = DATASET_ROOT / "task1-mix-area" / "add-change"

        self.assertEqual(build_run_dir_name(input_dir), "task1_add-change")


if __name__ == "__main__":
    unittest.main()
