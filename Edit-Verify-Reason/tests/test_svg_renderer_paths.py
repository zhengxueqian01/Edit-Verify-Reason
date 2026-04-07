from __future__ import annotations

import unittest

from chart_agent.perception.svg_renderer import default_output_paths


class SvgRendererPathTests(unittest.TestCase):
    def test_default_output_paths_include_task_scope_for_dataset_cases(self) -> None:
        add_del_svg, _ = default_output_paths("dataset/task1/add-del/000/000.svg", "area")
        add_change_svg, _ = default_output_paths("dataset/task1/add-change/000/000.svg", "area")

        self.assertIn("task1-add-del_000", add_del_svg)
        self.assertIn("task1-add-change_000", add_change_svg)
        self.assertNotEqual(add_del_svg, add_change_svg)


if __name__ == "__main__":
    unittest.main()
