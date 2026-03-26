from __future__ import annotations

import unittest

from main import _atomic_changes_from_step, _operation_steps_from_plan


class MainStructuredChangeLabelsTests(unittest.TestCase):
    def test_atomic_changes_from_step_keeps_label_from_operation_target(self) -> None:
        changes = _atomic_changes_from_step(
            {"category_name": "Charter Flights"},
            {
                "years": ["2024"],
                "values": [168860],
            },
        )

        self.assertEqual(
            changes,
            [
                {
                    "category_name": "Charter Flights",
                    "years": ["2024"],
                    "values": [168860],
                }
            ],
        )

    def test_operation_steps_from_plan_preserves_labels_for_split_change_steps(self) -> None:
        operation_plan = {
            "steps": [
                {
                    "operation": "add",
                    "operation_target": {"category_name": "Business Class"},
                    "data_change": {
                        "category_name": "Business Class",
                        "years": ["2015", "2016"],
                        "values": [1, 2],
                    },
                },
                {
                    "operation": "change",
                    "operation_target": {"category_name": "Charter Flights"},
                    "data_change": {
                        "years": ["2024"],
                        "values": [168860],
                    },
                },
                {
                    "operation": "change",
                    "operation_target": {"category_name": "Last-Minute Deals"},
                    "data_change": {
                        "years": ["2021", "2022"],
                        "values": [238865, 229291],
                    },
                },
            ]
        }

        steps = _operation_steps_from_plan(
            operation_plan,
            "Add Business Class category and apply specified value revisions.",
            {"operation_target": {}, "data_change": {}},
        )

        change_targets = [
            step.get("operation_target", {}).get("category_name")
            for step in steps
            if step.get("operation") == "change"
        ]
        self.assertEqual(
            change_targets,
            ["Charter Flights", "Last-Minute Deals", "Last-Minute Deals"],
        )


if __name__ == "__main__":
    unittest.main()
