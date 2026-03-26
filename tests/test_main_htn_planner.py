from __future__ import annotations

import unittest

from main import _operation_steps_from_plan


class MainHtnPlannerTests(unittest.TestCase):
    def test_operation_steps_from_plan_htn_selects_chart_specific_method(self) -> None:
        operation_plan = {"steps": []}
        structured_context = {
            "data_change": {
                "del": {"category_name": "Veridian Followers"},
                "add": {
                    "category_name": "Jade Point",
                    "years": ["1953", "1954"],
                    "values": [69.69, 68.9],
                },
            }
        }

        steps = _operation_steps_from_plan(
            operation_plan,
            "Delete Veridian Followers and add Jade Point.",
            structured_context,
            chart_type="line",
            update_mode="htn",
        )

        self.assertEqual(
            [(step.get("operation"), step.get("operation_target", {}).get("category_name")) for step in steps],
            [("delete", "Veridian Followers"), ("add", "Jade Point")],
        )
        methods = [item.get("method") for item in operation_plan.get("htn_trace", [])]
        self.assertIn("select_line_update_method", methods)
        self.assertIn("decompose_line_update", methods)
        self.assertEqual(operation_plan.get("resolved_steps"), steps)

    def test_operation_steps_from_plan_htn_splits_composite_delete_and_change_steps(self) -> None:
        operation_plan = {
            "steps": [
                {
                    "operation": "delete",
                    "question_hint": "Delete Alpha and Beta",
                    "operation_target": {"category_names": ["Alpha", "Beta"]},
                    "data_change": {"del": {"category_names": ["Alpha", "Beta"]}},
                    "new_points": [],
                },
                {
                    "operation": "change",
                    "question_hint": "Apply listed value revisions.",
                    "operation_target": {"category_name": "Gamma"},
                    "data_change": {
                        "changes": [
                            {"category_name": "Gamma", "years": ["2020"], "values": [10]},
                            {"category_name": "Gamma", "years": ["2021"], "values": [11]},
                        ]
                    },
                    "new_points": [],
                },
            ]
        }

        steps = _operation_steps_from_plan(
            operation_plan,
            "Delete Alpha and Beta, then apply listed value revisions.",
            {},
            chart_type="line",
            update_mode="htn",
        )

        self.assertEqual(
            [(step.get("operation"), step.get("operation_target", {}).get("category_name")) for step in steps],
            [("delete", "Alpha"), ("delete", "Beta"), ("change", "Gamma"), ("change", "Gamma")],
        )
        self.assertEqual(operation_plan["planning_form"], "htn")
        self.assertTrue(operation_plan.get("htn_trace"))
        self.assertEqual(operation_plan.get("resolved_steps"), steps)


if __name__ == "__main__":
    unittest.main()
