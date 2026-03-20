from __future__ import annotations

import unittest

from run_dataset_via_main import build_structured_update_context, choose_qa


class RunDatasetContextTests(unittest.TestCase):
    def test_choose_qa_preserves_extra_cluster_fields(self) -> None:
        payload = {
            "QA": [
                {
                    "question": "After adding these points, how many clusters are in the chart now?",
                    "answer": "4",
                    "eps": 5.0,
                    "min_samples": 3,
                }
            ]
        }

        question, answer, qa_item = choose_qa(payload, 0)

        self.assertIn("how many clusters", question.lower())
        self.assertEqual(answer, "4")
        self.assertEqual(qa_item["eps"], 5.0)
        self.assertEqual(qa_item["min_samples"], 3)

    def test_build_structured_update_context_keeps_only_execution_fields(self) -> None:
        context = build_structured_update_context(
            {
                "chart_type": "scatter",
                "operation": "add",
                "operation_target": {"add_category": "Echo Bowl"},
                "data_change": {"add": {"values": [1, 2, 3]}},
            },
            {
                "eps": 5.0,
                "min_samples": 3,
            },
        )

        self.assertEqual(context["operation_target"]["add_category"], "Echo Bowl")
        self.assertEqual(context["data_change"]["add"]["values"], [1, 2, 3])
        self.assertNotIn("task", context)
        self.assertNotIn("chart_type", context)
        self.assertNotIn("operation", context)
        self.assertNotIn("cluster_params", context)


if __name__ == "__main__":
    unittest.main()
