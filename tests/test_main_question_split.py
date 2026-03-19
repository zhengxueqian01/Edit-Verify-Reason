from __future__ import annotations

import json
import unittest

from main import (
    _coerce_points,
    _coerce_steps,
    _llm_plan_update,
    _llm_split_update_and_qa,
    _merge_structured_data_change,
    _normalize_gerund_clause,
    _resolve_questions,
)


class _StubLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompt = ""

    def invoke(self, prompt: str) -> object:
        self.prompt = prompt
        return type("Resp", (), {"content": self.content})()


class MainQuestionSplitTests(unittest.TestCase):
    def test_normalize_gerund_clause_rewrites_conjoined_gerunds(self) -> None:
        text = "deleting the category CrimsonLink and applying the listed value revisions"

        normalized = _normalize_gerund_clause(text)

        self.assertEqual(normalized, "Delete the category CrimsonLink and apply the listed value revisions.")

    def test_llm_split_accepts_explicit_stepwise_update_question(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "operation_text": "1. Delete the category CrimsonLink; 2. Apply the listed value revisions.",
                    "qa_question": "How many times do the lines for Starburst and AetherNet intersect?",
                    "data_change": {
                        "change": {
                            "changes": [
                                {"category_name": "Starburst", "years": [2020], "values": [12]}
                            ]
                        }
                    },
                    "llm_success": True,
                }
            )
        )

        result = _llm_split_update_and_qa(
            "After deleting the category CrimsonLink and applying the listed value revisions, how many times do the lines for Starburst and AetherNet intersect?",
            llm,
        )

        self.assertTrue(result["llm_success"])
        self.assertIn("1. Delete the category CrimsonLink;", result["update_question"])
        self.assertEqual(
            result["qa_question"],
            "How many times do the lines for Starburst and AetherNet intersect?",
        )
        self.assertEqual(
            result["data_change"],
            {
                "change": {
                    "changes": [
                        {"category_name": "Starburst", "years": [2020], "values": [12]}
                    ]
                }
            },
        )
        self.assertIn("Never use gerunds like 'adding', 'deleting', 'applying'", llm.prompt)
        self.assertIn("data_change", llm.prompt)

    def test_resolve_questions_returns_operation_qa_and_data_change(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "operation_text": "Add two points.",
                    "qa_question": "How many clusters are there now?",
                    "data_change": {"points": [{"x": 10, "y": 20}, {"x": 15, "y": 18}]},
                    "llm_success": True,
                }
            )
        )

        operation_text, qa_question, split_info, data_change = _resolve_questions(
            {"question": "Add points (10, 20) and (15, 18), then how many clusters are there now?"},
            llm,
        )

        self.assertEqual(operation_text, "Add two points.")
        self.assertEqual(qa_question, "How many clusters are there now?")
        self.assertEqual(data_change, {"points": [{"x": 10, "y": 20}, {"x": 15, "y": 18}]})
        self.assertEqual(split_info["operation_text"], "Add two points.")
        self.assertEqual(split_info["data_change"], data_change)

    def test_merge_structured_data_change_keeps_existing_and_adds_split_payload(self) -> None:
        merged = _merge_structured_data_change(
            {"data_change": {"add": {"values": [1, 2, 3]}}},
            {"change": {"changes": [{"category_name": "A", "years": [2024], "values": [9]}]}},
        )

        self.assertEqual(
            merged["data_change"],
            {
                "add": {"values": [1, 2, 3]},
                "change": {"changes": [{"category_name": "A", "years": [2024], "values": [9]}]},
            },
        )

    def test_coerce_points_preserves_per_point_color(self) -> None:
        points = _coerce_points(
            [
                {"x": 1, "y": 2, "color": "blue"},
                {"x": 3, "y": 4, "fill": "#ff7f0e"},
            ]
        )

        self.assertEqual(
            points,
            [
                {"x": 1.0, "y": 2.0, "color": "blue"},
                {"x": 3.0, "y": 4.0, "fill": "#ff7f0e"},
            ],
        )

    def test_llm_plan_update_prompt_requires_scatter_point_colors(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "operation": "add",
                    "normalized_question": "Add the specified points to the scatter chart.",
                    "steps": [{"operation": "add", "question_hint": "Insert points."}],
                    "new_points": [{"x": 1, "y": 2, "color": "blue"}],
                }
            )
        )

        result = _llm_plan_update("Add scatter points", "scatter", llm)

        self.assertTrue(result["llm_success"])
        self.assertEqual(result["new_points"], [{"x": 1.0, "y": 2.0, "color": "blue"}])
        self.assertIn("color?:string", llm.prompt)
        self.assertIn("copy them through to each new_points item", llm.prompt)

    def test_coerce_steps_preserves_step_level_targets_and_data(self) -> None:
        steps = _coerce_steps(
            [
                {
                    "operation": "change",
                    "question_hint": "Update the 2024 value for Alpha.",
                    "operation_target": {"category_name": "Alpha"},
                    "data_change": {"changes": [{"category_name": "Alpha", "years": [2024], "values": [9]}]},
                }
            ]
        )

        self.assertEqual(
            steps,
            [
                {
                    "operation": "change",
                    "question": "",
                    "question_hint": "Update the 2024 value for Alpha.",
                    "operation_target": {"category_name": "Alpha"},
                    "data_change": {"changes": [{"category_name": "Alpha", "years": [2024], "values": [9]}]},
                    "new_points": [],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
