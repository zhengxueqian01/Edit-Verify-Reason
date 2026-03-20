from __future__ import annotations

import json
import unittest

from main import (
    _maybe_apply_llm_intent_steps,
    _coerce_points,
    _coerce_steps,
    _heuristic_plan_update,
    _llm_plan_update,
    _llm_split_update_and_qa,
    _merge_structured_data_change,
    _merge_structured_operation_target,
    _normalize_structured_context,
    _normalize_gerund_clause,
    _operation_steps_from_plan,
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
                    "operation_target": {"del_category": "CrimsonLink"},
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
        self.assertEqual(result["operation_target"], {"del_category": "CrimsonLink"})
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
        self.assertIn("operation_target", llm.prompt)
        self.assertIn("data_change", llm.prompt)

    def test_resolve_questions_returns_operation_qa_and_data_change(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "operation_text": "Add two points.",
                    "qa_question": "How many clusters are there now?",
                    "operation_target": {"category_name": "new points"},
                    "data_change": {"points": [{"x": 10, "y": 20}, {"x": 15, "y": 18}]},
                    "llm_success": True,
                }
            )
        )

        operation_text, qa_question, split_info, data_change = _resolve_questions(
            {"question": "Add points (10, 20) and (15, 18), then how many clusters are there now?"},
            llm,
        )

        self.assertEqual(
            operation_text,
            'Add two points. operation_target: {"category_name": "new points"}, data_change: {"points": [{"x": 10, "y": 20}, {"x": 15, "y": 18}]}',
        )
        self.assertEqual(qa_question, "How many clusters are there now?")
        self.assertEqual(data_change, {"points": [{"x": 10, "y": 20}, {"x": 15, "y": 18}]})
        self.assertEqual(
            split_info["operation_text"],
            'Add two points. operation_target: {"category_name": "new points"}, data_change: {"points": [{"x": 10, "y": 20}, {"x": 15, "y": 18}]}',
        )
        self.assertEqual(split_info["operation_target"], {"category_name": "new points"})
        self.assertEqual(split_info["data_change"], data_change)

    def test_resolve_questions_prefers_llm_and_uses_rule_split_as_fallback(self) -> None:
        llm = _StubLLM(json.dumps({"llm_success": False}))

        operation_text, qa_question, split_info, data_change = _resolve_questions(
            {
                "question": (
                    "After adding a new category “Regional Carriers” with values across the years "
                    "2015–2024 (128,598; 186,977; 205,514; 136,129; 226,783; 246,727; 170,089; "
                    "154,587; 195,958; 176,685) and deleting the category “Charter Flights,” "
                    "in which year does the overall maximum occur?"
                )
            },
            llm,
        )

        self.assertEqual(
            operation_text,
            (
                "Add a new category “Regional Carriers” with values across the years 2015–2024 "
                "(128,598; 186,977; 205,514; 136,129; 226,783; 246,727; 170,089; 154,587; "
                "195,958; 176,685). Delete the category “Charter Flights."
            ),
        )
        self.assertEqual(qa_question, "in which year does the overall maximum occur?")
        self.assertEqual(
            split_info["operation_target"],
            {"add_category": "Regional Carriers", "del_category": "Charter Flights"},
        )
        self.assertEqual(
            data_change,
            {
                "add": {
                    "mode": "full_series",
                    "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
                    "values": [128598.0, 186977.0, 205514.0, 136129.0, 226783.0, 246727.0, 170089.0, 154587.0, 195958.0, 176685.0],
                },
                "del": {"category": "Charter Flights"},
            },
        )
        self.assertEqual(split_info["reason"], "llm_split_failed_fallback")

    def test_resolve_questions_embeds_inline_structured_suffix_into_operation(self) -> None:
        llm = _StubLLM(json.dumps({"llm_success": False}))

        operation_text, qa_question, split_info, data_change = _resolve_questions(
            {
                "question": (
                    'After adding the category Regional Carriers and deleting the category Charter Flights, '
                    'in which year does the overall maximum occur? '
                    '"operation_target": { "add_category": "Regional Carriers", "del_category": "Charter Flights" }, '
                    '"data_change": { "add": { "mode": "full_series", "years": [ "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024" ], '
                    '"values": [ 128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685 ] }, '
                    '"del": { "category": "Charter Flights" } },'
                )
            },
            llm,
        )

        self.assertIn("Add the category Regional Carriers", operation_text)
        self.assertIn("Delete the category Charter Flights", operation_text)
        self.assertIn('"operation_target": { "add_category": "Regional Carriers", "del_category": "Charter Flights" }', operation_text)
        self.assertIn('"data_change": { "add": { "mode": "full_series"', operation_text)
        self.assertEqual(qa_question, "in which year does the overall maximum occur?")
        self.assertEqual(split_info["operation_target"]["add_category"], "Regional Carriers")
        self.assertEqual(split_info["operation_target"]["del_category"], "Charter Flights")
        self.assertEqual(data_change["del"]["category"], "Charter Flights")

    def test_second_stage_builds_add_and_delete_steps_from_operation_text(self) -> None:
        text = (
            "After adding a new category “Regional Carriers” with values across the years "
            "2015–2024 (128,598; 186,977; 205,514; 136,129; 226,783; 246,727; 170,089; "
            "154,587; 195,958; 176,685) and deleting the category “Charter Flights,” "
            "in which year does the overall maximum occur?"
        )

        llm = _StubLLM(json.dumps({"llm_success": False}))
        operation_text, _qa_question, split_info, data_change = _resolve_questions({"question": text}, llm)
        structured_context = _normalize_structured_context({})
        structured_context = _merge_structured_operation_target(structured_context, split_info.get("operation_target"))
        structured_context = _merge_structured_data_change(structured_context, data_change)
        plan = {
            "operation": "unknown",
            "normalized_question": operation_text,
            "steps": [
                {
                    "operation": "add",
                    "question_hint": "Add the Regional Carriers series.",
                    "operation_target": {"category_name": "Regional Carriers"},
                    "data_change": {"mode": "full_series", "years": ["2015"], "values": [128598]},
                    "new_points": [],
                },
                {
                    "operation": "delete",
                    "question_hint": "Delete the Charter Flights series.",
                    "operation_target": {"category_name": "Charter Flights"},
                    "data_change": {},
                    "new_points": [],
                },
            ],
            "new_points": [],
            "llm_success": True,
        }
        steps = _operation_steps_from_plan(plan, operation_text, structured_context)

        self.assertEqual([step["operation"] for step in steps], ["add", "delete"])
        self.assertEqual(steps[0]["operation_target"], {"category_name": "Regional Carriers"})
        self.assertEqual(steps[0]["question_hint"], "Add the Regional Carriers series.")
        self.assertEqual(steps[0]["data_change"]["mode"], "full_series")
        self.assertEqual(steps[0]["data_change"]["years"][0], "2015")
        self.assertEqual(steps[0]["data_change"]["values"][0], 128598.0)
        self.assertEqual(steps[1]["operation_target"], {"category_name": "Charter Flights"})

    def test_second_stage_falls_back_to_rules_when_llm_steps_are_incomplete(self) -> None:
        llm = _StubLLM(json.dumps({"llm_success": False}))
        text = (
            'After adding the category Regional Carriers and deleting the category Charter Flights, '
            'in which year does the overall maximum occur? '
            '"operation_target": { "add_category": "Regional Carriers", "del_category": "Charter Flights" }, '
            '"data_change": { "add": { "mode": "full_series", "years": [ "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024" ], '
            '"values": [ 128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685 ] }, '
            '"del": { "category": "Charter Flights" } },'
        )

        operation_text, _qa_question, split_info, data_change = _resolve_questions({"question": text}, llm)
        structured_context = _normalize_structured_context({})
        structured_context = _merge_structured_operation_target(structured_context, split_info.get("operation_target"))
        structured_context = _merge_structured_data_change(structured_context, data_change)
        plan = {
            "operation": "delete",
            "normalized_question": operation_text,
            "steps": [
                {
                    "operation": "delete",
                    "question_hint": "Delete the Charter Flights series.",
                    "operation_target": {"category_name": "Charter Flights"},
                    "data_change": {},
                    "new_points": [],
                }
            ],
            "new_points": [],
            "llm_success": True,
        }
        steps = _operation_steps_from_plan(plan, operation_text, structured_context)

        self.assertEqual([step["operation"] for step in steps], ["add", "delete"])
        self.assertEqual(steps[0]["operation_target"], {"category_name": "Regional Carriers"})

    def test_llm_intent_mode_overrides_plan_steps_when_available(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "steps": [
                        {
                            "operation": "delete",
                            "question_hint": "Delete Oracle Labs.",
                            "operation_target": {"category_name": "Oracle Labs"},
                            "data_change": {},
                            "new_points": [],
                        }
                    ]
                }
            )
        )
        operation_plan = {
            "operation": "change",
            "normalized_question": "Delete Oracle Labs and add Cinder Guild.",
            "steps": [],
            "new_points": [],
            "llm_success": True,
        }

        updated = _maybe_apply_llm_intent_steps(
            operation_plan=operation_plan,
            operation_text="Delete Oracle Labs and add Cinder Guild.",
            chart_type="line",
            perception={"chart_type": "line", "primitives_summary": {"num_lines": 5}, "mapping_info": {}},
            structured_context={"operation_target": {"del_category": "Oracle Labs", "add_category": "Cinder Guild"}},
            llm=llm,
            update_mode="llm_intent",
        )

        self.assertTrue(updated["llm_intent_success"])
        self.assertEqual(len(updated["steps"]), 1)
        self.assertEqual(updated["steps"][0]["operation"], "delete")
        self.assertEqual(updated["steps"][0]["operation_target"], {"category_name": "Oracle Labs"})

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
