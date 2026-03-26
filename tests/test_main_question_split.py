from __future__ import annotations

import json
import unittest

from main import (
    _effective_update_spec_for_render,
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
    def test_resolve_questions_preserves_dbscan_suffix_in_qa_question(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "operation_text": "Add the listed points to the scatter chart.",
                    "qa_question": "How many clusters are in the chart now?",
                    "operation_target": {},
                    "data_change": {"add": {"points": [{"x": 10, "y": 20, "color": "red"}]}},
                    "llm_success": True,
                }
            )
        )

        _operation_text, qa_question, _split_info, _data_change = _resolve_questions(
            {
                "question": "After adding these points, how many clusters are in the chart now? (eps: 5.1, min_samples:3 )"
            },
            llm,
        )

        self.assertEqual(qa_question, "How many clusters are in the chart now? (eps: 5.1, min_samples:3 )")

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

    def test_resolve_questions_recovers_embedded_data_change_from_operation_text(self) -> None:
        llm = _StubLLM(
            json.dumps(
                {
                    "operation_text": (
                        '1. Add the category Business Class; 2. Apply the listed value revisions. '
                        '"data_change": {"add": {"category_name": "Business Class", "years": ["2015"], "values": [158822]}, '
                        '"change": {"changes": [{"category_name": "Charter Flights", "years": ["2024"], "values": [168860]}]}}'
                    ),
                    "qa_question": "In which year does the overall maximum occur?",
                    "operation_target": {"add_category": "Business Class"},
                    "data_change": {"add": {"category_name": "Business Class", "years": ["2015"], "values": [158822]}},
                    "llm_success": True,
                }
            )
        )

        _operation_text, _qa_question, split_info, data_change = _resolve_questions(
            {
                "question": "After adding the category Business Class and applying the listed value revisions, in which year does the overall maximum occur?"
            },
            llm,
        )

        self.assertEqual(
            data_change,
            {
                "add": {"category_name": "Business Class", "years": ["2015"], "values": [158822]},
                "change": {
                    "changes": [
                        {"category_name": "Charter Flights", "years": ["2024"], "values": [168860]}
                    ]
                },
            },
        )
        self.assertEqual(split_info["data_change"], data_change)

    def test_effective_update_spec_for_render_prefers_step_scatter_points(self) -> None:
        update_spec = {"new_points": [{"x": 5.0, "y": 3.0}]}
        step_logs = [
            {
                "operation": "add",
                "new_points": [
                    {"x": 34.69, "y": 72.6, "color": "orange"},
                    {"x": 35.82, "y": 68.84, "color": "orange"},
                ],
            }
        ]

        effective = _effective_update_spec_for_render(
            chart_type="scatter",
            update_spec=update_spec,
            step_logs=step_logs,
        )

        self.assertEqual(
            effective["new_points"],
            [
                {"x": 34.69, "y": 72.6, "color": "orange"},
                {"x": 35.82, "y": 68.84, "color": "orange"},
            ],
        )

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
                    "category_name": "Regional Carriers",
                    "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
                    "values": [128598.0, 186977.0, 205514.0, 136129.0, 226783.0, 246727.0, 170089.0, 154587.0, 195958.0, 176685.0],
                },
                "del": {"category_name": "Charter Flights"},
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
                    '"operation_target": { "category_name": "Regional Carriers" }, '
                    '"data_change": { "add": { "category_name": "Regional Carriers", "years": [ "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024" ], '
                    '"values": [ 128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685 ] }, '
                    '"del": { "category_name": "Charter Flights" } },'
                )
            },
            llm,
        )

        self.assertIn("Add the category Regional Carriers", operation_text)
        self.assertIn("Delete the category Charter Flights", operation_text)
        self.assertIn('"operation_target": { "category_name": "Regional Carriers" }', operation_text)
        self.assertIn('"data_change": { "add": { "category_name": "Regional Carriers"', operation_text)
        self.assertEqual(qa_question, "in which year does the overall maximum occur?")
        self.assertEqual(split_info["operation_target"]["category_name"], "Regional Carriers")
        self.assertEqual(data_change["del"]["category_name"], "Charter Flights")

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
            "normalized_question": operation_text,
            "steps": [
                {
                    "operation": "add",
                    "question_hint": "Add the Regional Carriers series.",
                    "operation_target": {"category_name": "Regional Carriers"},
                    "data_change": {"category_name": "Regional Carriers", "years": ["2015"], "values": [128598]},
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
        self.assertEqual(steps[0]["data_change"]["category_name"], "Regional Carriers")
        self.assertEqual(steps[0]["data_change"]["years"][0], "2015")
        self.assertEqual(steps[0]["data_change"]["values"][0], 128598.0)
        self.assertEqual(steps[1]["operation_target"], {"category_name": "Charter Flights"})

    def test_second_stage_falls_back_to_rules_when_llm_steps_are_incomplete(self) -> None:
        llm = _StubLLM(json.dumps({"llm_success": False}))
        text = (
            'After adding the category Regional Carriers and deleting the category Charter Flights, '
            'in which year does the overall maximum occur? '
            '"operation_target": { "category_name": "Regional Carriers" }, '
            '"data_change": { "add": { "category_name": "Regional Carriers", "years": [ "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024" ], '
            '"values": [ 128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685 ] }, '
            '"del": { "category_name": "Charter Flights" } },'
        )

        operation_text, _qa_question, split_info, data_change = _resolve_questions({"question": text}, llm)
        structured_context = _normalize_structured_context({})
        structured_context = _merge_structured_operation_target(structured_context, split_info.get("operation_target"))
        structured_context = _merge_structured_data_change(structured_context, data_change)
        plan = {
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

    def test_rule_split_preserves_multiple_delete_labels_in_single_clause(self) -> None:
        llm = _StubLLM(json.dumps({"llm_success": False}))
        text = (
            "After deleting the categories Meridian Territory Sanitation and Silverhaven District Recycling "
            "and applying the listed value revisions, how many times do the lines intersect?"
        )

        operation_text, qa_question, split_info, data_change = _resolve_questions({"question": text}, llm)

        self.assertIn("Delete the categories Meridian Territory Sanitation and Silverhaven District Recycling.", operation_text)
        self.assertEqual(qa_question, "how many times do the lines intersect?")
        self.assertEqual(
            split_info["operation_target"],
            {"del_categories": ["Meridian Territory Sanitation", "Silverhaven District Recycling"]},
        )
        self.assertEqual(
            data_change["del"],
            {"category_names": ["Meridian Territory Sanitation", "Silverhaven District Recycling"]},
        )

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
            structured_context={"data_change": {"del": {"category_name": "Oracle Labs"}, "add": {"category_name": "Cinder Guild"}}},
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

    def test_operation_steps_expand_multi_change_into_atomic_steps(self) -> None:
        plan = {
            "normalized_question": "Apply the listed value revisions.",
            "steps": [
                {
                    "operation": "change",
                    "question_hint": "Apply the listed value revisions.",
                    "operation_target": {"category_names": ["Alpha", "Beta"]},
                    "data_change": {
                        "changes": [
                            {"category_name": "Alpha", "years": ["2020"], "values": [11]},
                            {"category_name": "Beta", "years": ["2021", "2022"], "values": [12, 13]},
                        ],
                    },
                    "new_points": [],
                }
            ],
            "new_points": [],
            "llm_success": True,
        }

        steps = _operation_steps_from_plan(plan, "Apply the listed value revisions.", {})

        self.assertEqual([step["operation"] for step in steps], ["change", "change", "change"])
        self.assertEqual(
            [step["operation_target"] for step in steps],
            [
                {"category_name": "Alpha"},
                {"category_name": "Beta"},
                {"category_name": "Beta"},
            ],
        )
        self.assertEqual(
            [step["data_change"]["changes"][0]["years"][0] for step in steps],
            ["2020", "2021", "2022"],
        )

    def test_operation_steps_expand_structured_changes_when_llm_has_single_change_step(self) -> None:
        plan = {
            "normalized_question": "Add Ironclad Ventures and apply the listed value revisions.",
            "steps": [
                {
                    "operation": "add",
                    "question_hint": "Insert a new area-series/category with the provided year-value pairs.",
                    "operation_target": {"category_name": "Ironclad Ventures"},
                    "data_change": {
                        "add": {
                            "category_name": "Ironclad Ventures",
                            "years": ["2015", "2016"],
                            "values": [11.86, 11.24],
                        }
                    },
                    "new_points": [],
                },
                {
                    "operation": "change",
                    "question_hint": "Revise the specified category-year values to the provided numbers.",
                    "operation_target": {},
                    "data_change": {},
                    "new_points": [],
                },
            ],
            "new_points": [],
            "llm_success": True,
        }
        structured_context = {
            "operation_target": {},
            "data_change": {
                "add": {
                    "category_name": "Ironclad Ventures",
                    "years": ["2015", "2016"],
                    "values": [11.86, 11.24],
                },
                "change": {
                    "changes": [
                        {
                            "category_name": "Horizon Enterprises",
                            "years": ["2023", "2024"],
                            "values": [12.78, 14.41],
                        },
                        {
                            "category_name": "Aegis Corp",
                            "years": ["2020"],
                            "values": [15.35],
                        },
                    ]
                },
            },
        }

        steps = _operation_steps_from_plan(
            plan,
            "Add Ironclad Ventures and apply the listed value revisions.",
            structured_context,
        )

        self.assertEqual([step["operation"] for step in steps], ["add", "change", "change", "change"])
        self.assertEqual(
            [step["operation_target"] for step in steps[1:]],
            [
                {"category_name": "Horizon Enterprises"},
                {"category_name": "Horizon Enterprises"},
                {"category_name": "Aegis Corp"},
            ],
        )
        self.assertEqual(
            [(step["data_change"]["changes"][0]["category_name"], step["data_change"]["changes"][0]["years"][0]) for step in steps[1:]],
            [
                ("Horizon Enterprises", "2023"),
                ("Horizon Enterprises", "2024"),
                ("Aegis Corp", "2020"),
            ],
        )

    def test_operation_steps_expand_nested_change_payload_from_llm_step(self) -> None:
        plan = {
            "normalized_question": "Apply the listed value revisions.",
            "steps": [
                {
                    "operation": "change",
                    "question_hint": "Revise the specified category-year values.",
                    "operation_target": {"categories": ["Horizon Enterprises", "Aegis Corp"]},
                    "data_change": {
                        "change": {
                            "changes": [
                                {
                                    "category_name": "Horizon Enterprises",
                                    "years": ["2023", "2024"],
                                    "values": [12.78, 14.41],
                                },
                                {
                                    "category_name": "Aegis Corp",
                                    "years": ["2020"],
                                    "values": [15.35],
                                },
                            ]
                        }
                    },
                    "new_points": [],
                }
            ],
            "new_points": [],
            "llm_success": True,
        }

        steps = _operation_steps_from_plan(plan, "Apply the listed value revisions.", {})

        self.assertEqual([step["operation"] for step in steps], ["change", "change", "change"])
        self.assertEqual(
            [(step["data_change"]["changes"][0]["category_name"], step["data_change"]["changes"][0]["years"][0]) for step in steps],
            [
                ("Horizon Enterprises", "2023"),
                ("Horizon Enterprises", "2024"),
                ("Aegis Corp", "2020"),
            ],
        )

    def test_operation_steps_fallback_preserves_missing_change_after_delete(self) -> None:
        operation_text = "Delete the category CrimsonLink and apply the listed value revisions."
        structured_context = {
            "operation_target": {"del_category": "CrimsonLink"},
            "data_change": {
                "del": {"category_name": "CrimsonLink"},
                "change": {
                    "changes": [
                        {"category_name": "Starburst", "years": [2020], "values": [12]},
                        {"category_name": "AetherNet", "years": [2021], "values": [10]},
                    ]
                },
            },
        }
        plan = {
            "normalized_question": operation_text,
            "steps": [
                {
                    "operation": "delete",
                    "question_hint": "Delete CrimsonLink.",
                    "operation_target": {"category_name": "CrimsonLink"},
                    "data_change": {},
                    "new_points": [],
                }
            ],
            "new_points": [],
            "llm_success": True,
        }

        steps = _operation_steps_from_plan(plan, operation_text, structured_context)

        self.assertEqual([step["operation"] for step in steps], ["delete", "change", "change"])
        self.assertEqual(steps[0]["operation_target"], {"category_name": "CrimsonLink"})
        self.assertEqual(
            [(step["data_change"]["changes"][0]["category_name"], step["data_change"]["changes"][0]["years"][0]) for step in steps[1:]],
            [("Starburst", "2020"), ("AetherNet", "2021")],
        )

    def test_operation_steps_expand_update_style_changes_into_atomic_steps(self) -> None:
        plan = {
            "normalized_question": "Apply the listed value revisions.",
            "steps": [
                {
                    "operation": "change",
                    "question_hint": "Apply the listed value revisions.",
                    "operation_target": {"category_name": "Alpha"},
                    "data_change": {
                        "changes": [
                            {
                                "category": "Alpha",
                                "updates": [
                                    {"year": "2020", "value": 11},
                                    {"year": "2021", "value": 12},
                                ],
                            }
                        ],
                    },
                    "new_points": [],
                }
            ],
            "new_points": [],
            "llm_success": True,
        }

        steps = _operation_steps_from_plan(plan, "Apply the listed value revisions.", {})

        self.assertEqual([step["operation_target"] for step in steps], [{"category_name": "Alpha"}, {"category_name": "Alpha"}])
        self.assertEqual(
            [step["data_change"]["changes"][0]["values"][0] for step in steps],
            [11, 12],
        )

    def test_operation_steps_expand_target_years_with_values_only_payload(self) -> None:
        plan = {
            "normalized_question": "Delete two lines and apply the listed value revisions.",
            "steps": [
                {
                    "operation": "delete",
                    "question_hint": "",
                    "operation_target": {"category_name": "MegaMall Chain"},
                    "data_change": {"del": {"category_name": "MegaMall Chain"}},
                    "new_points": [],
                },
                {
                    "operation": "delete",
                    "question_hint": "",
                    "operation_target": {"category_name": "Tech Gadget Hubs"},
                    "data_change": {"del": {"category_name": "Tech Gadget Hubs"}},
                    "new_points": [],
                },
                {
                    "operation": "change",
                    "question_hint": "",
                    "operation_target": {"category_name": "Fashion Boutiques", "years": ["1993", "1994"]},
                    "data_change": {"values": [17625.13, 17964.39]},
                    "new_points": [],
                },
                {
                    "operation": "change",
                    "question_hint": "",
                    "operation_target": {"category_name": "Neighborhood Stores", "years": ["1993", "1994"]},
                    "data_change": {"values": [22644.4, 15162.12]},
                    "new_points": [],
                },
            ],
            "new_points": [],
            "llm_success": True,
        }

        steps = _operation_steps_from_plan(plan, "Delete two lines and apply changes.", {})

        self.assertEqual([step["operation"] for step in steps], ["delete", "delete", "change", "change", "change", "change"])
        self.assertEqual(
            [(step["data_change"]["changes"][0]["category_name"], step["data_change"]["changes"][0]["years"][0]) for step in steps[2:]],
            [
                ("Fashion Boutiques", "1993"),
                ("Fashion Boutiques", "1994"),
                ("Neighborhood Stores", "1993"),
                ("Neighborhood Stores", "1994"),
            ],
        )

    def test_operation_steps_expand_nested_single_change_objects_per_step(self) -> None:
        plan = {
            "normalized_question": "Add one series and apply grouped changes.",
            "steps": [
                {
                    "operation": "add",
                    "question_hint": "",
                    "operation_target": {"add_category": "Crystal Cove Villas", "element_type": "area"},
                    "data_change": {
                        "add": {
                            "category_name": "Crystal Cove Villas",
                            "years": ["2015", "2016"],
                            "values": [3.64, 4.45],
                        }
                    },
                    "new_points": [],
                },
                {
                    "operation": "change",
                    "question_hint": "",
                    "operation_target": {"category_name": "Skyline Towers", "years": ["2023", "2024"]},
                    "data_change": {"change": {"category_name": "Skyline Towers", "years": ["2023", "2024"], "values": [3.77, 3.54]}},
                    "new_points": [],
                },
                {
                    "operation": "change",
                    "question_hint": "",
                    "operation_target": {"category_name": "Willow Creek Estates", "years": ["2022", "2023"]},
                    "data_change": {"change": {"category_name": "Willow Creek Estates", "years": ["2022", "2023"], "values": [2.91, 3.32]}},
                    "new_points": [],
                },
            ],
            "new_points": [],
            "llm_success": True,
        }

        steps = _operation_steps_from_plan(plan, "Add one series and apply grouped changes.", {})

        self.assertEqual([step["operation"] for step in steps], ["add", "change", "change", "change", "change"])
        self.assertEqual(
            [(step["data_change"]["changes"][0]["category_name"], step["data_change"]["changes"][0]["years"][0]) for step in steps[1:]],
            [
                ("Skyline Towers", "2023"),
                ("Skyline Towers", "2024"),
                ("Willow Creek Estates", "2022"),
                ("Willow Creek Estates", "2023"),
            ],
        )

    def test_operation_steps_drop_redundant_legend_item_add_for_same_category(self) -> None:
        plan = {
            "normalized_question": "Add the category The Golden Spoon.",
            "steps": [
                {
                    "operation": "add",
                    "question_hint": "",
                    "operation_target": {"type": "area", "category_name": "The Golden Spoon", "fill": "#8c564b"},
                    "data_change": {
                        "category_name": "The Golden Spoon",
                        "years": ["2015", "2016"],
                        "values": [68.42, 74.48],
                    },
                    "new_points": [],
                },
                {
                    "operation": "add",
                    "question_hint": "",
                    "operation_target": {"type": "legend_item", "category_name": "The Golden Spoon"},
                    "data_change": {"category_name": "The Golden Spoon", "color": "#8c564b"},
                    "new_points": [],
                },
            ],
            "new_points": [],
            "llm_success": True,
        }

        steps = _operation_steps_from_plan(plan, "Add the category The Golden Spoon.", {})

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["operation"], "add")
        self.assertEqual(steps[0]["operation_target"]["category_name"], "The Golden Spoon")
        self.assertEqual(steps[0]["operation_target"]["type"], "area")

    def test_operation_steps_keep_multiple_series_adds_for_different_categories(self) -> None:
        plan = {
            "normalized_question": "Add two categories.",
            "steps": [
                {
                    "operation": "add",
                    "question_hint": "",
                    "operation_target": {"type": "area", "category_name": "Alpha"},
                    "data_change": {"category_name": "Alpha", "years": ["2015"], "values": [1]},
                    "new_points": [],
                },
                {
                    "operation": "add",
                    "question_hint": "",
                    "operation_target": {"type": "area", "category_name": "Beta"},
                    "data_change": {"category_name": "Beta", "years": ["2015"], "values": [2]},
                    "new_points": [],
                },
                {
                    "operation": "add",
                    "question_hint": "",
                    "operation_target": {"type": "legend_item", "category_name": "Alpha"},
                    "data_change": {"category_name": "Alpha", "color": "#111111"},
                    "new_points": [],
                },
                {
                    "operation": "add",
                    "question_hint": "",
                    "operation_target": {"type": "legend_item", "category_name": "Beta"},
                    "data_change": {"category_name": "Beta", "color": "#222222"},
                    "new_points": [],
                },
            ],
            "new_points": [],
            "llm_success": True,
        }

        steps = _operation_steps_from_plan(plan, "Add two categories.", {})

        self.assertEqual([step["operation_target"]["category_name"] for step in steps], ["Alpha", "Beta"])


if __name__ == "__main__":
    unittest.main()
