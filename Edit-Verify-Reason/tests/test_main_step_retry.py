from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from main import _StepExecutionError, _execute_planned_steps, _operation_steps_from_rules_plan, run_main


class MainStepRetryTests(unittest.TestCase):
    def test_operation_steps_from_rules_plan_drops_unknown_leftover_when_structured_steps_exist(self) -> None:
        operation_plan = {
            "steps": [
                {
                    "operation": "unknown",
                    "question": "",
                    "question_hint": "",
                    "operation_target": {},
                    "data_change": {
                        "change": {
                            "changes": [
                                {"category_name": "Alpha", "years": ["2018"], "values": [10]},
                                {"category_name": "Beta", "years": ["2019"], "values": [11]},
                            ]
                        }
                    },
                    "new_points": [],
                }
            ]
        }
        structured_context = {
            "data_change": {
                "change": {
                    "changes": [
                        {"category_name": "Alpha", "years": ["2018"], "values": [10]},
                        {"category_name": "Beta", "years": ["2019"], "values": [11]},
                    ]
                }
            }
        }

        steps = _operation_steps_from_rules_plan(operation_plan, "Apply changes", structured_context)

        self.assertEqual([step["operation"] for step in steps], ["change", "change"])
        self.assertTrue(all(str(step.get("operation")) != "unknown" for step in steps))

    def test_execute_planned_steps_retries_only_current_step(self) -> None:
        inputs = {"svg_path": "/tmp/sample.svg"}
        operation_plan = {
            "steps": [
                {
                    "operation": "delete",
                    "question_hint": "Delete Alpha",
                    "operation_target": {"category_name": "Alpha"},
                    "data_change": {},
                    "new_points": [],
                },
                {
                    "operation": "change",
                    "question_hint": "Change Beta in 2020 to 10",
                    "operation_target": {"category_name": "Beta"},
                    "data_change": {
                        "changes": [
                            {"category_name": "Beta", "years": ["2020"], "values": [10]}
                        ]
                    },
                    "new_points": [],
                },
            ],
            "new_points": [],
            "llm_success": True,
        }
        retry_step = {
            "operation": "change",
            "question_hint": "Change Beta in 2020 to 11",
            "operation_target": {"category_name": "Beta"},
            "data_change": {
                "changes": [
                    {"category_name": "Beta", "years": ["2020"], "values": [11]}
                ]
            },
            "new_points": [],
        }
        perception_state = SimpleNamespace(
            perception={
                "mapping_info": {},
                "update_spec": {},
                "chart_type": "line",
                "issues": [],
            }
        )
        current_svg_calls: list[str] = []
        step_questions: list[str] = []

        def _update_line_side_effect(
            current_svg: str,
            question: str,
            mapping_info: dict[str, object],
            *,
            output_path: str,
            svg_output_path: str,
            llm: object,
            operation_target: dict[str, object] | None,
            data_change: dict[str, object] | None,
            operation: str,
        ) -> str:
            del mapping_info, llm, operation_target, data_change, operation
            current_svg_calls.append(current_svg)
            step_questions.append(question)
            if len(current_svg_calls) == 2:
                raise ValueError("No matching line series found in question.")
            return output_path

        with patch("main.run_perception", side_effect=[perception_state, perception_state, perception_state]):
            with patch("main.update_line_svg", side_effect=_update_line_side_effect) as update_mock:
                with patch("main._replan_current_step", return_value=retry_step) as replan_mock:
                    output_image, step_logs, perception_steps, _last_state, _points = _execute_planned_steps(
                        inputs=inputs,
                        planned_question="Delete Alpha and change Beta.",
                        operation_plan=operation_plan,
                        structured_context={},
                        chart_type="line",
                        render_output_dir="/tmp/render-case",
                        llm=object(),
                        planner_llm=object(),
                        used_scatter_points=[],
                    )

        self.assertEqual(update_mock.call_count, 3)
        self.assertEqual(len(step_logs), 2)
        self.assertEqual(len(perception_steps), 2)
        self.assertEqual(current_svg_calls[0], "/tmp/sample.svg")
        self.assertEqual(current_svg_calls[1], step_logs[0]["output_svg_path"])
        self.assertEqual(current_svg_calls[2], step_logs[0]["output_svg_path"])
        self.assertEqual(step_logs[0]["step_attempt"], 1)
        self.assertEqual(step_logs[1]["step_attempt"], 2)
        self.assertEqual(step_logs[1]["question"], 'Change "Beta" in 2020 to 11')
        self.assertEqual(step_questions[2], 'Change "Beta" in 2020 to 11')
        self.assertTrue(str(step_logs[0]["output_svg_path"]).startswith("/tmp/render-case/"))
        self.assertTrue(str(step_logs[1]["output_svg_path"]).startswith("/tmp/render-case/"))
        self.assertEqual(len(step_logs[1]["retry_history"]), 1)
        self.assertEqual(step_logs[1]["retry_history"][0]["failure_info"]["failure_type"], "target_not_found")
        self.assertEqual(perception_steps[1]["retry_history"][0]["failure_info"]["replan_strategy"], "replan_current_step")
        self.assertEqual(output_image, step_logs[-1]["output_image_path"])
        replan_mock.assert_called_once()
        self.assertEqual(replan_mock.call_args.kwargs["chart_type"], "line")
        self.assertIn("No matching line series found", replan_mock.call_args.kwargs["retry_hint"])

    def test_run_main_returns_failure_info_for_failed_step(self) -> None:
        mock_config = SimpleNamespace(model="mock-model")
        perception_state = SimpleNamespace(
            perception={
                "chart_type": "line",
                "mapping_info": {},
                "update_spec": {},
                "primitives_summary": {},
            },
            trace=[],
        )
        failure_info = {
            "failure_type": "chart_structure_missing",
            "failure_stage": "step_execute",
            "retryable": False,
            "retry_hint": "SVG axes group not found.",
            "replan_strategy": "restart_attempt",
            "message": "SVG axes group not found.",
            "exception_type": "ValueError",
            "chart_type": "line",
            "operation": "delete",
        }
        failed_step = {
            "index": 1,
            "step_attempt": 1,
            "operation": "delete",
            "question": 'Delete the category/series "Alpha"',
            "question_hint": "Delete Alpha",
            "operation_target": {"category_name": "Alpha"},
            "data_change": {},
            "input_svg_path": "/tmp/sample.svg",
            "failure_info": failure_info,
        }
        step_logs = [
            {
                "index": 1,
                "step_attempt": 1,
                "operation": "delete",
                "question": 'Delete the category/series "Alpha"',
                "output_svg_path": "output/line/sample_step1.svg",
                "output_image_path": "output/line/sample_step1.png",
                "retry_history": [],
            }
        ]
        perception_steps = [
            {
                "index": 1,
                "step_attempt": 1,
                "operation": "delete",
                "question": 'Delete the category/series "Alpha"',
                "question_hint": "Delete Alpha",
                "perception": {"chart_type": "line"},
            }
        ]
        step_error = _StepExecutionError(
            "step 1: SVG axes group not found.",
            step_logs=step_logs,
            perception_steps=perception_steps,
            last_state=perception_state,
            scatter_points=[],
            failure_info=failure_info,
            failed_step=failed_step,
        )

        with patch("main.resolve_task_model_config", return_value=mock_config):
            with patch("main.make_llm", return_value=object()):
                with patch(
                    "main._resolve_questions",
                    return_value=("Delete Alpha", "What is the max?", {"operation_text": "Delete Alpha"}, {}),
                ):
                    with patch("main.answer_question", return_value={"answer": "base", "confidence": 0.9}):
                        with patch("main.run_perception", return_value=perception_state):
                            with patch(
                                "main._llm_plan_update",
                                return_value={
                                    "normalized_question": "Delete Alpha",
                                    "steps": [{"operation": "delete", "question_hint": "Delete Alpha"}],
                                    "new_points": [],
                                    "llm_success": True,
                                },
                            ):
                                with patch("main._execute_planned_steps", side_effect=step_error):
                                    with patch("main._should_answer_after_failed_render", return_value=False):
                                        result = run_main(
                                            {
                                                "question": "Delete Alpha then answer the question.",
                                                "image_path": "/tmp/sample.png",
                                                "svg_path": "/tmp/sample.svg",
                                                "max_render_retries": 0,
                                            }
                                        )

        self.assertEqual(result["render_check"]["failure_info"]["failure_type"], "chart_structure_missing")
        self.assertFalse(result["render_check"]["failure_info"]["retryable"])
        self.assertEqual(result["attempt_logs"][0]["failure_info"]["failure_stage"], "step_execute")
        self.assertEqual(result["attempt_logs"][0]["failed_step"]["failure_info"]["replan_strategy"], "restart_attempt")
        self.assertEqual(result["perception_steps"], perception_steps)
        self.assertIn("render_validation_failed", result["answer"]["issues"])


if __name__ == "__main__":
    unittest.main()
