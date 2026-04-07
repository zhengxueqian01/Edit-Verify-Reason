from __future__ import annotations

import unittest
from unittest.mock import patch

from main import (
    TOOL_AUG_CONFIDENCE_THRESHOLD,
    _apply_tool_augmented_answer,
    _normalize_experiment_mode,
    _should_answer_after_failed_render,
    run_main,
)


class MainToolPhaseOutputTests(unittest.TestCase):
    def test_tool_phase_threshold_constant_is_still_exposed_for_debugging(self) -> None:
        self.assertEqual(TOOL_AUG_CONFIDENCE_THRESHOLD, 0.85)

    def test_unknown_experiment_mode_defaults_to_full(self) -> None:
        self.assertEqual(_normalize_experiment_mode("bad_mode"), "full")

    def test_tool_augmented_answer_does_not_overwrite_base_output_image(self) -> None:
        output = {
            "output_image_path": "output/scatter/000_scatter_update_updated.png",
            "answer": {"answer": "base", "confidence": 0.8},
        }

        with patch("main.answer_question", return_value={"answer": "tool", "confidence": 0.9}) as answer_mock:
            _apply_tool_augmented_answer(
                output=output,
                qa_question="How many clusters?",
                answer_data_summary={"cluster_result": {}},
                tool_phase={
                    "ok": True,
                    "augmented_image_path": "output/scatter/000_scatter_update_updated_tool_aug.png",
                },
                answer_llm=object(),
            )

        self.assertEqual(output["output_image_path"], "output/scatter/000_scatter_update_updated.png")
        self.assertEqual(
            output["tool_augmented_image_path"],
            "output/scatter/000_scatter_update_updated_tool_aug.png",
        )
        self.assertEqual(output["answer_tool_augmented"]["answer"], "tool")
        self.assertEqual(output["answer"]["answer"], "tool")
        answer_mock.assert_called_once()
        self.assertEqual(answer_mock.call_args.kwargs["qa_question"], "How many clusters?")
        self.assertIn("visual augmentation has also been added", answer_mock.call_args.kwargs["image_context_note"])

    def test_tool_augmented_answer_skips_when_not_ok(self) -> None:
        output = {
            "output_image_path": "output/scatter/000_scatter_update_updated.png",
            "answer": {"answer": "base", "confidence": 0.8},
        }

        with patch("main.answer_question") as answer_mock:
            _apply_tool_augmented_answer(
                output=output,
                qa_question="How many clusters?",
                answer_data_summary={},
                tool_phase={"ok": False, "augmented_image_path": "output/scatter/000_tool_aug.png"},
                answer_llm=object(),
            )

        self.assertEqual(output["output_image_path"], "output/scatter/000_scatter_update_updated.png")
        self.assertEqual(output["tool_augmented_image_path"], "output/scatter/000_tool_aug.png")
        self.assertIsNone(output["answer_tool_augmented"])
        answer_mock.assert_not_called()

    def test_scatter_cluster_can_answer_after_failed_render_when_image_exists(self) -> None:
        allowed = _should_answer_after_failed_render(
            chart_type="scatter",
            structured_context={},
            qa_question="How many clusters are there now?",
            output_image_path="output/scatter/001_scatter_add_updated.png",
        )

        self.assertTrue(allowed)

    def test_non_scatter_can_answer_after_three_render_blocks_when_image_exists(self) -> None:
        with patch("main.os.path.exists", return_value=True):
            allowed = _should_answer_after_failed_render(
                chart_type="line",
                structured_context={},
                qa_question="What is the maximum value?",
                output_image_path="output/line/001_line_updated.png",
                attempt_logs=[
                    {
                        "output_image_path": "output/line/001_line_updated.png",
                        "render_check": {"ok": False, "issues": ["cannot verify exact value"]},
                    },
                    {
                        "output_image_path": "output/line/001_line_updated.png",
                        "render_check": {"ok": False, "issues": ["cannot verify exact value"]},
                    },
                    {
                        "output_image_path": "output/line/001_line_updated.png",
                        "render_check": {"ok": False, "issues": ["cannot verify exact value"]},
                    },
                ],
                max_render_retries=2,
            )

        self.assertTrue(allowed)

    def test_non_scatter_still_blocks_without_rendered_image(self) -> None:
        with patch("main.os.path.exists", return_value=False):
            allowed = _should_answer_after_failed_render(
                chart_type="line",
                structured_context={},
                qa_question="What is the maximum value?",
                output_image_path="output/line/001_line_updated.png",
                attempt_logs=[
                    {
                        "output_image_path": "output/line/001_line_updated.png",
                        "render_check": {"ok": False, "issues": ["cannot verify exact value"]},
                    },
                    {
                        "output_image_path": "output/line/001_line_updated.png",
                        "render_check": {"ok": False, "issues": ["cannot verify exact value"]},
                    },
                    {
                        "output_image_path": "output/line/001_line_updated.png",
                        "render_check": {"ok": False, "issues": ["cannot verify exact value"]},
                    },
                ],
                max_render_retries=2,
            )

        self.assertFalse(allowed)

    def test_wo_svg_update_answers_on_original_image_with_reasoning_context(self) -> None:
        model_config = type("Cfg", (), {"model": "gpt"})
        state = type(
            "State",
            (),
            {
                "trace": [],
                "perception": {
                    "chart_type": "line",
                    "primitives_summary": {"num_points": 0, "num_lines": 2, "num_areas": 0},
                    "update_spec": {},
                    "mapping_info": {},
                },
            },
        )()

        with (
            patch("main.resolve_task_model_config", return_value=model_config),
            patch("main.make_llm", return_value=object()),
            patch(
                "main._resolve_questions",
                return_value=("Delete one series.", "Which year has the maximum value?", {"operation_text": "Delete one series."}, {}),
            ),
            patch("main._resolve_original_image_path", return_value="/tmp/original.png"),
            patch("main.run_perception", return_value=state),
            patch("main._resolve_supported_chart_type", return_value="line"),
            patch("main._llm_plan_update", return_value={"normalized_question": "Delete one series.", "steps": [], "new_points": [], "llm_success": True}),
            patch("main._maybe_apply_llm_intent_steps", side_effect=lambda **kwargs: kwargs["operation_plan"]),
            patch(
                "main._operation_steps_from_plan",
                return_value=[{"operation": "delete", "question_hint": "Delete one series."}],
            ),
            patch("main.answer_question", return_value={"answer": "2024", "confidence": 0.4}) as answer_mock,
        ):
            output = run_main(
                {
                    "question": "Delete one series and tell me which year has the maximum value",
                    "image_path": "/tmp/original.png",
                    "svg_path": "/tmp/input.svg",
                    "experiment_mode": "wo_svg_update",
                }
            )

        self.assertEqual(output["experiment_mode"], "wo_svg_update")
        self.assertEqual(output["output_image_path"], "/tmp/original.png")
        self.assertEqual(output["answer_input"]["question"], "Delete one series and tell me which year has the maximum value")
        self.assertEqual(output["render_check"]["issues"], ["skipped:wo_svg_update"])
        self.assertIsNone(output["answer_tool_augmented"])
        self.assertIn("unexecuted_update_reasoning", output["answer_input"]["data_summary"])
        self.assertEqual(
            answer_mock.call_args.kwargs["qa_question"],
            "Delete one series and tell me which year has the maximum value",
        )


if __name__ == "__main__":
    unittest.main()
