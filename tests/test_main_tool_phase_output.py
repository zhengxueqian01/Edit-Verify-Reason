from __future__ import annotations

import unittest
from unittest.mock import patch

from main import (
    TOOL_AUG_CONFIDENCE_THRESHOLD,
    _apply_tool_augmented_answer,
    _should_answer_after_failed_render,
)


class MainToolPhaseOutputTests(unittest.TestCase):
    def test_tool_phase_threshold_constant_is_still_exposed_for_debugging(self) -> None:
        self.assertEqual(TOOL_AUG_CONFIDENCE_THRESHOLD, 0.85)

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


if __name__ == "__main__":
    unittest.main()
