from __future__ import annotations

import json
import unittest
from typing import Any

from chart_agent.core.answerer import ANSWER_SYSTEM_PROMPT, answer_question
from chart_agent.prompts.prompt import (
    ANSWER_TOOL_AUGMENTED_AREA_SYSTEM_PROMPT,
    ANSWER_TOOL_AUGMENTED_LINE_SYSTEM_PROMPT,
    ANSWER_TOOL_AUGMENTED_SCATTER_SYSTEM_PROMPT,
    ANSWER_UPDATED_AREA_SYSTEM_PROMPT,
    ANSWER_UPDATED_LINE_SYSTEM_PROMPT,
    ANSWER_UPDATED_SCATTER_SYSTEM_PROMPT,
)


class _PromptCaptureLLM:
    def __init__(self) -> None:
        self.prompt: Any = None

    def invoke(self, prompt: Any) -> object:
        self.prompt = prompt
        return type("Resp", (), {"content": json.dumps({"answer": "2", "confidence": 0.9, "reason": []})})()


class AnswererClusterPromptTests(unittest.TestCase):
    def test_answer_prompt_includes_image_context_and_qa_only_instruction(self) -> None:
        llm = _PromptCaptureLLM()

        answer_question(
            qa_question="How many intersections are there?",
            data_summary={},
            chart_type="line",
            output_image_path="output/line/example.png",
            answer_stage="updated",
            image_context_note="The requested chart update has already been applied.",
            llm=llm,
        )

        self.assertIsInstance(llm.prompt, list)
        self.assertEqual(2, len(llm.prompt))
        self.assertIn(ANSWER_SYSTEM_PROMPT, llm.prompt[0].content)
        self.assertIn(ANSWER_UPDATED_LINE_SYSTEM_PROMPT, llm.prompt[0].content)
        self.assertIn("Image context: The requested chart update has already been applied.", llm.prompt[0].content)
        self.assertNotIn("Never merge points of different colors/categories into the same cluster", llm.prompt[0].content)

        human_content = llm.prompt[1].content
        self.assertIsInstance(human_content, str)
        self.assertIn("Input: How many intersections are there?", human_content)
        self.assertNotIn("Image context:", human_content)
        self.assertNotIn("QA Question:", human_content)
        self.assertNotIn("Chart type:", human_content)
        self.assertNotIn("Image path", human_content)
        self.assertNotIn("Cluster Counting Rule:", human_content)
        self.assertNotIn("Cluster Parameters:", human_content)

    def test_scatter_cluster_prompt_uses_specialized_cluster_rules(self) -> None:
        llm = _PromptCaptureLLM()

        answer_question(
            qa_question="After adding these points, how many clusters are there now?",
            data_summary={"mapping_info_summary": {"num_points": 20, "num_lines": 0, "num_areas": 0}},
            chart_type="scatter",
            output_image_path="output/scatter/example.png",
            answer_stage="updated",
            image_context_note="The requested chart update has already been applied.",
            llm=llm,
        )

        self.assertIsInstance(llm.prompt, list)
        self.assertEqual(2, len(llm.prompt))
        self.assertIn(ANSWER_SYSTEM_PROMPT, llm.prompt[0].content)
        self.assertIn(ANSWER_UPDATED_SCATTER_SYSTEM_PROMPT, llm.prompt[0].content)
        self.assertIn("Never merge points of different colors/categories into the same cluster", llm.prompt[0].content)

    def test_area_prompt_uses_specialized_area_rules(self) -> None:
        llm = _PromptCaptureLLM()

        answer_question(
            qa_question="Which year has the highest total value?",
            data_summary={"mapping_info_summary": {"num_points": 0, "num_lines": 0, "num_areas": 3}},
            chart_type="area",
            output_image_path="output/area/example.png",
            answer_stage="updated",
            image_context_note="The requested chart update has already been applied.",
            llm=llm,
        )

        self.assertIsInstance(llm.prompt, list)
        self.assertEqual(2, len(llm.prompt))
        self.assertIn(ANSWER_SYSTEM_PROMPT, llm.prompt[0].content)
        self.assertIn(ANSWER_UPDATED_AREA_SYSTEM_PROMPT, llm.prompt[0].content)

    def test_tool_augmented_line_prompt_uses_specialized_stage_rules(self) -> None:
        llm = _PromptCaptureLLM()

        answer_question(
            qa_question="How many intersections are there?",
            data_summary={},
            chart_type="line",
            output_image_path="output/line/example_tool.png",
            answer_stage="tool_augmented",
            image_context_note="The chart has visual augmentation.",
            llm=llm,
        )

        self.assertIn(ANSWER_TOOL_AUGMENTED_LINE_SYSTEM_PROMPT, llm.prompt[0].content)
        self.assertIn("unrelated lines may be faded", llm.prompt[0].content)

    def test_tool_augmented_area_prompt_uses_specialized_stage_rules(self) -> None:
        llm = _PromptCaptureLLM()

        answer_question(
            qa_question="Which year has the highest total value?",
            data_summary={"mapping_info_summary": {"num_points": 0, "num_lines": 0, "num_areas": 3}},
            chart_type="area",
            output_image_path="output/area/example_tool.png",
            answer_stage="tool_augmented",
            image_context_note="The chart has visual augmentation.",
            llm=llm,
        )

        self.assertIn(ANSWER_TOOL_AUGMENTED_AREA_SYSTEM_PROMPT, llm.prompt[0].content)
        self.assertIn("top boundary may be highlighted", llm.prompt[0].content)

    def test_tool_augmented_scatter_prompt_uses_specialized_stage_rules(self) -> None:
        llm = _PromptCaptureLLM()

        answer_question(
            qa_question="After adding these points, how many clusters are there now?",
            data_summary={"mapping_info_summary": {"num_points": 20, "num_lines": 0, "num_areas": 0}},
            chart_type="scatter",
            output_image_path="output/scatter/example_tool.png",
            answer_stage="tool_augmented",
            image_context_note="The chart has visual augmentation.",
            llm=llm,
        )

        self.assertIn(ANSWER_TOOL_AUGMENTED_SCATTER_SYSTEM_PROMPT, llm.prompt[0].content)


if __name__ == "__main__":
    unittest.main()
