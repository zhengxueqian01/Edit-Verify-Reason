from __future__ import annotations

import json
import unittest
from typing import Any

from chart_agent.core.answerer import ANSWER_SYSTEM_PROMPT, answer_question


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
            output_image_path="output/line/example.png",
            image_context_note="The requested chart update has already been applied.",
            llm=llm,
        )

        self.assertIsInstance(llm.prompt, list)
        self.assertEqual(2, len(llm.prompt))
        self.assertIn(ANSWER_SYSTEM_PROMPT, llm.prompt[0].content)
        self.assertIn("Image context: The requested chart update has already been applied.", llm.prompt[0].content)
        self.assertIn("For cluster-counting questions, follow the clustering rule stated in the question.", llm.prompt[0].content)
        self.assertIn("same category that are connected by enough intermediate points", llm.prompt[0].content)
        self.assertIn("Do not treat each color/category as one cluster by default.", llm.prompt[0].content)
        self.assertIn("judge the number of clusters by their spatial proximity, separation, and density", llm.prompt[0].content)
        self.assertIn("same color can still form multiple clusters if they are spatially separated", llm.prompt[0].content)

        human_content = llm.prompt[1].content
        self.assertIsInstance(human_content, str)
        self.assertIn("Input: How many intersections are there?", human_content)
        self.assertNotIn("Image context:", human_content)
        self.assertNotIn("QA Question:", human_content)
        self.assertNotIn("Chart type:", human_content)
        self.assertNotIn("Image path", human_content)
        self.assertNotIn("Cluster Counting Rule:", human_content)
        self.assertNotIn("Cluster Parameters:", human_content)


if __name__ == "__main__":
    unittest.main()
