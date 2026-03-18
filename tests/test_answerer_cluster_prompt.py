from __future__ import annotations

import json
import unittest

from chart_agent.core.answerer import _cluster_prompt_block, answer_question


class _PromptCaptureLLM:
    def __init__(self) -> None:
        self.prompt = ""

    def invoke(self, prompt: str) -> object:
        self.prompt = prompt
        return type("Resp", (), {"content": json.dumps({"answer": "2", "confidence": 0.9, "reason": []})})()


class AnswererClusterPromptTests(unittest.TestCase):
    def test_cluster_prompt_block_includes_dbscan_params(self) -> None:
        block = _cluster_prompt_block(
            {
                "algorithm": "DBSCAN",
                "mode": "per_color",
                "eps": 6.0,
                "min_samples": 3,
                "source": "qa_question_suffix",
            }
        )

        self.assertIn('"eps": 6.0', block)
        self.assertIn('"min_samples": 3', block)
        self.assertIn('"mode": "per_color"', block)

    def test_answer_prompt_includes_image_context_and_qa_only_instruction(self) -> None:
        llm = _PromptCaptureLLM()

        answer_question(
            qa_question="How many intersections are there?",
            chart_type="line",
            data_summary={},
            output_image_path="output/line/example.png",
            image_context_note="The requested chart update has already been applied.",
            llm=llm,
        )

        self.assertIn("Use the provided image to answer the QA question only.", llm.prompt)
        self.assertIn("Image context: The requested chart update has already been applied.", llm.prompt)
        self.assertIn("QA Question: How many intersections are there?", llm.prompt)


if __name__ == "__main__":
    unittest.main()
