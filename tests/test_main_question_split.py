from __future__ import annotations

import json
import unittest

from main import _llm_split_update_and_qa, _normalize_gerund_clause


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
                    "update_question": "1. Delete the category CrimsonLink; 2. Apply the listed value revisions.",
                    "qa_question": "How many times do the lines for Starburst and AetherNet intersect?",
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
        self.assertIn("Never use gerunds like 'adding', 'deleting', 'applying'", llm.prompt)


if __name__ == "__main__":
    unittest.main()
