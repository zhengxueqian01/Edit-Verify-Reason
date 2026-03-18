from __future__ import annotations

import unittest

from chart_agent.perception.render_validator import _soften_uncertainty_only_failure


class RenderValidatorTests(unittest.TestCase):
    def test_uncertainty_only_line_failure_is_softened(self) -> None:
        result = _soften_uncertainty_only_failure(
            chart_type="line",
            llm_ok=False,
            issues=[
                "Cannot verify the exact values because the chart shows no data labels.",
                "The image alone is insufficient to confirm the precise point update.",
            ],
            confidence=0.2,
        )

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["confidence"], 0.45)

    def test_structural_render_failure_is_not_softened(self) -> None:
        result = _soften_uncertainty_only_failure(
            chart_type="line",
            llm_ok=False,
            issues=[
                "The requested new series is not visible in the legend.",
                "No new line is present in the chart.",
            ],
            confidence=0.2,
        )

        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
