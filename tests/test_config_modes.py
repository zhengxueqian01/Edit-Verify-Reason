from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from chart_agent.config import get_svg_perception_mode, get_svg_update_mode


class ConfigModesTests(unittest.TestCase):
    def test_svg_update_mode_defaults_to_rules(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SVG_UPDATE_MODE", None)
            self.assertEqual(get_svg_update_mode(), "rules")

    def test_svg_update_mode_accepts_llm_alias(self) -> None:
        with patch.dict(os.environ, {"SVG_UPDATE_MODE": "llm"}, clear=False):
            self.assertEqual(get_svg_update_mode(), "llm_intent")

    def test_svg_perception_mode_accepts_llm_alias(self) -> None:
        with patch.dict(os.environ, {"SVG_PERCEPTION_MODE": "llm"}, clear=False):
            self.assertEqual(get_svg_perception_mode(), "llm_summary")

    def test_svg_update_mode_prefers_explicit_override(self) -> None:
        with patch.dict(os.environ, {"SVG_UPDATE_MODE": "rules"}, clear=False):
            self.assertEqual(get_svg_update_mode("llm"), "llm_intent")

    def test_svg_perception_mode_prefers_explicit_override(self) -> None:
        with patch.dict(os.environ, {"SVG_PERCEPTION_MODE": "rules"}, clear=False):
            self.assertEqual(get_svg_perception_mode("llm"), "llm_summary")


if __name__ == "__main__":
    unittest.main()
