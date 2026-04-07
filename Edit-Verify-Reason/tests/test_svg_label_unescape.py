from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from chart_agent.perception.area_svg_updater import _extract_text_label as extract_area_text_label
from chart_agent.perception.line_svg_updater import _extract_text_label as extract_line_text_label


class SvgLabelUnescapeTests(unittest.TestCase):
    def test_extract_text_label_unescapes_html_entities(self) -> None:
        group = ET.fromstring('<g xmlns="http://www.w3.org/2000/svg" id="text_23"></g>')
        content = """
        <g id="text_23">
          <!-- Research &amp; Development -->
        </g>
        """

        area_label = extract_area_text_label(group, content)
        line_label = extract_line_text_label(group, content)

        self.assertEqual(area_label, "Research & Development")
        self.assertEqual(line_label, "Research & Development")


if __name__ == "__main__":
    unittest.main()
