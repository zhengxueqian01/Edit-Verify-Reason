from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from chart_agent.perception.scatter_svg_updater import update_scatter_svg


class ScatterSvgUpdaterTests(unittest.TestCase):
    def test_update_scatter_svg_preserves_per_point_colors_without_llm(self) -> None:
        svg_text = """
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
          <g id="axes_1">
            <g id="PathCollection_1">
              <g clip-path="url(#clip-red)">
                <use xlink:href="#red-marker" x="10" y="10" style="fill: #d62728; fill-opacity: 0.82; stroke: #ffffff" />
              </g>
            </g>
            <g id="PathCollection_2">
              <g clip-path="url(#clip-orange)">
                <use xlink:href="#orange-marker" x="20" y="20" style="fill: #ff7f0e; fill-opacity: 0.82; stroke: #ffffff" />
              </g>
            </g>
          </g>
        </svg>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "input.svg"
            out_svg = Path(tmpdir) / "output.svg"
            out_png = Path(tmpdir) / "output.png"
            svg_path.write_text(svg_text, encoding="utf-8")

            with patch("chart_agent.perception.scatter_svg_updater.render_svg_to_png", return_value=str(out_png)):
                update_scatter_svg(
                    str(svg_path),
                    [
                        {"x": 1.0, "y": 1.0, "color": "orange"},
                        {"x": 2.0, "y": 2.0, "color": "red"},
                    ],
                    {
                        "x_ticks": [(10.0, 0.0), (110.0, 10.0)],
                        "y_ticks": [(10.0, 0.0), (110.0, 10.0)],
                        "existing_point_colors": ["#d62728", "#ff7f0e"],
                    },
                    output_path=str(out_png),
                    svg_output_path=str(out_svg),
                )

            root = ET.parse(out_svg).getroot()
            ns = {"svg": "http://www.w3.org/2000/svg"}
            uses = root.findall(".//svg:g[@id='PathCollection_update']//svg:use", ns)
            styles = [str(use.get("style") or "") for use in uses]

            self.assertEqual(len(styles), 2)
            self.assertTrue(any("fill: #ff7f0e" in style for style in styles))
            self.assertTrue(any("fill: #d62728" in style for style in styles))


if __name__ == "__main__":
    unittest.main()
