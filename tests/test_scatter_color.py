from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from main import _extract_scatter_requested_color, _resolve_supported_chart_type, _structured_delete_labels
from chart_agent.perception.question_parser import parse_question
from chart_agent.perception.scatter_svg_updater import _point_requested_color, _resolve_new_point_style, _select_marker_info


class ScatterColorTests(unittest.TestCase):
    def test_parse_question_extracts_named_color(self) -> None:
        update_spec, issues = parse_question("在 scatter 里加一个蓝色点 (3, 4)", llm=None, chart_type_hint="scatter")
        self.assertFalse(issues)
        self.assertEqual(update_spec["point_color"], "blue")

    def test_parse_question_extracts_hex_color(self) -> None:
        update_spec, _ = parse_question("Add a point at (3, 4) with color #D62728", llm=None, chart_type_hint="scatter")
        self.assertEqual(update_spec["point_color"], "#d62728")

    def test_resolve_new_point_style_prefers_existing_palette_match(self) -> None:
        fill, style = _resolve_new_point_style(
            requested_color="red",
            existing_colors=["#1f77b4", "#d62728", "#d62728"],
            template_styles_by_fill={"#d62728": "fill: #d62728; fill-opacity: 0.82; stroke: #ffffff"},
        )
        self.assertEqual(fill, "#d62728")
        self.assertIn("fill: #d62728", style)
        self.assertIn("stroke: #ffffff", style)

    def test_resolve_new_point_style_falls_back_to_dominant_existing_color(self) -> None:
        fill, style = _resolve_new_point_style(
            requested_color="",
            existing_colors=["#1f77b4", "#1f77b4", "#d62728"],
            template_styles_by_fill={"#1f77b4": "fill: #1f77b4; stroke: #ffffff"},
        )
        self.assertEqual(fill, "#1f77b4")
        self.assertIn("fill: #1f77b4", style)

    def test_extract_scatter_requested_color_prefers_data_change(self) -> None:
        color = _extract_scatter_requested_color(
            step={
                "operation_target": {},
                "data_change": {
                    "points": [{"x": 1, "y": 2}],
                    "color": "purple",
                },
            },
            update_spec={"point_color": "blue"},
        )
        self.assertEqual(color, "purple")

    def test_resolve_new_point_style_maps_purple_to_existing_svg_rgb(self) -> None:
        fill, style = _resolve_new_point_style(
            requested_color="purple",
            existing_colors=["#9467bd", "#9467bd", "#1f77b4"],
            template_styles_by_fill={"#9467bd": "fill: #9467bd; fill-opacity: 0.82; stroke: #ffffff"},
        )
        self.assertEqual(fill, "#9467bd")
        self.assertIn("fill: #9467bd", style)

    def test_select_marker_info_uses_matching_collection_in_multi_color_svg(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
              <g id="axes_1">
                <g id="PathCollection_1">
                  <g clip-path="url(#clip1)">
                    <use xlink:href="#green-marker" x="1" y="2" style="fill: #2ca02c; fill-opacity: 0.82; stroke: #ffffff" />
                  </g>
                </g>
                <g id="PathCollection_3">
                  <g clip-path="url(#clip3)">
                    <use xlink:href="#purple-marker" x="3" y="4" style="fill: #9467bd; fill-opacity: 0.82; stroke: #ffffff" />
                  </g>
                </g>
              </g>
            </svg>
            """
        )
        axes = root.find('.//{http://www.w3.org/2000/svg}g[@id="axes_1"]')
        self.assertIsNotNone(axes)
        href, clip_path, fill, style = _select_marker_info(
            axes=axes,
            requested_color="purple",
            existing_colors=["#2ca02c", "#9467bd"],
        )
        self.assertEqual(href, "#purple-marker")
        self.assertEqual(clip_path, "url(#clip3)")
        self.assertEqual(fill, "#9467bd")
        self.assertIn("fill: #9467bd", style)

    def test_point_requested_color_prefers_per_point_color(self) -> None:
        color = _point_requested_color({"x": 1, "y": 2, "color": "orange"})
        self.assertEqual(color, "orange")

    def test_resolve_supported_chart_type_prefers_explicit_scatter_hint(self) -> None:
        chart_type = _resolve_supported_chart_type(
            {
                "chart_type": "line",
                "primitives_summary": {
                    "num_lines": 11,
                    "num_points": 139,
                },
            },
            "scatter",
        )
        self.assertEqual(chart_type, "scatter")

    def test_structured_delete_labels_supports_del_category(self) -> None:
        labels = _structured_delete_labels({"del_category": "Loyalty Member Order"})
        self.assertEqual(labels, ["Loyalty Member Order"])

    def test_structured_delete_labels_supports_del_categories(self) -> None:
        labels = _structured_delete_labels({"del_categories": ["A", "B"]})
        self.assertEqual(labels, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
