from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from chart_agent.perception.svg_matcher import (
    _line_style_similarity,
    _extract_scatter_points,
    _numeric_list_similarity,
    _polyline_similarity_relaxed,
    compare_svgs,
)


class SvgMatcherTests(unittest.TestCase):
    def test_relaxed_polyline_similarity_ignores_constant_x_shift(self) -> None:
        left = [(10.0, 100.0), (20.0, 80.0), (30.0, 60.0), (40.0, 50.0)]
        right = [(15.0, 100.0), (25.0, 80.0), (35.0, 60.0), (45.0, 50.0)]

        score = _polyline_similarity_relaxed(left, right)

        self.assertGreaterEqual(score, 0.99)

    def test_numeric_list_similarity_keeps_high_score_for_close_y_ticks(self) -> None:
        left = [10.0, 20.0, 30.0, 40.0, 50.0]
        right = [10.0, 20.0, 30.0, 40.0, 50.0]

        score = _numeric_list_similarity(left, right)

        self.assertGreaterEqual(score, 0.99)

    def test_extract_scatter_points_reads_all_pathcollection_groups(self) -> None:
        svg = """\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <g id="axes_1">
    <g id="PathCollection_1">
      <use xlink:href="#p1" x="10" y="20" />
    </g>
    <g id="PathCollection_2">
      <use xlink:href="#p2" x="30" y="40" />
    </g>
    <g id="PathCollection_update">
      <circle cx="50" cy="60" />
    </g>
  </g>
</svg>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "scatter.svg"
            svg_path.write_text(svg, encoding="utf-8")
            root = ET.parse(svg_path).getroot()

        points = _extract_scatter_points(root)

        self.assertEqual(
            points,
            [
                {"x": 10.0, "y": 20.0, "fill": ""},
                {"x": 30.0, "y": 40.0, "fill": ""},
                {"x": 50.0, "y": 60.0, "fill": ""},
            ],
        )

    def test_line_style_similarity_ignores_color(self) -> None:
        score = _line_style_similarity(
            {
                "stroke_width": 2.0,
                "stroke_dasharray": (),
                "stroke_linecap": "round",
                "stroke_linejoin": "miter",
                "has_markers": False,
            },
            {
                "stroke_width": 2.0,
                "stroke_dasharray": (),
                "stroke_linecap": "round",
                "stroke_linejoin": "miter",
                "has_markers": False,
            },
        )

        self.assertGreaterEqual(score, 0.99)

    def test_compare_svgs_line_style_penalizes_dash_difference(self) -> None:
        pred_svg = """\
<svg xmlns="http://www.w3.org/2000/svg">
  <g id="axes_1">
    <g id="line2d_1">
      <path d="M 10 30 L 20 20 L 30 10" style="fill: none; stroke: #ff0000; stroke-width: 2; stroke-linecap: round" />
    </g>
  </g>
  <g id="matplotlib.axis_2">
    <g id="ytick_1"><g id="text_1"><!-- 10 --></g></g>
    <g id="ytick_2"><g id="text_2"><!-- 20 --></g></g>
  </g>
  <g id="legend_1">
    <g id="text_3"><!-- Revenue --></g>
  </g>
</svg>
"""
        gt_svg = """\
<svg xmlns="http://www.w3.org/2000/svg">
  <g id="axes_1">
    <g id="line2d_1">
      <path d="M 10 30 L 20 20 L 30 10" style="fill: none; stroke: #0000ff; stroke-width: 2; stroke-linecap: round; stroke-dasharray: 6,2" />
    </g>
  </g>
  <g id="matplotlib.axis_2">
    <g id="ytick_1"><g id="text_1"><!-- 10 --></g></g>
    <g id="ytick_2"><g id="text_2"><!-- 20 --></g></g>
  </g>
  <g id="legend_1">
    <g id="text_3"><!-- Revenue --></g>
  </g>
</svg>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_path = Path(tmpdir) / "pred.svg"
            gt_path = Path(tmpdir) / "gt.svg"
            pred_path.write_text(pred_svg, encoding="utf-8")
            gt_path.write_text(gt_svg, encoding="utf-8")

            result = compare_svgs(pred_path, gt_path)

        self.assertEqual(result["chart_type"], "line")
        self.assertLess(result["metrics"]["style_score"], 0.9)
        self.assertLess(result["score"], 0.99)

    def test_compare_svgs_area_penalizes_legend_label_mismatch(self) -> None:
        pred_svg = """\
<svg xmlns="http://www.w3.org/2000/svg">
  <g id="axes_1">
    <g id="FillBetweenPolyCollection_1">
      <path d="M 10 40 L 20 30 L 30 20 L 30 60 L 20 60 L 10 60" />
    </g>
  </g>
  <g id="legend_1">
    <g id="text_1"><!-- Series A --></g>
  </g>
</svg>
"""
        gt_svg = """\
<svg xmlns="http://www.w3.org/2000/svg">
  <g id="axes_1">
    <g id="FillBetweenPolyCollection_1">
      <path d="M 10 40 L 20 30 L 30 20 L 30 60 L 20 60 L 10 60" />
    </g>
  </g>
  <g id="legend_1">
    <g id="text_1"><!-- Series B --></g>
  </g>
</svg>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_path = Path(tmpdir) / "pred.svg"
            gt_path = Path(tmpdir) / "gt.svg"
            pred_path.write_text(pred_svg, encoding="utf-8")
            gt_path.write_text(gt_svg, encoding="utf-8")

            result = compare_svgs(pred_path, gt_path)

        self.assertEqual(result["chart_type"], "area")
        self.assertEqual(result["metrics"]["top_boundary_similarity"], 1.0)
        self.assertEqual(result["metrics"]["label_score"], 0.0)
        self.assertLess(result["score"], 1.0)

    def test_compare_svgs_scatter_penalizes_color_mismatch(self) -> None:
        pred_svg = """\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <g id="axes_1">
    <g id="PathCollection_1">
      <g><use xlink:href="#p1" x="10" y="20" style="fill: #ff0000; stroke: #ffffff" /></g>
      <g><use xlink:href="#p2" x="30" y="40" style="fill: #00ff00; stroke: #ffffff" /></g>
    </g>
  </g>
</svg>
"""
        gt_svg = """\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <g id="axes_1">
    <g id="PathCollection_1">
      <g><use xlink:href="#p1" x="10" y="20" style="fill: #ff0000; stroke: #ffffff" /></g>
      <g><use xlink:href="#p2" x="30" y="40" style="fill: #0000ff; stroke: #ffffff" /></g>
    </g>
  </g>
</svg>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_path = Path(tmpdir) / "pred.svg"
            gt_path = Path(tmpdir) / "gt.svg"
            pred_path.write_text(pred_svg, encoding="utf-8")
            gt_path.write_text(gt_svg, encoding="utf-8")

            result = compare_svgs(pred_path, gt_path)

        self.assertEqual(result["chart_type"], "scatter")
        self.assertEqual(result["metrics"]["matched_points"], 2)
        self.assertEqual(result["metrics"]["color_matched_points"], 1)
        self.assertEqual(result["metrics"]["point_match_ratio"], 1.0)
        self.assertEqual(result["metrics"]["color_match_ratio"], 0.5)
        self.assertLess(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
