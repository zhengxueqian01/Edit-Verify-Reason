from __future__ import annotations

import unittest

from chart_agent.perception.line_svg_updater import (
    _compute_draw_style_y_limits,
    _compute_matplotlib_y_axis_layout,
    _extract_line_style,
    _extract_legend_items,
    _map_data_to_pixel,
    _append_legend_item,
    _pick_unused_line_stroke,
    _resolve_delete_labels,
)
import xml.etree.ElementTree as ET


class LineSvgAxisLayoutTests(unittest.TestCase):
    def test_resolve_delete_labels_falls_back_when_llm_returns_empty(self) -> None:
        class EmptyLLM:
            def invoke(self, prompt: str) -> object:
                return type("Resp", (), {"content": '{"labels": []}'})()

        labels = ["Customs Clearance", "Security Checks", "Tariff Negotiations"]
        resolved = _resolve_delete_labels(
            'Delete the categories Customs Clearance and Political Disruptions.',
            labels,
            EmptyLLM(),
        )

        self.assertEqual(resolved, ["Customs Clearance"])

    def test_crossing_into_1e10_uses_1e10_scale(self) -> None:
        view_min, view_max = _compute_draw_style_y_limits(0.0, 1.0e10)
        layout = _compute_matplotlib_y_axis_layout(view_min, view_max, 6)

        self.assertEqual(layout["axis_scale"], 1.0e10)
        self.assertEqual(layout["view_min"], view_min)
        self.assertEqual(layout["view_max"], view_max)
        self.assertEqual(layout["tick_labels"], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    def test_sub_1e10_range_can_stay_at_1e9_scale(self) -> None:
        view_min, view_max = _compute_draw_style_y_limits(0.0, 9.9e9)
        layout = _compute_matplotlib_y_axis_layout(view_min, view_max, 6)

        self.assertEqual(layout["axis_scale"], 1.0e10)
        self.assertEqual(layout["tick_labels"], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    def test_1e8_range_can_use_1e7_scale_when_visible_ticks_top_out_below_1e8(self) -> None:
        view_min, view_max = _compute_draw_style_y_limits(2.5e7, 9.2e7)
        layout = _compute_matplotlib_y_axis_layout(view_min, view_max, 6)

        self.assertEqual(layout["axis_scale"], 1.0e8)
        self.assertEqual(layout["tick_labels"], ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])

    def test_1e7_range_can_use_1e6_scale(self) -> None:
        view_min, view_max = _compute_draw_style_y_limits(2.5e6, 9.2e6)
        layout = _compute_matplotlib_y_axis_layout(view_min, view_max, 6)

        self.assertEqual(layout["axis_scale"], 1.0e7)
        self.assertEqual(layout["tick_labels"], ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])

    def test_view_interval_follows_draw_style_limits(self) -> None:
        data_min = 5.242348275e8
        data_max = 1.73141213073e9
        view_min, view_max = _compute_draw_style_y_limits(data_min, data_max)
        layout = _compute_matplotlib_y_axis_layout(view_min, view_max, 5)

        self.assertEqual(layout["view_min"], view_min)
        self.assertEqual(layout["view_max"], view_max)
        self.assertLessEqual(layout["view_min"], data_min)
        self.assertGreaterEqual(layout["view_max"], data_max)

    def test_reference_aug_case_000_matches_visible_tick_pattern(self) -> None:
        view_min, view_max = _compute_draw_style_y_limits(524234827.5, 1723595572.24)
        layout = _compute_matplotlib_y_axis_layout(view_min, view_max, 5)

        self.assertEqual(layout["axis_scale"], 1.0e9)
        self.assertEqual(layout["tick_labels"], ["0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "1.8"])

    def test_reference_aug_case_000_tick_spacing_uses_plot_bounds(self) -> None:
        view_min, view_max = _compute_draw_style_y_limits(524234827.5, 1723595572.24)
        layout = _compute_matplotlib_y_axis_layout(view_min, view_max, 5)
        pixels = [
            _map_data_to_pixel(value, view_min, view_max, 314.449375, 37.249375)
            for value in layout["tick_values"]
        ]
        diffs = [round(pixels[i] - pixels[i + 1], 6) for i in range(len(pixels) - 1)]

        for diff in diffs:
            self.assertAlmostEqual(diff, 38.52052, places=5)

    def test_draw_style_limit_matches_reference_rule(self) -> None:
        view_min, view_max = _compute_draw_style_y_limits(100.0, 200.0)

        self.assertEqual(view_min, 95.0)
        self.assertEqual(view_max, 215.0)

    def test_pick_unused_line_stroke_skips_existing_palette_colors(self) -> None:
        stroke = _pick_unused_line_stroke(["#1f77b4", "#ff7f0e", "#2ca02c"])

        self.assertEqual(stroke, "#d62728")

    def test_append_legend_item_places_new_label_after_last_existing_row(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="legend_1">
                <g id="line2d_1"><path d="M 10 10 L 20 10" style="stroke: #1f77b4; stroke-width: 2" /></g>
                <g id="text_1"><!-- First --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
                <g id="line2d_2"><path d="M 10 24 L 20 24" style="stroke: #ff7f0e; stroke-width: 2" /></g>
                <g id="text_2"><!-- Second --><g transform="translate(30 34) scale(0.1 -0.1)"></g></g>
              </g>
            </svg>
            """
        )
        content = ET.tostring(root, encoding="unicode")
        legend, items = _extract_legend_items(root, content)

        assert legend is not None
        _append_legend_item(legend, items, "Third", "#2ca02c", show_marker=False)

        text_nodes = legend.findall("{http://www.w3.org/2000/svg}text")
        self.assertEqual(len(text_nodes), 1)
        self.assertEqual(text_nodes[0].text, "Third")
        self.assertEqual(text_nodes[0].get("y"), "48.000000")

    def test_extract_line_style_uses_unused_color_and_preserves_marker_flag(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="line2d_1">
                  <path d="M 0 0 L 1 1" clip-path="url(#plot)" style="fill: none; stroke: #1f77b4; stroke-width: 2" />
                </g>
              </g>
            </svg>
            """
        )

        style = _extract_line_style(root)

        self.assertEqual(style["stroke"], "#ff7f0e")
        self.assertEqual(style["stroke_width"], 2.0)
        self.assertFalse(style["has_markers"])

    def test_extract_line_style_ignores_tick_line_groups(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="line2d_1">
                  <path d="M 0 0 L 0 3.5" style="stroke: #000000; stroke-width: 0.8" />
                  <use href="#tick" x="10" y="10" />
                </g>
                <g id="line2d_20">
                  <path
                    d="M 0 0 L 10 10"
                    clip-path="url(#plot)"
                    style="fill: none; stroke: #1f77b4; stroke-width: 2; stroke-linecap: square"
                  />
                </g>
              </g>
            </svg>
            """
        )

        style = _extract_line_style(root)

        self.assertEqual(style["stroke"], "#ff7f0e")
        self.assertEqual(style["stroke_width"], 2.0)
        self.assertFalse(style["has_markers"])


if __name__ == "__main__":
    unittest.main()
