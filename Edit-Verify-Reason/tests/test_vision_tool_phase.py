from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from chart_agent.core.vision_tool_phase import (
    _coerce_tool_calls,
    _extract_area_top_boundary,
    _extract_colored_scatter_points,
    _filter_explicit_answer_markup_tools,
    _extract_line_legend_map,
    _extract_line_labels_from_intersection_question,
    _prefer_line_intersection_tools,
    _prefer_multi_color_scatter_tools,
    _svg_highlight_top_boundary,
    _svg_isolate_all_color_topologies,
    _svg_isolate_color_topology,
    _svg_isolate_target_lines,
    _svg_zoom_and_highlight_intersection,
)


class VisionToolPhaseTests(unittest.TestCase):
    def test_scatter_prefers_multi_color_topology_by_default(self) -> None:
        calls = [{"tool": "isolate_color_topology", "args": {"target_color": "red"}}]

        updated = _prefer_multi_color_scatter_tools(
            chart_type="scatter",
            question="How many clusters are there in the scatter plot?",
            tool_calls=calls,
        )

        self.assertEqual(updated, [{"tool": "isolate_all_color_topologies", "args": {}}])

    def test_scatter_keeps_single_color_when_question_names_color(self) -> None:
        calls = [{"tool": "isolate_color_topology", "args": {"target_color": "red"}}]

        updated = _prefer_multi_color_scatter_tools(
            chart_type="scatter",
            question="How many red clusters are visible?",
            tool_calls=calls,
        )

        self.assertEqual(updated, calls)

    def test_scatter_keeps_single_color_when_question_names_chinese_color(self) -> None:
        calls = [{"tool": "isolate_color_topology", "args": {"target_color": "blue"}}]

        updated = _prefer_multi_color_scatter_tools(
            chart_type="scatter",
            question="蓝色点形成了几个簇？",
            tool_calls=calls,
        )

        self.assertEqual(updated, calls)

    def test_scatter_replaces_all_single_color_calls_when_question_is_not_color_specific(self) -> None:
        calls = [
            {"tool": "isolate_color_topology", "args": {"target_color": "red"}},
            {"tool": "isolate_color_topology", "args": {"target_color": "blue"}},
        ]

        updated = _prefer_multi_color_scatter_tools(
            chart_type="scatter",
            question="How many clusters are visible overall?",
            tool_calls=calls,
        )

        self.assertEqual(updated, [{"tool": "isolate_all_color_topologies", "args": {}}])

    def test_markup_tools_are_accepted(self) -> None:
        calls, rejected = _coerce_tool_calls(
            [
                {"tool": "add_point", "args": {"x": 1, "y": 2, "radius": 4}},
                {"tool": "draw_line", "args": {"x1": 1, "y1": 2, "x2": 13, "y2": 14}},
                {"tool": "highlight_rect", "args": {"x1": 1, "y1": 2, "x2": 13, "y2": 14}},
            ],
            canvas_width=100,
            canvas_height=100,
        )

        self.assertEqual(
            calls,
            [
                {
                    "tool": "add_point",
                    "args": {"x": 1.0, "y": 2.0, "radius": 4.0, "color": "#ff2d55", "label": ""},
                },
                {
                    "tool": "draw_line",
                    "args": {
                        "x1": 1.0,
                        "y1": 2.0,
                        "x2": 13.0,
                        "y2": 14.0,
                        "width": 1.6,
                        "color": "#ff9500",
                        "label": "",
                    },
                },
                {
                    "tool": "highlight_rect",
                    "args": {
                        "x1": 1.0,
                        "y1": 2.0,
                        "x2": 13.0,
                        "y2": 14.0,
                        "width": 1.2,
                        "color": "#007aff",
                        "fill_opacity": 0.08,
                        "label": "",
                    },
                }
            ],
        )
        self.assertEqual(rejected, [])

    def test_non_scatter_tools_are_unchanged(self) -> None:
        calls = [{"tool": "isolate_color_topology", "args": {"target_color": "red"}}]

        updated = _prefer_multi_color_scatter_tools(chart_type="line", question="How many crossings?", tool_calls=calls)

        self.assertEqual(updated, calls)

    def test_existing_multi_color_call_is_preserved(self) -> None:
        calls = [{"tool": "isolate_all_color_topologies", "args": {}}]

        updated = _prefer_multi_color_scatter_tools(
            chart_type="scatter",
            question="Count all clusters in the scatter plot.",
            tool_calls=calls,
        )

        self.assertEqual(updated, calls)

    def test_area_chart_forbids_markup_tools_even_without_explicit_answer_label(self) -> None:
        calls = [
            {"tool": "highlight_top_boundary", "args": {}},
            {
                "tool": "add_point",
                "args": {"x": 250.0, "y": 60.0, "radius": 4.0, "color": "#ff2d55", "label": ""},
            },
            {
                "tool": "draw_line",
                "args": {"x1": 250.0, "y1": 60.0, "x2": 250.0, "y2": 300.0, "width": 2.0, "color": "#ff9500", "label": "guide"},
            },
            {
                "tool": "highlight_rect",
                "args": {"x1": 230.0, "y1": 320.0, "x2": 270.0, "y2": 340.0, "label": ""},
            },
        ]

        kept, rejected = _filter_explicit_answer_markup_tools(
            chart_type="area",
            question="In which year does the overall maximum occur?",
            tool_calls=calls,
        )

        self.assertEqual(kept, [{"tool": "highlight_top_boundary", "args": {}}])
        self.assertEqual(
            [item["reason"] for item in rejected],
            [
                "area_markup_tool_forbidden",
                "area_markup_tool_forbidden",
                "area_markup_tool_forbidden",
            ],
        )

    def test_explicit_answer_markup_is_filtered_for_line_questions(self) -> None:
        calls = [
            {
                "tool": "add_point",
                "args": {"x": 250.0, "y": 60.0, "radius": 4.0, "color": "#ff2d55", "label": "Overall Maximum"},
            },
            {
                "tool": "highlight_rect",
                "args": {"x1": 230.0, "y1": 320.0, "x2": 270.0, "y2": 340.0, "label": "Year of Maximum"},
            },
        ]

        kept, rejected = _filter_explicit_answer_markup_tools(
            chart_type="line",
            question="Which year has the maximum value?",
            tool_calls=calls,
        )

        self.assertEqual(kept, [])
        self.assertEqual(
            [item["reason"] for item in rejected],
            ["explicit_answer_markup", "explicit_answer_markup"],
        )

    def test_line_intersection_question_adds_zoom_highlight_call(self) -> None:
        updated = _prefer_line_intersection_tools(
            chart_type="line",
            question="How many times do the lines for Starburst Online and AetherNet intersect?",
            tool_calls=[],
        )

        self.assertEqual(
            updated,
            [
                {
                    "tool": "isolate_target_lines",
                    "args": {"line_A": "Starburst Online", "line_B": "AetherNet"},
                },
                {
                    "tool": "zoom_and_highlight_intersection",
                    "args": {"line_A": "Starburst Online", "line_B": "AetherNet"},
                }
            ],
        )

    def test_extract_line_labels_from_intersection_question(self) -> None:
        labels = _extract_line_labels_from_intersection_question(
            "After deleting the category CrimsonLink, how many times do the lines for Starburst Online and AetherNet intersect?"
        )

        self.assertEqual(labels, ("Starburst Online", "AetherNet"))

    def test_extract_colored_scatter_points_reads_all_path_collections(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
              <g id="axes_1">
                <g id="PathCollection_1">
                  <g><use xlink:href="#m1" x="1" y="2" style="fill: #2ca02c; stroke: #ffffff" /></g>
                </g>
                <g id="PathCollection_2">
                  <g><use xlink:href="#m2" x="3" y="4" style="fill: #ff7f0e; stroke: #ffffff" /></g>
                </g>
                <g id="PathCollection_3">
                  <g><use xlink:href="#m3" x="5" y="6" style="fill: #9467bd; stroke: #ffffff" /></g>
                </g>
              </g>
            </svg>
            """
        )

        points = _extract_colored_scatter_points(root, "http://www.w3.org/2000/svg")
        fills = sorted(point["fill"] for point in points)

        self.assertEqual(fills, ["#2ca02c", "#9467bd", "#ff7f0e"])

    def test_isolate_all_color_topologies_draws_background_halos_within_eps(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="PathCollection_1">
                  <g><use href="#m1" x="10" y="20" style="fill: #2ca02c; stroke: #ffffff" /></g>
                </g>
                <g id="PathCollection_2">
                  <g><use href="#m2" x="30" y="40" style="fill: #ff7f0e; stroke: #ffffff" /></g>
                </g>
              </g>
            </svg>
            """
        )
        overlay = ET.Element("{http://www.w3.org/2000/svg}g")

        _svg_isolate_all_color_topologies(
            root,
            overlay,
            "http://www.w3.org/2000/svg",
            100,
            100,
            scatter_cluster_context={
                "x_ticks": [(0.0, 0.0), (10.0, 10.0)],
                "y_ticks": [(0.0, 0.0), (10.0, 10.0)],
                "eps": 4.0,
            },
        )

        ns = {"svg": "http://www.w3.org/2000/svg"}
        axes = root.find(".//svg:g[@id='axes_1']", ns)
        self.assertIsNotNone(axes)
        children_ids = [str(child.get("id") or "") for child in list(axes)]
        self.assertEqual(children_ids[0], "tool_aug_scatter_background")

        halos = root.findall(".//svg:g[@id='tool_aug_scatter_background']/svg:circle", ns)
        self.assertEqual(len(halos), 2)
        self.assertEqual(sorted(circle.get("fill") for circle in halos), ["#2ca02c", "#ff7f0e"])
        self.assertTrue(all(abs(float(circle.get("r") or 0.0) - 2.0) < 1e-6 for circle in halos))

    def test_isolate_color_topology_draws_only_target_color_backgrounds(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="PathCollection_1">
                  <g><use href="#m1" x="10" y="20" style="fill: #2ca02c; stroke: #ffffff" /></g>
                </g>
                <g id="PathCollection_2">
                  <g><use href="#m2" x="30" y="40" style="fill: #ff7f0e; stroke: #ffffff" /></g>
                </g>
                <g id="PathCollection_3">
                  <g><use href="#m3" x="50" y="60" style="fill: #2ca02c; stroke: #ffffff" /></g>
                </g>
              </g>
            </svg>
            """
        )
        overlay = ET.Element("{http://www.w3.org/2000/svg}g")

        _svg_isolate_color_topology(
            root,
            overlay,
            "http://www.w3.org/2000/svg",
            {"target_color": "green"},
            100,
            100,
            scatter_cluster_context={
                "x_ticks": [(0.0, 0.0), (10.0, 10.0)],
                "y_ticks": [(0.0, 0.0), (10.0, 10.0)],
                "eps": 6.0,
            },
        )

        ns = {"svg": "http://www.w3.org/2000/svg"}
        halos = root.findall(".//svg:g[@id='tool_aug_scatter_background']/svg:circle", ns)
        self.assertEqual(len(halos), 2)
        self.assertEqual(sorted(circle.get("fill") for circle in halos), ["#2ca02c", "#2ca02c"])
        self.assertTrue(all(abs(float(circle.get("r") or 0.0) - 3.0) < 1e-6 for circle in halos))

    def test_zoom_and_highlight_intersection_draws_only_target_line_overlays(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="line2d_1">
                  <path d="M 10 10 L 20 20 L 30 30" style="fill: none; stroke: #ff0000; stroke-width: 2" />
                </g>
                <g id="line2d_2">
                  <path d="M 10 30 L 20 20 L 30 10" style="fill: none; stroke: #0000ff; stroke-width: 2" />
                </g>
              </g>
              <g id="legend_1">
                <g id="legend_line_a"><path d="M 0 0 L 5 0" style="stroke: #ff0000" /></g>
                <g id="text_1"><text>Starburst Online</text></g>
                <g id="legend_line_b"><path d="M 0 0 L 5 0" style="stroke: #0000ff" /></g>
                <g id="text_2"><text>AetherNet</text></g>
              </g>
            </svg>
            """
        )
        overlay = ET.Element("{http://www.w3.org/2000/svg}g")

        _svg_zoom_and_highlight_intersection(
            root,
            overlay,
            "http://www.w3.org/2000/svg",
            {"line_A": "Starburst Online", "line_B": "AetherNet"},
            100,
            100,
        )

        paths = overlay.findall("{http://www.w3.org/2000/svg}path")
        lines = overlay.findall("{http://www.w3.org/2000/svg}line")
        circles = overlay.findall("{http://www.w3.org/2000/svg}circle")
        texts = overlay.findall("{http://www.w3.org/2000/svg}text")

        self.assertGreaterEqual(len(paths), 2)
        self.assertEqual(len(lines), 0)
        self.assertEqual(len(circles), 0)
        self.assertEqual(texts, [])
        overlay_strokes = [path.get("stroke") for path in paths[:2]]
        self.assertEqual(sorted(overlay_strokes), ["#0000ff", "#ff0000"])

    def test_zoom_and_highlight_intersection_ignores_legend_sample_lines_inside_axes(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="legend_1">
                  <g id="line2d_update_legend"><path d="M 0 0 L 5 0" style="stroke: #2ca02c" /></g>
                  <g id="text_update_legend"><!-- Cinder Guild --><text>Cinder Guild</text></g>
                  <g id="legend_line_a"><path d="M 0 14 L 5 14" style="stroke: #98df8a" /></g>
                  <g id="text_1"><!-- Aether University --><text>Aether University</text></g>
                </g>
                <g id="line2d_1">
                  <path d="M 10 10 L 20 20 L 30 30" style="fill: none; stroke: #98df8a; stroke-width: 2" />
                </g>
                <g id="line2d_update">
                  <path d="M 10 30 L 20 20 L 30 10" style="fill: none; stroke: #2ca02c; stroke-width: 2" />
                </g>
              </g>
            </svg>
            """
        )
        overlay = ET.Element("{http://www.w3.org/2000/svg}g")

        _svg_zoom_and_highlight_intersection(
            root,
            overlay,
            "http://www.w3.org/2000/svg",
            {"line_A": "Aether University", "line_B": "Cinder Guild"},
            100,
            100,
        )

        lines = overlay.findall("{http://www.w3.org/2000/svg}line")
        circles = overlay.findall("{http://www.w3.org/2000/svg}circle")
        self.assertEqual(len(lines), 0)
        self.assertEqual(len(circles), 0)

    def test_extract_area_top_boundary_returns_upper_envelope(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="FillBetweenPolyCollection_1">
                  <path d="M 10 60 L 20 40 L 30 50 L 30 90 L 20 90 L 10 90 Z" />
                </g>
                <g id="FillBetweenPolyCollection_2">
                  <path d="M 10 45 L 20 30 L 30 35 L 30 60 L 20 70 L 10 65 Z" />
                </g>
              </g>
            </svg>
            """
        )

        boundary = _extract_area_top_boundary(root, "http://www.w3.org/2000/svg")

        self.assertEqual(boundary, [(10.0, 45.0), (20.0, 30.0), (30.0, 35.0)])

    def test_highlight_top_boundary_draws_only_polyline_overlays(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="FillBetweenPolyCollection_1">
                  <path d="M 10 60 L 20 40 L 30 50 L 30 90 L 20 90 L 10 90 Z" />
                </g>
                <g id="FillBetweenPolyCollection_2">
                  <path d="M 10 45 L 20 30 L 30 35 L 30 60 L 20 70 L 10 65 Z" />
                </g>
              </g>
            </svg>
            """
        )
        overlay = ET.Element("{http://www.w3.org/2000/svg}g")

        _svg_highlight_top_boundary(
            root,
            overlay,
            "http://www.w3.org/2000/svg",
            100,
            100,
        )

        paths = overlay.findall("{http://www.w3.org/2000/svg}path")
        lines = overlay.findall("{http://www.w3.org/2000/svg}line")
        circles = overlay.findall("{http://www.w3.org/2000/svg}circle")

        self.assertEqual(len(paths), 3)
        self.assertEqual(len(lines), 0)
        self.assertEqual(len(circles), 0)
        self.assertEqual(paths[0].get("d"), "M 10.000000 45.000000 L 20.000000 30.000000 L 30.000000 35.000000")

    def test_isolate_target_lines_fades_only_non_target_lines(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="axes_1">
                <g id="legend_1">
                  <g id="line2d_legend_1"><path d="M 0 0 L 5 0" style="stroke: #ff0000" /></g>
                  <g id="text_1"><!-- Alpha --><text>Alpha</text></g>
                  <g id="line2d_legend_2"><path d="M 0 14 L 5 14" style="stroke: #0000ff" /></g>
                  <g id="text_2"><!-- Beta --><text>Beta</text></g>
                  <g id="line2d_legend_3"><path d="M 0 28 L 5 28" style="stroke: #00aa00" /></g>
                  <g id="text_3"><!-- Gamma --><text>Gamma</text></g>
                </g>
                <g id="line2d_1">
                  <path d="M 10 10 L 20 20 L 30 30" style="fill: none; stroke: #ff0000; stroke-width: 2" />
                </g>
                <g id="line2d_2">
                  <path d="M 10 30 L 20 20 L 30 10" style="fill: none; stroke: #0000ff; stroke-width: 2" />
                </g>
                <g id="line2d_3">
                  <path d="M 10 25 L 20 25 L 30 25" style="fill: none; stroke: #00aa00; stroke-width: 2" />
                </g>
              </g>
            </svg>
            """
        )

        _svg_isolate_target_lines(
            root,
            "http://www.w3.org/2000/svg",
            {"line_A": "Alpha", "line_B": "Beta"},
        )

        gamma_path = root.find(".//{http://www.w3.org/2000/svg}g[@id='line2d_3']/{http://www.w3.org/2000/svg}path")
        alpha_path = root.find(".//{http://www.w3.org/2000/svg}g[@id='line2d_1']/{http://www.w3.org/2000/svg}path")
        gamma_text = root.find(".//{http://www.w3.org/2000/svg}g[@id='text_3']/{http://www.w3.org/2000/svg}text")
        alpha_text = root.find(".//{http://www.w3.org/2000/svg}g[@id='text_1']/{http://www.w3.org/2000/svg}text")

        assert gamma_path is not None
        assert alpha_path is not None
        assert gamma_text is not None
        assert alpha_text is not None
        self.assertIn("opacity: 0.46", gamma_path.get("style", ""))
        self.assertIn("opacity: 1.0", alpha_path.get("style", ""))
        self.assertNotIn("opacity:", gamma_text.get("style", ""))
        self.assertNotIn("opacity:", alpha_text.get("style", ""))

    def test_extract_line_legend_map_supports_generated_update_legend_groups(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="legend_1">
                <g id="line2d_1"><path d="M 0 0 L 5 0" style="stroke: #ff0000" /></g>
                <g id="text_1"><!-- Starburst Online --><text>Starburst Online</text></g>
                <g id="line2d_update_legend"><path d="M 0 14 L 5 14" style="stroke: #0000ff" /></g>
                <g id="text_update_legend"><!-- AetherNet --><text>AetherNet</text></g>
              </g>
            </svg>
            """
        )

        content = ET.tostring(root, encoding="unicode")
        legend_map = _extract_line_legend_map(root, "http://www.w3.org/2000/svg", content)

        self.assertEqual(legend_map, {"Starburst Online": "#ff0000", "AetherNet": "#0000ff"})

    def test_extract_line_legend_map_supports_direct_text_nodes(self) -> None:
        root = ET.fromstring(
            """
            <svg xmlns="http://www.w3.org/2000/svg">
              <g id="legend_1">
                <g id="legend_line_a"><path d="M 0 0 L 5 0" style="stroke: #ff0000" /></g>
                <text x="10" y="10">Starburst Online</text>
                <g id="legend_line_b"><path d="M 0 14 L 5 14" style="stroke: #0000ff" /></g>
                <text x="10" y="24">AetherNet</text>
              </g>
            </svg>
            """
        )

        content = ET.tostring(root, encoding="unicode")
        legend_map = _extract_line_legend_map(root, "http://www.w3.org/2000/svg", content)

        self.assertEqual(legend_map, {"Starburst Online": "#ff0000", "AetherNet": "#0000ff"})


if __name__ == "__main__":
    unittest.main()
