from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from chart_agent.perception.area_svg_updater import (
    SVG_NS,
    _area_series_values,
    _extract_area_groups,
    _extract_legend_items,
    _extract_multi_add_series_specs,
    _resolve_area_ops,
    _update_area_remove_series,
    _update_area_year_point,
    update_area_svg,
)


class AreaSvgUpdaterTests(unittest.TestCase):
    def _write_temp_svg(self, content: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmpdir = tempfile.TemporaryDirectory()
        path = Path(tmpdir.name) / "case.svg"
        path.write_text(content, encoding="utf-8")
        return tmpdir, path

    def test_extract_area_groups_keeps_document_order_for_update_group(self) -> None:
        axes = ET.fromstring(
            f"""
            <g xmlns="{SVG_NS}" id="axes_1">
              <g id="FillBetweenPolyCollection_1">
                <path d="M 0 10 L 1 10 L 1 20 L 0 20 Z" style="fill: #111111" />
              </g>
              <g id="FillBetweenPolyCollection_2">
                <path d="M 0 5 L 1 5 L 1 10 L 0 10 Z" style="fill: #222222" />
              </g>
              <g id="FillBetweenPolyCollection_update">
                <path d="M 0 0 L 1 0 L 1 5 L 0 5 Z" style="fill: #333333" />
              </g>
            </g>
            """
        )

        groups = _extract_area_groups(axes)

        self.assertEqual(
            [group["id"] for group in groups],
            [
                "FillBetweenPolyCollection_1",
                "FillBetweenPolyCollection_2",
                "FillBetweenPolyCollection_update",
            ],
        )

    def test_multi_add_specs_ignore_change_clauses(self) -> None:
        question = (
            'Change "Zenith Enterprises" in 2021 to 11.81; '
            'Change "Zenith Enterprises" in 2022 to 13.36'
        )

        specs = _extract_multi_add_series_specs(question)

        self.assertEqual(specs, [])

    def test_resolve_area_ops_keeps_multi_change_sequence(self) -> None:
        question = (
            'Change "Zenith Enterprises" in 2021 to 11.81; '
            'Change "Zenith Enterprises" in 2022 to 13.36'
        )

        ops = _resolve_area_ops(question)

        self.assertEqual(ops, ["change", "change"])

    def test_update_area_add_series_rebuilds_from_svg_not_mapping_top_boundary(self) -> None:
        content = f"""
        <svg xmlns="{SVG_NS}">
          <g id="axes_1">
            <g id="FillBetweenPolyCollection_1">
              <path d="M 0 90 L 10 80 L 10 100 L 0 100 Z" style="fill: #111111" />
            </g>
            <g id="FillBetweenPolyCollection_2">
              <path d="M 0 85 L 10 75 L 10 80 L 0 90 Z" style="fill: #222222" />
            </g>
          </g>
          <g id="legend_1">
            <g id="patch_1"><path d="M 0 0 L 5 0 L 5 5 L 0 5 z" style="fill: #111111" /></g>
            <g id="text_1"><!-- Alpha --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
            <g id="patch_2"><path d="M 0 14 L 5 14 L 5 19 L 0 19 z" style="fill: #222222" /></g>
            <g id="text_2"><!-- Beta --><g transform="translate(30 34) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        tmpdir, svg_path = self._write_temp_svg(content)
        out_svg = Path(tmpdir.name) / "out.svg"
        out_png = Path(tmpdir.name) / "out.png"
        mapping_info = {
            "top_boundary": [(0.0, 40.0), (10.0, 35.0)],
            "y_ticks": [(100.0, 0.0), (0.0, 100.0)],
        }

        with patch("chart_agent.perception.area_svg_updater.render_svg_to_png", return_value=str(out_png)):
            update_area_svg(
                str(svg_path),
                'Add the category/series "C"',
                mapping_info,
                output_path=str(out_png),
                svg_output_path=str(out_svg),
                operation_target={"category_name": "C"},
                data_change={"add": {"category_name": "C", "values": [2, 3]}},
            )

        root = ET.parse(out_svg).getroot()
        axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
        assert axes is not None
        groups = _extract_area_groups(axes)
        self.assertEqual(len(groups), 3)
        update_d = groups[-1]["path"].get("d", "")
        self.assertIn("0.000000 83.000000", update_d)
        self.assertIn("10.000000 72.000000", update_d)

        tmpdir.cleanup()

    def test_update_area_add_series_legend_uses_text_group_not_raw_text(self) -> None:
        content = f"""
        <svg xmlns="{SVG_NS}">
          <g id="axes_1">
            <g id="FillBetweenPolyCollection_1">
              <path d="M 0 90 L 10 80 L 10 100 L 0 100 Z" style="fill: #111111" />
            </g>
            <g id="FillBetweenPolyCollection_2">
              <path d="M 0 85 L 10 75 L 10 80 L 0 90 Z" style="fill: #222222" />
            </g>
          </g>
          <g id="legend_1">
            <g id="patch_1"><path d="M 0 0 L 5 0 L 5 5 L 0 5 z" style="fill: #111111" /></g>
            <g id="text_1"><!-- Alpha --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
            <g id="patch_2"><path d="M 0 14 L 5 14 L 5 19 L 0 19 z" style="fill: #222222" /></g>
            <g id="text_2"><!-- Beta --><g transform="translate(30 34) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        tmpdir, svg_path = self._write_temp_svg(content)
        out_svg = Path(tmpdir.name) / "out.svg"
        out_png = Path(tmpdir.name) / "out.png"
        mapping_info = {
            "top_boundary": [(0.0, 40.0), (10.0, 35.0)],
            "y_ticks": [(100.0, 0.0), (0.0, 100.0)],
        }

        with patch("chart_agent.perception.area_svg_updater.render_svg_to_png", return_value=str(out_png)):
            update_area_svg(
                str(svg_path),
                'Add the category/series "C"',
                mapping_info,
                output_path=str(out_png),
                svg_output_path=str(out_svg),
                operation_target={"category_name": "C"},
                data_change={"add": {"category_name": "C", "values": [2, 3]}},
            )

        content_out = out_svg.read_text(encoding="utf-8")
        root = ET.parse(out_svg).getroot()
        legend = root.find(f'.//{{{SVG_NS}}}g[@id="legend_1"]')
        assert legend is not None
        direct_texts = legend.findall(f'./{{{SVG_NS}}}text')
        self.assertEqual(direct_texts, [])
        self.assertIn('id="text_update"', content_out)
        self.assertIn("<!-- C -->", content_out)
        _, items = _extract_legend_items(root, content_out)
        self.assertIn("C", [item["label"] for item in items])

        tmpdir.cleanup()

    def test_update_area_remove_series_uses_legend_fill_mapping(self) -> None:
        content = f"""
        <svg xmlns="{SVG_NS}">
          <g id="axes_1">
            <g id="FillBetweenPolyCollection_1">
              <path d="M 0 90 L 10 80 L 10 100 L 0 100 Z" style="fill: #111111" />
            </g>
            <g id="FillBetweenPolyCollection_2">
              <path d="M 0 85 L 10 75 L 10 80 L 0 90 Z" style="fill: #222222" />
            </g>
            <g id="FillBetweenPolyCollection_3">
              <path d="M 0 80 L 10 70 L 10 75 L 0 85 Z" style="fill: #333333" />
            </g>
          </g>
          <g id="legend_1">
            <g id="patch_1"><path d="M 0 0 L 5 0 L 5 5 L 0 5 z" style="fill: #111111" /></g>
            <g id="text_1"><!-- Alpha --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
            <g id="patch_2"><path d="M 0 14 L 5 14 L 5 19 L 0 19 z" style="fill: #333333" /></g>
            <g id="text_2"><!-- Beta --><g transform="translate(30 34) scale(0.1 -0.1)"></g></g>
            <g id="patch_3"><path d="M 0 28 L 5 28 L 5 33 L 0 33 z" style="fill: #222222" /></g>
            <g id="text_3"><!-- Gamma --><g transform="translate(30 48) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        tmpdir, svg_path = self._write_temp_svg(content)
        out_svg = Path(tmpdir.name) / "out.svg"
        out_png = Path(tmpdir.name) / "out.png"
        mapping_info = {
            "x_ticks": [(0.0, 0.0), (10.0, 10.0)],
            "y_ticks": [(100.0, 0.0), (0.0, 100.0)],
        }

        with patch("chart_agent.perception.area_svg_updater.render_svg_to_png", return_value=str(out_png)):
            _update_area_remove_series(
                str(svg_path),
                'Delete the category/series "Beta"',
                mapping_info,
                output_path=str(out_png),
                svg_output_path=str(out_svg),
            )

        text = out_svg.read_text(encoding="utf-8")
        self.assertIn("<!-- Alpha -->", text)
        self.assertNotIn("<!-- Beta -->", text)
        self.assertIn("<!-- Gamma -->", text)

        tmpdir.cleanup()

    def test_update_area_change_uses_fill_mapping_when_legend_order_differs(self) -> None:
        content = f"""
        <svg xmlns="{SVG_NS}">
          <g id="axes_1">
            <g id="FillBetweenPolyCollection_1">
              <path d="M 0 95 L 10 94 L 10 100 L 0 100 Z" style="fill: #333333" />
            </g>
            <g id="FillBetweenPolyCollection_2">
              <path d="M 0 85 L 10 84 L 10 95 L 0 95 Z" style="fill: #111111" />
            </g>
            <g id="FillBetweenPolyCollection_3">
              <path d="M 0 80 L 10 79 L 10 85 L 0 85 Z" style="fill: #222222" />
            </g>
          </g>
          <g id="legend_1">
            <g id="patch_1"><path d="M 0 0 L 5 0 L 5 5 L 0 5 z" style="fill: #111111" /></g>
            <g id="text_1"><!-- Alpha --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
            <g id="patch_2"><path d="M 0 14 L 5 14 L 5 19 L 0 19 z" style="fill: #222222" /></g>
            <g id="text_2"><!-- Beta --><g transform="translate(30 34) scale(0.1 -0.1)"></g></g>
            <g id="patch_3"><path d="M 0 28 L 5 28 L 5 33 L 0 33 z" style="fill: #333333" /></g>
            <g id="text_3"><!-- Gamma --><g transform="translate(30 48) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        tmpdir, svg_path = self._write_temp_svg(content)
        out_svg = Path(tmpdir.name) / "out.svg"
        out_png = Path(tmpdir.name) / "out.png"
        mapping_info = {
            "x_ticks": [(0.0, 0.0), (10.0, 10.0)],
            "y_ticks": [(100.0, 0.0), (0.0, 100.0)],
        }

        with patch("chart_agent.perception.area_svg_updater.render_svg_to_png", return_value=str(out_png)):
            _update_area_year_point(
                str(svg_path),
                'Change "Beta" in 10 to 12',
                mapping_info,
                output_path=str(out_png),
                svg_output_path=str(out_svg),
                llm=None,
                operation_target={"category_name": "Beta"},
                data_change={
                    "change": {
                        "changes": [
                            {"category_name": "Beta", "years": [10], "values": [12]},
                        ]
                    }
                },
            )

        root = ET.parse(out_svg).getroot()
        axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
        assert axes is not None
        groups = _extract_area_groups(axes)
        x_values, series_values = _area_series_values(groups, mapping_info["y_ticks"])
        beta_idx = next(i for i, group in enumerate(groups) if group["fill"] == "#222222")
        year_idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - 10.0))
        self.assertEqual(series_values[beta_idx][year_idx], 12.0)

        tmpdir.cleanup()

    def test_update_area_change_applies_structured_multi_change(self) -> None:
        content = f"""
        <svg xmlns="{SVG_NS}">
          <g id="axes_1">
            <g id="FillBetweenPolyCollection_1">
              <path d="M 0 90 L 10 80 L 10 100 L 0 100 Z" style="fill: #111111" />
            </g>
            <g id="FillBetweenPolyCollection_2">
              <path d="M 0 85 L 10 75 L 10 80 L 0 90 Z" style="fill: #222222" />
            </g>
          </g>
          <g id="legend_1">
            <g id="patch_1"><path d="M 0 0 L 5 0 L 5 5 L 0 5 z" style="fill: #111111" /></g>
            <g id="text_1"><!-- A --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
            <g id="patch_2"><path d="M 0 14 L 5 14 L 5 19 L 0 19 z" style="fill: #222222" /></g>
            <g id="text_2"><!-- B --><g transform="translate(30 34) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        tmpdir, svg_path = self._write_temp_svg(content)
        out_svg = Path(tmpdir.name) / "out.svg"
        out_png = Path(tmpdir.name) / "out.png"
        mapping_info = {
            "x_ticks": [(0.0, 0.0), (10.0, 10.0)],
            "y_ticks": [(100.0, 0.0), (0.0, 100.0)],
        }

        with patch("chart_agent.perception.area_svg_updater.render_svg_to_png", return_value=str(out_png)):
            _update_area_year_point(
                str(svg_path),
                'Change "A" in 0 to 15; Change "B" in 10 to 8',
                mapping_info,
                output_path=str(out_png),
                svg_output_path=str(out_svg),
                llm=None,
                data_change={
                    "change": {
                        "changes": [
                            {"category_name": "A", "years": [0], "values": [15]},
                            {"category_name": "B", "years": [10], "values": [8]},
                        ]
                    }
                },
            )

        text = out_svg.read_text(encoding="utf-8")
        self.assertIn("0.000000 85.000000", text)
        self.assertIn("10.000000 72.000000", text)

        tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
