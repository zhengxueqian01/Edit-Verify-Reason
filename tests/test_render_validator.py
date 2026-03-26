from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from main import (
    _validate_line_change_step,
    _validate_line_delete_step,
    _validate_render_with_programmatic,
    _validate_scatter_add_step,
)
from chart_agent.perception.render_validator import _soften_uncertainty_only_failure


class RenderValidatorTests(unittest.TestCase):
    def _write_svg(self, tmpdir: str, name: str, content: str) -> Path:
        path = Path(tmpdir) / name
        path.write_text(content, encoding="utf-8")
        return path

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

    def test_validate_line_change_step_checks_exact_svg_point_value(self) -> None:
        svg_text = """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="axes_1">
            <g id="line2d_1">
              <path d="M 0 100 L 10 90 L 20 80" style="fill: none; stroke: #111111; stroke-width: 2" />
            </g>
          </g>
          <g id="legend_1">
            <g id="line_1"><path d="M 0 0 L 5 0" style="fill: none; stroke: #111111; stroke-width: 2" /></g>
            <g id="text_1"><!-- Alpha --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = self._write_svg(tmpdir, "line.svg", svg_text)

            result = _validate_line_change_step(
                {
                    "output_svg_path": str(svg_path),
                    "validation_context": {
                        "x_ticks": [(0.0, 2000.0), (10.0, 2001.0), (20.0, 2002.0)],
                        "y_ticks": [(100.0, 0.0), (0.0, 100.0)],
                    },
                    "operation_target": {"category_name": "Alpha"},
                    "data_change": {"category_name": "Alpha", "years": [2002], "values": [20]},
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["issues"], [])

    def test_validate_line_delete_step_compares_input_and_output_legends(self) -> None:
        input_svg_text = """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="axes_1">
            <g id="line2d_1">
              <path d="M 0 100 L 20 80" style="fill: none; stroke: #111111; stroke-width: 2" />
            </g>
            <g id="line2d_2">
              <path d="M 0 90 L 20 70" style="fill: none; stroke: #222222; stroke-width: 2" />
            </g>
          </g>
          <g id="legend_1">
            <g id="line_1"><path d="M 0 0 L 5 0" style="fill: none; stroke: #111111; stroke-width: 2" /></g>
            <g id="text_1"><!-- Alpha --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
            <g id="line_2"><path d="M 0 14 L 5 14" style="fill: none; stroke: #222222; stroke-width: 2" /></g>
            <g id="text_2"><!-- Beta --><g transform="translate(30 34) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        output_svg_text = """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="axes_1">
            <g id="line2d_2">
              <path d="M 0 90 L 20 70" style="fill: none; stroke: #222222; stroke-width: 2" />
            </g>
          </g>
          <g id="legend_1">
            <g id="line_2"><path d="M 0 14 L 5 14" style="fill: none; stroke: #222222; stroke-width: 2" /></g>
            <g id="text_2"><!-- Beta --><g transform="translate(30 34) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            input_svg = self._write_svg(tmpdir, "before.svg", input_svg_text)
            output_svg = self._write_svg(tmpdir, "after.svg", output_svg_text)

            result = _validate_line_delete_step(
                {
                    "input_svg_path": str(input_svg),
                    "output_svg_path": str(output_svg),
                    "operation_target": {"del_category": "Alpha"},
                    "question": 'Delete the category "Alpha".',
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["issues"], [])

    def test_validate_scatter_add_step_checks_rendered_svg_points(self) -> None:
        svg_text = """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="axes_1">
            <g id="PathCollection_update">
              <circle cx="10" cy="30" r="3" />
              <circle cx="20" cy="20" r="3" />
            </g>
          </g>
        </svg>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = self._write_svg(tmpdir, "scatter.svg", svg_text)

            result = _validate_scatter_add_step(
                {
                    "output_svg_path": str(svg_path),
                    "new_points": [
                        {"x": 1.0, "y": 1.0, "color": "orange"},
                        {"x": 2.0, "y": 2.0, "color": "orange"},
                    ],
                    "validation_context": {
                        "x_ticks": [(10.0, 1.0), (20.0, 2.0)],
                        "y_ticks": [(40.0, 0.0), (20.0, 2.0)],
                    },
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["issues"], [])

    def test_validate_render_with_programmatic_short_circuits_line_uncertainty(self) -> None:
        svg_text = """
        <svg xmlns="http://www.w3.org/2000/svg">
          <g id="axes_1">
            <g id="line2d_1">
              <path d="M 0 100 L 10 90 L 20 80" style="fill: none; stroke: #111111; stroke-width: 2" />
            </g>
          </g>
          <g id="legend_1">
            <g id="line_1"><path d="M 0 0 L 5 0" style="fill: none; stroke: #111111; stroke-width: 2" /></g>
            <g id="text_1"><!-- Alpha --><g transform="translate(30 20) scale(0.1 -0.1)"></g></g>
          </g>
        </svg>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = self._write_svg(tmpdir, "line.svg", svg_text)
            png_path = Path(tmpdir) / "line.png"
            Image.new("RGBA", (2, 2), (0, 0, 0, 255)).save(png_path)

            result = _validate_render_with_programmatic(
                output_image=str(png_path),
                chart_type="line",
                update_spec={},
                step_logs=[
                    {
                        "operation": "change",
                        "output_svg_path": str(svg_path),
                        "validation_context": {
                            "x_ticks": [(0.0, 2000.0), (10.0, 2001.0), (20.0, 2002.0)],
                            "y_ticks": [(100.0, 0.0), (0.0, 100.0)],
                        },
                        "operation_target": {"category_name": "Alpha"},
                        "data_change": {"category_name": "Alpha", "years": [2002], "values": [20]},
                    }
                ],
                llm=None,
                svg_perception_mode=None,
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["programmatic"])


if __name__ == "__main__":
    unittest.main()
