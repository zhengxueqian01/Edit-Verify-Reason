from __future__ import annotations

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chart_agent.perception.text_line_renderer import (
    _apply_y_padding,
    _lock_in_matplotlib_y_axis_format,
    render_line_from_text,
)


class TextLineRendererTests(unittest.TestCase):
    def test_lock_in_y_axis_format_matches_matplotlib_threshold(self) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [0, 1.0e10])
        _apply_y_padding(ax, [0, 1.0e10], 0.2)

        axis_meta = _lock_in_matplotlib_y_axis_format(fig, ax)

        self.assertEqual(axis_meta["offset_text"], "1e10")
        self.assertTrue(axis_meta["tick_labels"])
        plt.close(fig)

    def test_lock_in_y_axis_format_stays_at_1e9_below_threshold(self) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [0, 9.9e9])
        ax.set_ylim(0, 9.9e9)

        axis_meta = _lock_in_matplotlib_y_axis_format(fig, ax)

        self.assertEqual(axis_meta["offset_text"], "1e9")
        plt.close(fig)

    def test_render_line_from_text_outputs_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "line.png")
            target, meta = render_line_from_text(
                '[{"year": 1991, "A": 0, "B": 10000000000}, {"year": 1992, "A": 2, "B": 9000000000}]',
                output_path=output_path,
            )

            self.assertEqual(target, output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(meta["years"], [1991, 1992])


if __name__ == "__main__":
    unittest.main()
