from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from chart_agent.perception.area_svg_updater import (
    SVG_NS,
    _extract_area_groups,
    _extract_multi_add_series_specs,
    _resolve_area_ops,
)


class AreaSvgUpdaterTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
