from __future__ import annotations

import unittest

from chart_agent.perception.area_svg_updater import _extract_multi_add_series_specs, _resolve_area_ops


class AreaSvgUpdaterTests(unittest.TestCase):
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
