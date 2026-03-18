from __future__ import annotations

import unittest

from chart_agent.core.clusterer import run_dbscan_by_color


class ClustererTests(unittest.TestCase):
    def test_run_dbscan_by_color_counts_clusters_separately(self) -> None:
        points_by_color = {
            "#9467bd": [(0.0, 0.0), (0.5, 0.4), (1.0, 0.2), (10.0, 10.0), (10.4, 10.2), (10.8, 10.1)],
            "#2ca02c": [(30.0, 30.0), (30.5, 30.4), (31.0, 30.8)],
        }

        result = run_dbscan_by_color(points_by_color, "How many clusters are there?")

        self.assertEqual(result["clusters"], 3)
        self.assertEqual(result["cluster_counts_by_color"]["#9467bd"], 2)
        self.assertEqual(result["cluster_counts_by_color"]["#2ca02c"], 1)
        self.assertEqual(result["min_samples"], 3)
        self.assertEqual(result["mode"], "per_color")

    def test_run_dbscan_by_color_merges_named_and_hex_colors(self) -> None:
        points_by_color = {
            "red": [(0.0, 0.0), (0.5, 0.4), (1.0, 0.2)],
            "#d62728": [(10.0, 10.0), (10.3, 10.2), (10.7, 10.1)],
        }

        result = run_dbscan_by_color(points_by_color, "How many clusters are there? eps=2 min_sample=3")

        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["cluster_counts_by_color"]["#d62728"], 2)


if __name__ == "__main__":
    unittest.main()
