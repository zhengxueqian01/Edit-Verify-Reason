from __future__ import annotations

import unittest

from main import _resolve_scatter_cluster_params


class MainClusterParamsTests(unittest.TestCase):
    def test_resolve_scatter_cluster_params_reads_question_suffix(self) -> None:
        params = _resolve_scatter_cluster_params(
            {},
            "How many clusters are there? eps=6.4 min_sample=3",
        )

        self.assertEqual(params["mode"], "per_color")
        self.assertEqual(params["algorithm"], "DBSCAN")
        self.assertEqual(params["eps"], 6.4)
        self.assertEqual(params["min_samples"], 3)


if __name__ == "__main__":
    unittest.main()
