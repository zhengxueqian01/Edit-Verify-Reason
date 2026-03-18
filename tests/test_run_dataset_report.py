from __future__ import annotations

import unittest

from run_dataset_via_main import _sanitize_report_payload


class RunDatasetReportTests(unittest.TestCase):
    def test_sanitize_report_payload_removes_labels_recursively(self) -> None:
        payload = {
            "answer": "x",
            "labels": [0, 1, -1],
            "labels_by_color": {"#9467bd": [0, 0, 1]},
            "cluster_result": {
                "clusters": 2,
                "labels": [0, 0, 1],
                "labels_by_color": {"#2ca02c": [0, 0, 0]},
            },
            "items": [
                {"kind": "a", "labels": ["foo"], "labels_by_color": {"#ff7f0e": [1]}},
                {"kind": "b", "value": 1},
            ],
        }

        sanitized = _sanitize_report_payload(payload)

        self.assertNotIn("labels", sanitized)
        self.assertNotIn("labels_by_color", sanitized)
        self.assertNotIn("labels", sanitized["cluster_result"])
        self.assertNotIn("labels_by_color", sanitized["cluster_result"])
        self.assertNotIn("labels", sanitized["items"][0])
        self.assertNotIn("labels_by_color", sanitized["items"][0])
        self.assertEqual(sanitized["items"][1], {"kind": "b", "value": 1})


if __name__ == "__main__":
    unittest.main()
