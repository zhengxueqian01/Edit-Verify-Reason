from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_dataset_via_main import extract_final_svg_path


class RunDatasetSvgRecordingTests(unittest.TestCase):
    def test_extract_final_svg_path_ignores_stale_default_svg_without_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_svg = tmp_path / "dataset" / "task1" / "add-change" / "002" / "002.svg"
            source_svg.parent.mkdir(parents=True, exist_ok=True)
            source_svg.write_text("<svg />", encoding="utf-8")
            (source_svg.parent / "002.json").write_text(
                '{"chart_type": "area", "operation": "update"}',
                encoding="utf-8",
            )

            stale_fallback = Path("output/area/task1-add-change_002_area_update_updated.svg")
            stale_fallback.parent.mkdir(parents=True, exist_ok=True)
            stale_fallback.write_text("<svg>stale</svg>", encoding="utf-8")

            result = {
                "attempt_logs": [
                    {
                        "attempt": 1,
                        "step_logs": [],
                        "render_check": {
                            "ok": False,
                            "issues": ["operation_step_failed: No valid area update request found in question."],
                        },
                    }
                ]
            }

            self.assertIsNone(extract_final_svg_path(result, source_svg, "area"))

            stale_fallback.unlink()


if __name__ == "__main__":
    unittest.main()
