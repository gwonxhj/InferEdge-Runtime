from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class JetsonReportSnapshotTest(unittest.TestCase):
    def test_jetson_evidence_snapshot_records_capture_depth(self):
        text = (ROOT / "docs" / "reports" / "jetson_evidence_summary.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("## Evidence Depth", text)
        self.assertIn("capture_depth | `short_smoke`", text)
        self.assertIn("not a sustained thermal claim", text)

    def test_power_mode_snapshot_records_depth_for_both_runs(self):
        text = (ROOT / "docs" / "reports" / "jetson_power_mode_comparison.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("## Run Depth Comparison", text)
        self.assertIn("capture_depth | `short_smoke` | `short_smoke`", text)
        self.assertIn("should not be described as a sustained thermal benchmark", text)


if __name__ == "__main__":
    unittest.main()
