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

    def test_jetson_evidence_depth_audit_records_roadmap_boundary(self):
        text = (
            ROOT / "docs" / "reports" / "jetson_evidence_depth_audit.md"
        ).read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_ko = (ROOT / "README.ko.md").read_text(encoding="utf-8")

        for marker in [
            "p95/p99 latency",
            "Thermal behavior starter evidence",
            "Memory behavior starter evidence",
            "Power draw starter evidence",
            "Not yet sustained",
            "runs >= 500",
            "tegrastats sample_count >= 300",
            "Jetson hardware is required only for collecting that new sustained evidence",
            "Runtime `result.json` compatibility",
        ]:
            self.assertIn(marker, text)

        self.assertIn("docs/reports/jetson_evidence_depth_audit.md", readme)
        self.assertIn("docs/reports/jetson_evidence_depth_audit.md", readme_ko)


if __name__ == "__main__":
    unittest.main()
