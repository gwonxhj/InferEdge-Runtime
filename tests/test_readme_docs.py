import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReadmeDocsTest(unittest.TestCase):
    def test_english_readme_keeps_language_link_and_role_boundary(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        required_markers = [
            "Language: English | [한국어](README.ko.md)",
            "## Role Boundary At A Glance",
            "Runtime owns",
            "Runtime does not own",
            "Lab-compatible `result.json`",
            "`runtime_health_snapshot`",
            "EdgeEnv registry",
            "runtime regression owner",
            "multi-workload scheduler",
            "queue/drop/fallback owner",
            "production inference server",
            "same-condition regression",
            "thermal endurance validation",
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, readme)

    def test_korean_readme_keeps_language_link_and_role_boundary(self):
        readme = (ROOT / "README.ko.md").read_text(encoding="utf-8")

        required_markers = [
            "언어: [English](README.md) | 한국어",
            "## 역할 경계 한눈에 보기",
            "Runtime이 소유하는 것",
            "Runtime이 소유하지 않는 것",
            "Lab-compatible `result.json`",
            "`runtime_health_snapshot`",
            "EdgeEnv registry",
            "runtime regression owner",
            "multi-workload scheduler",
            "queue/drop/fallback owner",
            "production inference server",
            "same-condition regression",
            "thermal endurance validation",
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, readme)

    def test_agent_runtime_result_contract_has_korean_quick_guide(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_ko = (ROOT / "README.ko.md").read_text(encoding="utf-8")
        contract = (ROOT / "docs" / "agent_runtime_result_contract.md").read_text(
            encoding="utf-8"
        )
        contract_ko = (
            ROOT / "docs" / "agent_runtime_result_contract.ko.md"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "Language: English | [한국어](agent_runtime_result_contract.ko.md)",
            contract,
        )
        self.assertIn(
            "언어: [English](agent_runtime_result_contract.md) | 한국어",
            contract_ko,
        )
        self.assertIn("대표/canonical 문서", contract_ko)
        self.assertIn(
            "[Agent Runtime Result Contract](agent_runtime_result_contract.md)",
            contract_ko,
        )
        self.assertIn(
            "[docs/agent_runtime_result_contract.md]"
            "(docs/agent_runtime_result_contract.md)",
            readme,
        )
        self.assertIn(
            "[한국어: Agent Runtime Result Contract quick guide]"
            "(docs/agent_runtime_result_contract.ko.md)",
            readme,
        )
        self.assertIn(
            "[Agent Runtime Result Contract]"
            "(docs/agent_runtime_result_contract.ko.md)",
            readme_ko,
        )
        self.assertIn(
            "[English contract](docs/agent_runtime_result_contract.md)",
            readme_ko,
        )

        for marker in [
            "Lab-compatible `result.json`",
            "`compare_key`, `backend_key`, `run_config`",
            "`runtime_health_snapshot`",
            "`runtime_error_classification`",
            "`runtime_events`",
            "`runtime_operation_summary`",
            "`runtime_telemetry.history_seed`",
            "`decision_owner: lab`",
            "`scheduler_owner: orchestrator`",
            "`registry_owner: edgeenv`",
            "production request cancellation",
            "production worker daemon",
            "multi-workload scheduler",
            "queue/drop/fallback owner",
            "runtime regression owner",
            "production observability platform",
            "Lab-owned deployment decision",
            "same-condition regression",
            "production_monitoring: false",
            "Jetson 필요 여부",
        ]:
            with self.subTest(marker=marker):
                self.assertIn(marker, contract_ko)


if __name__ == "__main__":
    unittest.main()
