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


if __name__ == "__main__":
    unittest.main()
