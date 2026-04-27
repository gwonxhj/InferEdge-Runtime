import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
BINARY = ROOT / "build" / "inferedge-runtime"


class ForgeHandoffFixtureTest(unittest.TestCase):
    def test_manifest_fixture_contains_runtime_handoff_fields(self):
        handoff = load_json(FIXTURES / "forge_handoff_manifest.json")

        runtime = handoff["runtime"]
        artifact = handoff["artifact"]
        source_model = handoff["source_model"]

        self.assertEqual(runtime["engine"], "tensorrt")
        self.assertEqual(runtime["device"], "jetson")
        self.assertEqual(runtime["precision"], "fp16")
        self.assertEqual(runtime["batch"], 1)
        self.assertEqual(runtime["height"], 640)
        self.assertEqual(runtime["width"], 640)
        self.assertEqual(runtime["model_path"], artifact["model_path"])
        self.assertEqual(runtime["artifact_path"], artifact["path"])
        self.assertEqual(artifact["format"], "engine")
        self.assertTrue(artifact["sha256"])
        self.assertTrue(source_model["sha256"])
        self.assertTrue(handoff["build"]["preset_name"])
        self.assertTrue(handoff["build"]["build_id"])

    def test_metadata_fixture_contains_runtime_handoff_fields(self):
        handoff = load_json(FIXTURES / "forge_handoff_metadata.json")

        runtime = handoff["lab_compat"]["runtime"]
        artifact = handoff["artifacts"][0]

        self.assertEqual(runtime["engine"], "tensorrt")
        self.assertEqual(runtime["device"], "jetson")
        self.assertEqual(runtime["precision"], "fp16")
        self.assertEqual(runtime["requested_batch"], 1)
        self.assertEqual(runtime["requested_height"], 640)
        self.assertEqual(runtime["requested_width"], 640)
        self.assertEqual(runtime["runtime_artifact_path"], artifact["path"])
        self.assertEqual(artifact["format"], "engine")
        self.assertTrue(artifact["sha256"])

    def test_cli_validates_forge_manifest_when_binary_exists(self):
        skip_without_forge_handoff_binary(self)

        result = subprocess.run(
            [
                str(BINARY),
                "--forge-manifest",
                str(FIXTURES / "forge_handoff_manifest.json"),
                "--validate-forge-handoff",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("status: ok", result.stdout)
        self.assertIn("engine: tensorrt", result.stdout)
        self.assertIn("device: jetson", result.stdout)
        self.assertIn("precision: fp16", result.stdout)

    def test_cli_validates_forge_metadata_when_binary_exists(self):
        skip_without_forge_handoff_binary(self)

        result = subprocess.run(
            [
                str(BINARY),
                "--forge-metadata",
                str(FIXTURES / "forge_handoff_metadata.json"),
                "--validate-forge-handoff",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("status: ok", result.stdout)
        self.assertIn("engine: tensorrt", result.stdout)
        self.assertIn("device: jetson", result.stdout)
        self.assertIn("precision: fp16", result.stdout)

    def test_cli_rejects_missing_required_forge_field_when_binary_exists(self):
        skip_without_forge_handoff_binary(self)

        result = subprocess.run(
            [
                str(BINARY),
                "--forge-manifest",
                str(FIXTURES / "forge_handoff_missing_engine.json"),
                "--validate-forge-handoff",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing runtime engine", result.stderr)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return data


def skip_without_forge_handoff_binary(test_case: unittest.TestCase) -> None:
    if not BINARY.exists():
        test_case.skipTest("inferedge-runtime binary was not built")

    result = subprocess.run(
        [str(BINARY), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if "--forge-manifest" not in result.stdout:
        test_case.skipTest("inferedge-runtime binary does not include Forge handoff options")


if __name__ == "__main__":
    unittest.main()
