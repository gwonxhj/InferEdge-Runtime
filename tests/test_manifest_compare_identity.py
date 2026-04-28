import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "inferedge-runtime"
FIXTURES = ROOT / "tests" / "fixtures"


class ManifestCompareIdentityTest(unittest.TestCase):
    def test_manifest_source_model_path_wins_over_explicit_engine_path(self):
        skip_without_runtime_binary(self)
        output_path = ROOT / "results" / "manifest_source_identity_fp32.json"

        run_runtime(
            [
                "--manifest",
                str(FIXTURES / "manifest_source_identity_fp32.json"),
                "--model",
                "builds/yolov8n__jetson__tensorrt__jetson_fp32/model.engine",
                "--engine",
                "tensorrt",
                "--device",
                "jetson",
                "--height",
                "640",
                "--width",
                "640",
                "--warmup",
                "1",
                "--runs",
                "1",
                "--output",
                str(output_path),
            ]
        )

        result = load_json(output_path)
        extra = result["extra"]
        self.assertEqual(result["compare_key"], "yolov8n__b1__h640w640__fp32")
        self.assertEqual(extra["compare_model_name"], "yolov8n")
        self.assertEqual(extra["compare_model_source"], "manifest_source_model")
        self.assertEqual(extra["source_model_path"], "models/onnx/yolov8n.onnx")
        self.assertEqual(extra["source_model_sha256"], "a" * 64)
        self.assertEqual(extra["runtime_artifact_sha256"], "b" * 64)
        self.assertEqual(
            extra["runtime_artifact_path"],
            "builds/yolov8n__jetson__tensorrt__jetson_fp32/model.engine",
        )

    def test_manifest_fixture_preserves_source_model_even_when_artifact_name_is_engine(self):
        fixture = load_json(FIXTURES / "forge_handoff_manifest.json")
        self.assertEqual(fixture["source_model"]["path"], "models/yolov8n.onnx")
        self.assertEqual(fixture["artifact"]["model_name"], "model.engine")

    def test_without_manifest_falls_back_to_model_path_stem(self):
        skip_without_runtime_binary(self)
        output_path = ROOT / "results" / "manifest_identity_fallback.json"

        run_runtime(
            [
                "--model",
                "builds/yolov8n__jetson__tensorrt__jetson_fp32/model.engine",
                "--engine",
                "tensorrt",
                "--device",
                "jetson",
                "--height",
                "640",
                "--width",
                "640",
                "--warmup",
                "1",
                "--runs",
                "1",
                "--output",
                str(output_path),
            ]
        )

        result = load_json(output_path)
        extra = result["extra"]
        self.assertEqual(result["compare_key"], "model__b1__h640w640__fp32")
        self.assertEqual(extra["compare_model_name"], "model")
        self.assertEqual(extra["compare_model_source"], "model_path")


def run_runtime(args: list[str]) -> None:
    subprocess.run(
        [str(BINARY), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return data


def skip_without_runtime_binary(test_case: unittest.TestCase) -> None:
    if not BINARY.exists():
        test_case.skipTest("inferedge-runtime binary was not built")


if __name__ == "__main__":
    unittest.main()
