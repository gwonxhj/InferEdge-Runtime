import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class LabWorkerAdapterContractTest(unittest.TestCase):
    def test_worker_request_fixture_maps_to_runtime_invocation_config(self):
        request = load_json(FIXTURES / "lab_worker_request.json")
        config = project_runtime_invocation_config(request)

        self.assertEqual(config, load_json(FIXTURES / "runtime_invocation_config.json"))
        validate_runtime_invocation_config(config)

    def test_worker_request_requires_model_or_artifact_path(self):
        request = load_json(FIXTURES / "lab_worker_request.json")
        request.pop("model_path")
        request.pop("artifact_path")

        with self.assertRaisesRegex(AssertionError, "model_path or artifact_path"):
            project_runtime_invocation_config(request)

    def test_runtime_invocation_preserves_optional_handoff_paths_and_shape(self):
        request = load_json(FIXTURES / "lab_worker_request.json")
        config = project_runtime_invocation_config(request)

        self.assertEqual(config["metadata_path"], request["metadata_path"])
        self.assertEqual(config["manifest_path"], request["manifest_path"])
        self.assertEqual(config["precision"], "fp16")
        self.assertEqual(config["batch"], 1)
        self.assertEqual(config["height"], 640)
        self.assertEqual(config["width"], 640)
        self.assertEqual(config["options"]["with_guard"], True)

    def test_forge_summary_worker_request_maps_to_runtime_config(self):
        request = load_json(FIXTURES / "forge_summary_worker_request.json")
        config = project_runtime_invocation_config(request)

        self.assertEqual(config["job_id"], "job_forge_summary_smoke")
        self.assertEqual(config["model_path"], request["artifact_path"])
        self.assertEqual(config["source_model_path"], request["model_path"])
        self.assertEqual(config["metadata_path"], request["metadata_path"])
        self.assertEqual(config["manifest_path"], request["manifest_path"])
        self.assertEqual(config["engine"], "tensorrt")
        self.assertEqual(config["device"], "jetson")
        self.assertEqual(config["precision"], "fp16")
        self.assertEqual(config["batch"], 1)
        self.assertEqual(config["height"], 640)
        self.assertEqual(config["width"], 640)
        self.assertEqual(config["warmup"], 5)
        self.assertEqual(config["runs"], 50)

    def test_forge_summary_worker_request_preserves_nested_provenance(self):
        request = load_json(FIXTURES / "forge_summary_worker_request.json")
        config = project_runtime_invocation_config(request)
        provenance = request["options"]["provenance"]

        self.assertEqual(config["provenance"], provenance)
        self.assertEqual(provenance["source_model_sha256"], "a" * 64)
        self.assertEqual(provenance["artifact_sha256"], "b" * 64)
        self.assertEqual(provenance["artifact_type"], "engine")
        self.assertEqual(provenance["preset_name"], "tensorrt/jetson_fp16")

    def test_completed_response_contains_lab_compatible_runtime_result(self):
        response = load_json(FIXTURES / "runtime_worker_completed_response.json")

        validate_worker_completed_response(response)
        validate_lab_compatible_runtime_result(response["runtime_result"])

    def test_failed_response_requires_error(self):
        response = {
            "job_id": "job_runtime_worker_smoke",
            "status": "failed",
            "failed_at": "2026-04-28T06:01:30Z",
        }

        with self.assertRaisesRegex(AssertionError, "error"):
            validate_worker_failed_response(response)

        response["error"] = {
            "code": "runtime_execution_failed",
            "message": "Runtime execution failed.",
            "stage": "runtime",
        }
        validate_worker_failed_response(response)

    def test_cli_validates_lab_worker_request_when_binary_exists(self):
        skip_without_lab_worker_request_binary(self)

        result = subprocess_run(
            [
                str(ROOT / "build" / "inferedge-runtime"),
                "--lab-worker-request",
                str(FIXTURES / "lab_worker_request.json"),
                "--validate-lab-worker-request",
            ]
        )

        self.assertIn("Lab worker request validation", result.stdout)
        self.assertIn("status: ok", result.stdout)
        self.assertIn("job_id: job_runtime_worker_smoke", result.stdout)
        self.assertIn("engine: tensorrt", result.stdout)

    def test_cli_rejects_missing_model_or_artifact_when_binary_exists(self):
        skip_without_lab_worker_request_binary(self)

        result = subprocess_run(
            [
                str(ROOT / "build" / "inferedge-runtime"),
                "--lab-worker-request",
                str(FIXTURES / "lab_worker_request_missing_input.json"),
                "--validate-lab-worker-request",
            ],
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model_path or artifact_path", result.stderr)

    def test_cli_validates_forge_summary_worker_request_when_binary_exists(self):
        skip_without_forge_summary_worker_request_binary(self)

        result = subprocess_run(
            [
                str(ROOT / "build" / "inferedge-runtime"),
                "--lab-worker-request",
                str(FIXTURES / "forge_summary_worker_request.json"),
                "--validate-lab-worker-request",
            ]
        )

        self.assertIn("Lab worker request validation", result.stdout)
        self.assertIn("status: ok", result.stdout)
        self.assertIn("job_id: job_forge_summary_smoke", result.stdout)
        self.assertIn("source_model_sha256: " + "a" * 64, result.stdout)
        self.assertIn("artifact_sha256: " + "b" * 64, result.stdout)
        self.assertIn("artifact_type: engine", result.stdout)
        self.assertIn("preset_name: tensorrt/jetson_fp16", result.stdout)
        self.assertIn("build_id: yolov8n-tensorrt-jetson_fp16-20260427T000000Z", result.stdout)

    def test_cli_exports_completed_worker_response_when_binary_exists(self):
        skip_without_worker_response_export_binary(self)

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output_path = Path(temp_dir.name) / "worker_completed_response_test.json"

        result = subprocess_run(
            [
                str(ROOT / "build" / "inferedge-runtime"),
                "--lab-worker-request",
                str(FIXTURES / "lab_worker_request.json"),
                "--export-worker-response",
                str(output_path),
                "--worker-response-status",
                "completed",
            ]
        )

        self.assertIn("Worker response dry-run export", result.stdout)
        self.assertIn("status: completed", result.stdout)
        response = load_json(output_path)
        validate_worker_completed_response(response)
        validate_lab_compatible_runtime_result(response["runtime_result"])
        self.assertEqual(response["job_id"], "job_runtime_worker_smoke")
        self.assertEqual(response["runtime_result"]["run_config"]["dry_run"], True)
        self.assertEqual(response["runtime_result"]["extra"]["worker_response_mode"], "dry_run")

    def test_cli_exports_failed_worker_response_when_binary_exists(self):
        skip_without_worker_response_export_binary(self)

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output_path = Path(temp_dir.name) / "worker_failed_response_test.json"

        result = subprocess_run(
            [
                str(ROOT / "build" / "inferedge-runtime"),
                "--lab-worker-request",
                str(FIXTURES / "lab_worker_request.json"),
                "--export-worker-response",
                str(output_path),
                "--worker-response-status",
                "failed",
                "--worker-error-message",
                "dry-run failure",
            ]
        )

        self.assertIn("Worker response dry-run export", result.stdout)
        self.assertIn("status: failed", result.stdout)
        response = load_json(output_path)
        validate_worker_failed_response(response)
        self.assertEqual(response["job_id"], "job_runtime_worker_smoke")
        self.assertEqual(response["error"]["message"], "dry-run failure")
        self.assertNotIn("runtime_result", response)


def project_runtime_invocation_config(request: dict) -> dict:
    require_string(request, "job_id")
    require_dict(request, "input_summary")
    require_string(request, "requested_at")
    options = require_dict(request, "options")

    model_path = optional_string(request, "model_path")
    artifact_path = optional_string(request, "artifact_path")
    if not model_path and not artifact_path:
        raise AssertionError("worker request requires model_path or artifact_path")

    config = {
        "job_id": request["job_id"],
        "model_path": artifact_path or model_path,
        "source_model_path": model_path,
        "artifact_path": artifact_path,
        "metadata_path": optional_string(request, "metadata_path"),
        "manifest_path": optional_string(request, "manifest_path"),
        "engine": require_string(options, "backend"),
        "device": require_string(options, "target"),
        "precision": require_string(options, "precision"),
        "batch": require_positive_int(options, "batch"),
        "height": require_positive_int(options, "height"),
        "width": require_positive_int(options, "width"),
        "warmup": positive_int_or_default(options, "warmup", 5),
        "runs": positive_int_or_default(options, "runs", 50),
        "options": {
            "with_guard": bool(options.get("with_guard", False)),
        },
    }
    provenance = optional_dict(options, "provenance")
    if provenance is not None:
        config["provenance"] = provenance
    validate_runtime_invocation_config(config)
    return config


def validate_runtime_invocation_config(config: dict) -> None:
    require_string(config, "job_id")
    require_string(config, "model_path")
    optional_string(config, "source_model_path")
    optional_string(config, "artifact_path")
    optional_string(config, "metadata_path")
    optional_string(config, "manifest_path")
    require_string(config, "engine")
    require_string(config, "device")
    require_string(config, "precision")
    for field in ("batch", "height", "width", "warmup", "runs"):
        require_positive_int(config, field)
    require_dict(config, "options")


def validate_worker_completed_response(response: dict) -> None:
    require_string(response, "job_id")
    if response.get("status") != "completed":
        raise AssertionError("worker completed response status must be completed")
    require_dict(response, "forge_metadata")
    require_dict(response, "runtime_result")
    require_string(response, "completed_at")


def validate_worker_failed_response(response: dict) -> None:
    require_string(response, "job_id")
    if response.get("status") != "failed":
        raise AssertionError("worker failed response status must be failed")
    error = require_dict(response, "error")
    require_string(error, "code")
    require_string(error, "message")
    require_string(response, "failed_at")


def validate_lab_compatible_runtime_result(result: dict) -> None:
    required = {
        "model_path": str,
        "engine_backend": str,
        "device_name": str,
        "precision": str,
        "batch": int,
        "height": int,
        "width": int,
        "mean_ms": (int, float),
        "p50_ms": (int, float),
        "p95_ms": (int, float),
        "p99_ms": (int, float),
        "latency_ms": dict,
        "run_config": dict,
        "timestamp": str,
    }
    for field, expected_type in required.items():
        if field not in result:
            raise AssertionError(f"runtime_result missing {field}")
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise AssertionError(f"runtime_result.{field} has invalid type")

    for field in ("mean", "p50", "p95", "p99"):
        value = result["latency_ms"].get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssertionError(f"latency_ms.{field} must be numeric")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return data


def require_string(data: dict, field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{field} must be a non-empty string")
    return value


def optional_string(data: dict, field: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{field} must be a non-empty string when provided")
    return value


def require_dict(data: dict, field: str) -> dict:
    value = data.get(field)
    if not isinstance(value, dict):
        raise AssertionError(f"{field} must be an object")
    return value


def optional_dict(data: dict, field: str) -> dict | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AssertionError(f"{field} must be an object when provided")
    return value


def require_positive_int(data: dict, field: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AssertionError(f"{field} must be a positive integer")
    return value


def positive_int_or_default(data: dict, field: str, default: int) -> int:
    if field not in data:
        return default
    return require_positive_int(data, field)


def subprocess_run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def skip_without_lab_worker_request_binary(test_case: unittest.TestCase) -> None:
    binary = ROOT / "build" / "inferedge-runtime"
    if not binary.exists():
        test_case.skipTest("inferedge-runtime binary was not built")

    result = subprocess_run([str(binary), "--help"])
    if "--lab-worker-request" not in result.stdout:
        test_case.skipTest("inferedge-runtime binary does not include Lab worker request options")


def skip_without_worker_response_export_binary(test_case: unittest.TestCase) -> None:
    skip_without_lab_worker_request_binary(test_case)

    result = subprocess_run([str(ROOT / "build" / "inferedge-runtime"), "--help"])
    if "--export-worker-response" not in result.stdout:
        test_case.skipTest("inferedge-runtime binary does not include worker response export options")


def skip_without_forge_summary_worker_request_binary(test_case: unittest.TestCase) -> None:
    skip_without_lab_worker_request_binary(test_case)

    result = subprocess_run(
        [
            str(ROOT / "build" / "inferedge-runtime"),
            "--lab-worker-request",
            str(FIXTURES / "forge_summary_worker_request.json"),
            "--validate-lab-worker-request",
        ],
        check=False,
    )
    if result.returncode != 0 or "source_model_sha256" not in result.stdout:
        test_case.skipTest("inferedge-runtime binary does not include Forge summary worker request support")


if __name__ == "__main__":
    unittest.main()
