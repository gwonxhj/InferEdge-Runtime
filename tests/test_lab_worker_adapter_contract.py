import json
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
        "warmup": require_positive_int(options, "warmup"),
        "runs": require_positive_int(options, "runs"),
        "options": {
            "with_guard": bool(options.get("with_guard", False)),
        },
    }
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


def require_positive_int(data: dict, field: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AssertionError(f"{field} must be a positive integer")
    return value


if __name__ == "__main__":
    unittest.main()
