import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


LAB_COMPATIBLE_REQUIRED_FIELDS = {
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
    "run_config": dict,
    "timestamp": str,
}


class LabCompatibleRuntimeResultSchemaTest(unittest.TestCase):
    def test_fixture_satisfies_lab_compatible_schema(self):
        path = ROOT / "tests" / "fixtures" / "lab_compatible_result.json"
        result = load_json(path)

        validate_lab_compatible_result(result)

    def test_runtime_output_satisfies_lab_compatible_schema_when_provided(self):
        result_env = os.environ.get("INFEREDGE_RUNTIME_RESULT_JSON")
        if not result_env:
            self.skipTest("runtime result JSON path was not provided")

        result_path = Path(result_env)
        result = load_json(result_path)

        validate_lab_compatible_result(result)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return data


def validate_lab_compatible_result(result: dict) -> None:
    missing = [key for key in LAB_COMPATIBLE_REQUIRED_FIELDS if key not in result]
    if missing:
        raise AssertionError(f"missing Lab-compatible fields: {missing}")

    for field, expected_type in LAB_COMPATIBLE_REQUIRED_FIELDS.items():
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise AssertionError(
                f"{field} must be {expected_type}, got {type(value).__name__}"
            )

    if not result["model_path"]:
        raise AssertionError("model_path must be non-empty")

    if not result["engine_backend"]:
        raise AssertionError("engine_backend must be non-empty")

    if not result["device_name"]:
        raise AssertionError("device_name must be non-empty")

    if result["precision"].lower() not in {"fp32", "fp16", "int8", "unknown"}:
        raise AssertionError(f"unexpected precision: {result['precision']}")

    for field in ("batch", "height", "width"):
        if result[field] <= 0:
            raise AssertionError(f"{field} must be positive")

    run_config = result["run_config"]
    for field in ("batch", "height", "width", "warmup", "runs"):
        if field not in run_config:
            raise AssertionError(f"run_config.{field} is required")

    latency = result.get("latency_ms")
    if not isinstance(latency, dict):
        raise AssertionError("latency_ms must be an object")

    for field in ("mean", "p50", "p95", "p99"):
        value = latency.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssertionError(f"latency_ms.{field} must be numeric")

    if result.get("compare_key") is not None and not isinstance(result["compare_key"], str):
        raise AssertionError("compare_key must be a string when present")

    if result.get("backend_key") is not None and not isinstance(result["backend_key"], str):
        raise AssertionError("backend_key must be a string when present")


if __name__ == "__main__":
    unittest.main()
