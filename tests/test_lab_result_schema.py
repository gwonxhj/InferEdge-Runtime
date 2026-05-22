import json
import os
import subprocess
import tempfile
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
    "jetson_evidence": dict,
    "timestamp": str,
}


class LabCompatibleRuntimeResultSchemaTest(unittest.TestCase):
    def test_fixture_satisfies_lab_compatible_schema(self):
        path = ROOT / "tests" / "fixtures" / "lab_compatible_result.json"
        result = load_json(path)

        validate_lab_compatible_result(result)

    def test_timeout_observed_fixture_satisfies_lab_compatible_schema(self):
        path = ROOT / "tests" / "fixtures" / "runtime_timeout_observed_result.json"
        result = load_json(path)

        validate_lab_compatible_result(result)
        health = result["runtime_health_snapshot"]
        self.assertEqual(health["timeout_policy"], "latency_threshold")
        self.assertEqual(health["timeout_budget_ms"], 10)
        self.assertTrue(health["timeout_observed"])
        error = result["runtime_error_classification"]
        self.assertEqual(error["category"], "runtime_timeout_observed")
        self.assertTrue(error["timeout_observed"])
        self.assertTrue(error["retryable"])

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
    for field in ("power_mode", "jetson_clocks", "tegrastats_log_path"):
        if field not in run_config:
            raise AssertionError(f"run_config.{field} is required")

    jetson_evidence = result["jetson_evidence"]
    for field in ("power_mode", "jetson_clocks", "tegrastats_log_path", "tegrastats_summary"):
        if field not in jetson_evidence:
            raise AssertionError(f"jetson_evidence.{field} is required")

    tegrastats_summary = jetson_evidence["tegrastats_summary"]
    if not isinstance(tegrastats_summary, dict):
        raise AssertionError("jetson_evidence.tegrastats_summary must be an object")
    for field in (
        "status",
        "sample_count",
        "ram_used_mb_avg",
        "ram_used_mb_max",
        "ram_total_mb",
        "max_temp_c",
        "max_temp_name",
        "vdd_in_mw_avg",
        "vdd_in_mw_max",
    ):
        if field not in tegrastats_summary:
            raise AssertionError(f"jetson_evidence.tegrastats_summary.{field} is required")

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

    validate_optional_runtime_operation_evidence(result)
    validate_optional_runtime_telemetry(result)


def validate_optional_runtime_operation_evidence(result: dict) -> None:
    health = result.get("runtime_health_snapshot")
    if health is not None:
        if not isinstance(health, dict):
            raise AssertionError("runtime_health_snapshot must be an object when present")
        for field in ("schema_version", "status", "engine_backend", "device", "input_mode"):
            if field not in health:
                raise AssertionError(f"runtime_health_snapshot.{field} is required")
            if not isinstance(health[field], str):
                raise AssertionError(f"runtime_health_snapshot.{field} must be a string")
        for field in ("warmup", "runs"):
            if not isinstance(health.get(field), int):
                raise AssertionError(f"runtime_health_snapshot.{field} must be an integer")
        for field in ("success", "run_once", "timeout_observed"):
            if not isinstance(health.get(field), bool):
                raise AssertionError(f"runtime_health_snapshot.{field} must be a boolean")
        for field in (
            "engine_available",
            "latency_budget_exceeded",
            "deadline_missed",
            "thermal_memory_evidence_available",
        ):
            if field in health and not isinstance(health[field], bool):
                raise AssertionError(f"runtime_health_snapshot.{field} must be a boolean when present")
        for field in ("tegrastats_status", "engine_status_message"):
            if field in health and not isinstance(health[field], str):
                raise AssertionError(f"runtime_health_snapshot.{field} must be a string when present")
        timeout_budget = health.get("timeout_budget_ms")
        if timeout_budget is not None and (
            isinstance(timeout_budget, bool) or not isinstance(timeout_budget, int)
        ):
            raise AssertionError("runtime_health_snapshot.timeout_budget_ms must be an integer or null")

    error = result.get("runtime_error_classification")
    if error is not None:
        if not isinstance(error, dict):
            raise AssertionError("runtime_error_classification must be an object when present")
        for field in ("schema_version", "status", "category", "message"):
            if field not in error:
                raise AssertionError(f"runtime_error_classification.{field} is required")
            if not isinstance(error[field], str):
                raise AssertionError(f"runtime_error_classification.{field} must be a string")
        for field in ("timeout_observed", "retryable"):
            if not isinstance(error.get(field), bool):
                raise AssertionError(f"runtime_error_classification.{field} must be a boolean")
        if "severity" in error and not isinstance(error["severity"], str):
            raise AssertionError("runtime_error_classification.severity must be a string when present")
        if "retry_hint" in error and not isinstance(error["retry_hint"], str):
            raise AssertionError("runtime_error_classification.retry_hint must be a string when present")

    events = result.get("runtime_events")
    if events is not None:
        if not isinstance(events, list):
            raise AssertionError("runtime_events must be an array when present")
        event_types = []
        event_indexes = []
        for event in events:
            if not isinstance(event, dict):
                raise AssertionError("runtime_events items must be objects")
            event_type = event.get("type")
            if not isinstance(event_type, str) or not event_type:
                raise AssertionError("runtime_events[].type must be a non-empty string")
            event_types.append(event_type)
            if "schema_version" in event and not isinstance(event["schema_version"], str):
                raise AssertionError("runtime_events[].schema_version must be a string when present")
            if "event_index" in event:
                if isinstance(event["event_index"], bool) or not isinstance(event["event_index"], int):
                    raise AssertionError("runtime_events[].event_index must be an integer when present")
                event_indexes.append(event["event_index"])
        if event_indexes and event_indexes != list(range(len(event_indexes))):
            raise AssertionError("runtime_events[].event_index must be sequential when present")
        for expected in ("runtime_configured", "benchmark_completed", "runtime_error_classified"):
            if expected not in event_types:
                raise AssertionError(f"runtime_events must include {expected}")

    operation_summary = result.get("runtime_operation_summary")
    if operation_summary is not None:
        if not isinstance(operation_summary, dict):
            raise AssertionError("runtime_operation_summary must be an object when present")
        for field in (
            "schema_version",
            "observation_scope",
            "decision_owner",
            "scheduler_owner",
            "health_status",
            "health_reason",
            "error_category",
            "recommended_action",
        ):
            if field not in operation_summary:
                raise AssertionError(f"runtime_operation_summary.{field} is required")
            if not isinstance(operation_summary[field], str):
                raise AssertionError(f"runtime_operation_summary.{field} must be a string")
        if operation_summary["schema_version"] != "inferedge-runtime-operation-summary-v1":
            raise AssertionError("runtime_operation_summary.schema_version is invalid")
        if operation_summary["decision_owner"] != "lab":
            raise AssertionError("runtime_operation_summary.decision_owner must remain lab")
        if operation_summary["scheduler_owner"] != "orchestrator":
            raise AssertionError("runtime_operation_summary.scheduler_owner must remain orchestrator")
        for field in (
            "production_cancellation",
            "retryable",
            "timeout_observed",
            "latency_budget_exceeded",
            "deadline_missed",
            "thermal_memory_evidence_available",
        ):
            if not isinstance(operation_summary.get(field), bool):
                raise AssertionError(f"runtime_operation_summary.{field} must be a boolean")
        for field in ("risk_labels", "evidence_gaps"):
            values = operation_summary.get(field)
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise AssertionError(f"runtime_operation_summary.{field} must be a string array")


def validate_optional_runtime_telemetry(result: dict) -> None:
    telemetry = result.get("runtime_telemetry")
    if telemetry is None:
        return
    if not isinstance(telemetry, dict):
        raise AssertionError("runtime_telemetry must be an object when present")
    for field in (
        "schema_version",
        "evidence_role",
        "collection_mode",
        "source_result_schema_version",
        "telemetry_timestamp",
        "sequence_scope",
        "engine_backend",
        "device",
        "input_mode",
        "power_mode",
        "jetson_clocks",
    ):
        if field not in telemetry:
            raise AssertionError(f"runtime_telemetry.{field} is required")
        if not isinstance(telemetry[field], str):
            raise AssertionError(f"runtime_telemetry.{field} must be a string")
    if telemetry["schema_version"] != "inferedge-runtime-telemetry-v1":
        raise AssertionError("runtime_telemetry.schema_version is invalid")
    if telemetry["collection_mode"] != "single_result_export":
        raise AssertionError("runtime_telemetry.collection_mode must be single_result_export")
    if isinstance(telemetry.get("execution_sequence_id"), bool) or not isinstance(
        telemetry.get("execution_sequence_id"), int
    ):
        raise AssertionError("runtime_telemetry.execution_sequence_id must be an integer")
    if not isinstance(telemetry.get("production_monitoring"), bool):
        raise AssertionError("runtime_telemetry.production_monitoring must be a boolean")

    latency = telemetry.get("latency")
    if not isinstance(latency, dict):
        raise AssertionError("runtime_telemetry.latency must be an object")
    for field in (
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "fps",
        "inference_interval_ms",
        "rolling_latency_mean_ms",
        "rolling_latency_std_ms",
    ):
        value = latency.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssertionError(f"runtime_telemetry.latency.{field} must be numeric")
    if isinstance(latency.get("sample_count"), bool) or not isinstance(latency.get("sample_count"), int):
        raise AssertionError("runtime_telemetry.latency.sample_count must be an integer")

    resource = telemetry.get("resource")
    if not isinstance(resource, dict):
        raise AssertionError("runtime_telemetry.resource must be an object")
    for field in ("telemetry_source", "tegrastats_status"):
        if not isinstance(resource.get(field), str):
            raise AssertionError(f"runtime_telemetry.resource.{field} must be a string")
    if isinstance(resource.get("tegrastats_sample_count"), bool) or not isinstance(
        resource.get("tegrastats_sample_count"), int
    ):
        raise AssertionError("runtime_telemetry.resource.tegrastats_sample_count must be an integer")

    operation = telemetry.get("operation")
    if not isinstance(operation, dict):
        raise AssertionError("runtime_telemetry.operation must be an object")
    for field in ("timeout_observed", "latency_budget_exceeded", "deadline_missed"):
        if not isinstance(operation.get(field), bool):
            raise AssertionError(f"runtime_telemetry.operation.{field} must be a boolean")

    missing_fields = telemetry.get("missing_fields")
    if not isinstance(missing_fields, list) or not all(isinstance(item, str) for item in missing_fields):
        raise AssertionError("runtime_telemetry.missing_fields must be a string array")


class JetsonEvidenceContractTest(unittest.TestCase):
    def test_runtime_binary_parses_tegrastats_log_when_available(self):
        runtime_binary = ROOT / "build" / "inferedge-runtime"
        if not runtime_binary.exists():
            self.skipTest("runtime binary was not built")

        tegrastats_log = ROOT / "tests" / "fixtures" / "tegrastats_sample.log"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "jetson_evidence_contract.json"
            subprocess.run(
                [
                    str(runtime_binary),
                    "--model",
                    "models/sample.onnx",
                    "--engine",
                    "onnxruntime",
                    "--device",
                    "cpu",
                    "--power-mode",
                    "15W",
                    "--jetson-clocks",
                    "on",
                    "--tegrastats-log",
                    str(tegrastats_log),
                    "--warmup",
                    "1",
                    "--runs",
                    "1",
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            result = load_json(output_path)

        validate_lab_compatible_result(result)
        self.assertEqual(result["runtime_health_snapshot"]["schema_version"], "inferedge-runtime-health-v1")
        self.assertIn(result["runtime_health_snapshot"]["status"], {"ok", "degraded", "error"})
        if result["success"]:
            self.assertEqual(result["runtime_error_classification"]["category"], "none")
        else:
            self.assertNotEqual(result["runtime_error_classification"]["category"], "none")
        event_types = {event["type"] for event in result["runtime_events"]}
        self.assertIn("runtime_configured", event_types)
        self.assertIn("benchmark_completed", event_types)
        self.assertEqual(result["run_config"]["power_mode"], "15W")
        self.assertEqual(result["run_config"]["jetson_clocks"], "on")
        summary = result["jetson_evidence"]["tegrastats_summary"]
        self.assertEqual(summary["status"], "parsed")
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["ram_used_mb_max"], 2304)
        self.assertEqual(summary["ram_total_mb"], 7620)
        self.assertEqual(summary["max_temp_name"], "cpu")
        self.assertEqual(summary["max_temp_c"], 45.0)
        self.assertEqual(summary["vdd_in_mw_max"], 3100)


if __name__ == "__main__":
    unittest.main()
