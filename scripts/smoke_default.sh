#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

OUTPUT_PATH="results/smoke_default.json"
PRETTY_PATH="/tmp/inferedge_runtime_smoke_default_pretty.json"
JETSON_REPORT_PATH="/tmp/inferedge_runtime_jetson_evidence.md"
POWER_REPORT_PATH="/tmp/inferedge_runtime_power_mode_compare.md"

cmake -S . -B build
cmake --build build

./build/inferedge-runtime --help
./build/inferedge-runtime --version
./build/inferedge-runtime \
  --model models/sample.onnx \
  --engine onnxruntime \
  --device cpu \
  --batch 1 \
  --height 224 \
  --width 224 \
  --warmup 1 \
  --runs 1 \
  --output "${OUTPUT_PATH}"

python3 -m json.tool "${OUTPUT_PATH}" > "${PRETTY_PATH}"

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/smoke_default.json").read_text())
required = [
    "model_name",
    "model_path",
    "engine_name",
    "engine_backend",
    "device_name",
    "batch",
    "height",
    "width",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "fps_value",
    "success",
    "status",
    "jetson_evidence",
]
missing = [key for key in required if key not in data]
assert not missing, missing
assert data["status"] == "skipped", data["status"]
assert data["success"] is False, data["success"]
assert data["run_config"]["power_mode"] == "unknown", data["run_config"]
assert data["run_config"]["jetson_clocks"] == "unknown", data["run_config"]
assert data["jetson_evidence"]["tegrastats_summary"]["status"] == "not_provided"
health = data["runtime_health_snapshot"]
assert health["status"] == "degraded", health
assert health["success"] is False
assert health["health_reason"] == "backend_unavailable_or_not_enabled", health
assert health["timeout_policy"] == "not_configured"
assert health["timeout_observed"] is False
error = data["runtime_error_classification"]
assert error["status"] == "classified", error
assert error["category"] == "runtime_execution_skipped", error
assert error["severity"] == "warning", error
assert error["retryable"] is True, error
assert error["retry_hint"] == "check_backend_availability", error
events = {event["type"]: event for event in data["runtime_events"]}
assert events["runtime_error_classified"]["category"] == "runtime_execution_skipped"
assert events["runtime_error_classified"]["retryable"] is True
assert events["runtime_error_classified"]["health_reason"] == health["health_reason"]
summary_event = events["runtime_operation_summary_recorded"]
assert summary_event["health_reason"] == health["health_reason"], summary_event
assert summary_event["recommended_action"] == "check_backend_availability", summary_event
operation = data["runtime_operation_summary"]
assert operation["schema_version"] == "inferedge-runtime-operation-summary-v1", operation
assert operation["decision_owner"] == "lab", operation
assert operation["scheduler_owner"] == "orchestrator", operation
assert operation["production_cancellation"] is False, operation
assert operation["health_status"] == "degraded", operation
assert operation["health_reason"] == health["health_reason"], operation
assert operation["recommended_action"] == "check_backend_availability", operation
assert "runtime_execution_skipped" in operation["risk_labels"], operation
assert "backend_unavailable" in operation["risk_labels"], operation
assert "timeout_policy_not_configured" in operation["evidence_gaps"], operation
telemetry = data["runtime_telemetry"]
assert telemetry["schema_version"] == "inferedge-runtime-telemetry-v1", telemetry
assert telemetry["collection_mode"] == "single_result_export", telemetry
assert telemetry["source_result_schema_version"] == "inferedge-runtime-result-v1", telemetry
assert telemetry["engine_backend"] == data["engine_backend"], telemetry
assert telemetry["latency"]["mean_ms"] == data["mean_ms"], telemetry
assert telemetry["operation"]["timeout_observed"] == health["timeout_observed"], telemetry
assert telemetry["production_monitoring"] is False, telemetry
assert "queue_depth" in telemetry["missing_fields"], telemetry
coverage = telemetry["coverage"]
assert coverage["schema_version"] == "inferedge-runtime-telemetry-coverage-v1", coverage
assert coverage["comparability_owner"] == "edgeenv", coverage
assert coverage["missing_telemetry_is_failure"] is False, coverage
assert "queue_depth" in coverage["expected_fields"], coverage
assert "queue_depth" in coverage["missing_fields"], coverage
assert "telemetry_timestamp" in coverage["observed_fields"], coverage
assert coverage["missing_fields"] == telemetry["missing_fields"], coverage
assert events["runtime_telemetry_recorded"]["observed_field_count"] == coverage["observed_field_count"]
assert events["runtime_telemetry_recorded"]["missing_field_count"] == coverage["missing_field_count"]
assert events["runtime_telemetry_recorded"]["schema"] == "inferedge-runtime-telemetry-v1"
PY

INFEREDGE_RUNTIME_RESULT_JSON="${OUTPUT_PATH}" python3 tests/test_lab_result_schema.py

AGENT_OUTPUT_PATH="results/smoke_agent_result.json"
./build/inferedge-runtime \
  --agent-manifest tests/fixtures/agent_manifest.json \
  --agent-task-id task_camera_frame_0001 \
  --agent-queue-wait-ms 7 \
  --agent-fallback-used \
  --timeout-ms 1 \
  --warmup 1 \
  --runs 1 \
  --output "${AGENT_OUTPUT_PATH}"

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/smoke_agent_result.json").read_text())
agent = data.get("agent")
assert isinstance(agent, dict), data
assert agent["schema_version"] == "inferedge-runtime-agent-task-v1"
assert agent["source_contract"] == "inferedge-agent-manifest-v1"
assert agent["agent_id"] == "vision_detector"
assert agent["task_id"] == "task_camera_frame_0001"
assert agent["agent_type"] == "vision"
assert agent["scheduled_priority"] == 90
assert agent["latency_budget_ms"] == 33
assert agent["queue_wait_ms"] == 7
assert agent["fallback_used"] is True
health = data["runtime_health_snapshot"]
assert health["timeout_policy"] == "latency_threshold"
assert health["timeout_budget_ms"] == 1
assert health["timeout_observed"] is False
assert "health_reason" in health
assert health["latency_budget_ms"] == 33
assert "latency_budget_exceeded" in health
assert "deadline_missed" in health
assert health["tegrastats_status"] == "not_provided"
error = data["runtime_error_classification"]
assert error["timeout_observed"] is False
assert error["timeout_budget_ms"] == 1
assert error["severity"] in {"none", "warning", "error"}
assert "retry_hint" in error
events = {event["type"]: event for event in data["runtime_events"]}
assert events["runtime_error_classified"]["timeout_policy"] == "latency_threshold"
assert events["runtime_error_classified"]["timeout_budget_ms"] == 1
assert events["runtime_error_classified"]["health_reason"] == health["health_reason"]
assert events["benchmark_completed"]["latency_budget_ms"] == 33
assert events["runtime_operation_summary_recorded"]["health_reason"] == health["health_reason"]
assert [event["event_index"] for event in data["runtime_events"]] == list(range(len(data["runtime_events"])))
operation = data["runtime_operation_summary"]
assert operation["decision_owner"] == "lab", operation
assert operation["scheduler_owner"] == "orchestrator", operation
assert operation["production_cancellation"] is False, operation
assert operation["health_reason"] == health["health_reason"], operation
assert isinstance(operation["risk_labels"], list), operation
assert isinstance(operation["evidence_gaps"], list), operation
telemetry = data["runtime_telemetry"]
assert telemetry["schema_version"] == "inferedge-runtime-telemetry-v1", telemetry
assert telemetry["operation"]["timeout_observed"] == health["timeout_observed"], telemetry
assert telemetry["operation"]["latency_budget_exceeded"] == health["latency_budget_exceeded"], telemetry
assert telemetry["operation"]["deadline_missed"] == health["deadline_missed"], telemetry
assert telemetry["latency"]["p99_ms"] == data["p99_ms"], telemetry
coverage = telemetry["coverage"]
assert coverage["schema_version"] == "inferedge-runtime-telemetry-coverage-v1", coverage
assert coverage["comparability_owner"] == "edgeenv", coverage
assert coverage["missing_fields"] == telemetry["missing_fields"], coverage
assert "runtime_telemetry_recorded" in events, events
assert data["extra"]["agent_manifest_recorded"] is True
PY

python3 tests/test_agent_runtime_result_contract.py

./build/inferedge-runtime \
  --report-jetson-evidence \
  --result-json tests/fixtures/jetson_tensorrt_25w_result.json \
  --tegrastats-log tests/fixtures/tegrastats_sample.log \
  --report-output "${JETSON_REPORT_PATH}"

grep -q "InferEdge Runtime Jetson Evidence Summary" "${JETSON_REPORT_PATH}"
grep -q "Lab-compatible import path" "${JETSON_REPORT_PATH}"
grep -q "| sample_count | 2 |" "${JETSON_REPORT_PATH}"
grep -q "capture_depth" "${JETSON_REPORT_PATH}"
grep -q "short_smoke" "${JETSON_REPORT_PATH}"

./build/inferedge-runtime \
  --compare-power-modes \
  --base-result tests/fixtures/jetson_tensorrt_25w_result.json \
  --candidate-result tests/fixtures/jetson_tensorrt_15w_result.json \
  --report-output "${POWER_REPORT_PATH}"

grep -q "InferEdge Runtime Jetson Power Mode Comparison" "${POWER_REPORT_PATH}"
grep -q "25W" "${POWER_REPORT_PATH}"
grep -q "15W" "${POWER_REPORT_PATH}"
grep -q "Run Depth Comparison" "${POWER_REPORT_PATH}"
grep -q "short_smoke" "${POWER_REPORT_PATH}"
grep -q "TensorRT INT8 automatic calibration is outside this report scope" "${POWER_REPORT_PATH}"

echo "[smoke_default] success"
