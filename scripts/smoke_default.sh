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
assert events["benchmark_completed"]["latency_budget_ms"] == 33
assert [event["event_index"] for event in data["runtime_events"]] == list(range(len(data["runtime_events"])))
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
