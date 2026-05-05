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
PY

INFEREDGE_RUNTIME_RESULT_JSON="${OUTPUT_PATH}" python3 tests/test_lab_result_schema.py

./build/inferedge-runtime \
  --report-jetson-evidence \
  --result-json tests/fixtures/jetson_tensorrt_25w_result.json \
  --tegrastats-log tests/fixtures/tegrastats_sample.log \
  --report-output "${JETSON_REPORT_PATH}"

grep -q "InferEdge Runtime Jetson Evidence Summary" "${JETSON_REPORT_PATH}"
grep -q "Lab-compatible import path" "${JETSON_REPORT_PATH}"
grep -q "| sample_count | 2 |" "${JETSON_REPORT_PATH}"

./build/inferedge-runtime \
  --compare-power-modes \
  --base-result tests/fixtures/jetson_tensorrt_25w_result.json \
  --candidate-result tests/fixtures/jetson_tensorrt_15w_result.json \
  --report-output "${POWER_REPORT_PATH}"

grep -q "InferEdge Runtime Jetson Power Mode Comparison" "${POWER_REPORT_PATH}"
grep -q "25W" "${POWER_REPORT_PATH}"
grep -q "15W" "${POWER_REPORT_PATH}"
grep -q "TensorRT INT8 automatic calibration is outside this report scope" "${POWER_REPORT_PATH}"

echo "[smoke_default] success"
