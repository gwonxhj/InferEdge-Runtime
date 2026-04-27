#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

OUTPUT_PATH="results/smoke_default.json"
PRETTY_PATH="/tmp/inferedge_runtime_smoke_default_pretty.json"

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
]
missing = [key for key in required if key not in data]
assert not missing, missing
assert data["status"] == "skipped", data["status"]
assert data["success"] is False, data["success"]
PY

INFEREDGE_RUNTIME_RESULT_JSON="${OUTPUT_PATH}" python3 tests/test_lab_result_schema.py

echo "[smoke_default] success"
