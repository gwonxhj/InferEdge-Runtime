#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

OUTPUT_PATH="results/smoke_forge_handoff.json"
FALLBACK_OUTPUT_PATH="results/smoke_compare_key_fallback.json"
MANIFEST_FIXTURE="tests/fixtures/forge_handoff_manifest.json"
METADATA_FIXTURE="tests/fixtures/forge_handoff_metadata.json"
BROKEN_FIXTURE="tests/fixtures/forge_handoff_missing_engine.json"

cmake -S . -B build
cmake --build build

./build/inferedge-runtime \
  --forge-manifest "${MANIFEST_FIXTURE}" \
  --validate-forge-handoff

./build/inferedge-runtime \
  --forge-metadata "${METADATA_FIXTURE}" \
  --validate-forge-handoff

if ./build/inferedge-runtime --forge-manifest "${BROKEN_FIXTURE}" --validate-forge-handoff 2>/tmp/inferedge_forge_handoff_error.txt; then
  echo "expected broken Forge handoff fixture to fail" >&2
  exit 1
fi
grep -q "missing runtime engine" /tmp/inferedge_forge_handoff_error.txt

./build/inferedge-runtime \
  --forge-manifest "${MANIFEST_FIXTURE}" \
  --model "builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine" \
  --warmup 1 \
  --runs 1 \
  --output "${OUTPUT_PATH}"

python3 -m json.tool "${OUTPUT_PATH}" >/tmp/inferedge_runtime_forge_handoff_pretty.json
INFEREDGE_RUNTIME_RESULT_JSON="${OUTPUT_PATH}" python3 tests/test_lab_result_schema.py
python3 tests/test_forge_handoff.py

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/smoke_forge_handoff.json").read_text(encoding="utf-8"))
extra = data["extra"]
assert data["model_path"] == "builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine"
assert data["engine_backend"] == "tensorrt"
assert data["device_name"] == "jetson"
assert data["precision"] == "fp16"
assert data["batch"] == 1
assert data["height"] == 640
assert data["width"] == 640
assert data["status"] == "skipped"
assert data["compare_key"] == "yolov8n__b1__h640w640__fp16"
assert extra["manifest_preset_name"] == "tensorrt/jetson_fp16"
assert extra["manifest_build_id"] == "yolov8n-tensorrt-jetson_fp16-20260427T000000Z"
assert extra["runtime_artifact_sha256"] == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
assert extra["source_model_path"] == "models/yolov8n.onnx"
assert extra["source_model_sha256"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
assert extra["runtime_artifact_path"] == data["model_path"]
assert extra["compare_model_name"] == "yolov8n"
assert extra["compare_model_source"] == "manifest_source_model"
PY

./build/inferedge-runtime \
  --model "builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine" \
  --engine tensorrt \
  --device jetson \
  --height 640 \
  --width 640 \
  --warmup 1 \
  --runs 1 \
  --output "${FALLBACK_OUTPUT_PATH}"

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/smoke_compare_key_fallback.json").read_text(encoding="utf-8"))
extra = data["extra"]
assert data["compare_key"] == "model__b1__h640w640__fp32"
assert extra["compare_model_name"] == "model"
assert extra["compare_model_source"] == "model_path"
PY

echo "[smoke_forge_handoff] success"
