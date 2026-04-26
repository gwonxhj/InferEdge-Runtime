#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/smoke_ort.sh <ORT_ROOT> <MODEL_PATH>

Example:
  scripts/smoke_ort.sh "$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0" /path/to/model.onnx

Note:
  If macOS blocks the ONNX Runtime dylib because of quarantine metadata, see the README and run:
  xattr -dr com.apple.quarantine <ORT_ROOT>
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

ORT_ROOT="$1"
MODEL_PATH="$2"

if [[ -z "${ORT_ROOT}" || ! -d "${ORT_ROOT}" ]]; then
  echo "error: ORT_ROOT must be an existing ONNX Runtime package directory" >&2
  usage >&2
  exit 2
fi

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "error: MODEL_PATH must be an existing ONNX model file" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

OUTPUT_PATH="results/smoke_ort.json"
PRETTY_PATH="/tmp/inferedge_runtime_smoke_ort_pretty.json"

echo "[smoke_ort] using ORT root: ${ORT_ROOT}"
echo "[smoke_ort] using model: ${MODEL_PATH}"
echo "[smoke_ort] if dylib loading fails on macOS, remove quarantine metadata as documented in README.md"

cmake -S . -B build-ort -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT="${ORT_ROOT}"
cmake --build build-ort

./build-ort/inferedge-runtime \
  --model "${MODEL_PATH}" \
  --engine onnxruntime \
  --device cpu \
  --batch 1 \
  --height 224 \
  --width 224 \
  --warmup 3 \
  --runs 10 \
  --output "${OUTPUT_PATH}"

python3 -m json.tool "${OUTPUT_PATH}" > "${PRETTY_PATH}"

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/smoke_ort.json").read_text())
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
    "p99_ms",
    "fps_value",
    "success",
    "status",
]
missing = [key for key in required if key not in data]
assert not missing, missing
assert data["status"] == "success", data["status"]
assert data["success"] is True, data["success"]
assert data["mean_ms"] > 0, data["mean_ms"]
assert data["p99_ms"] > 0, data["p99_ms"]
assert data["fps_value"] > 0, data["fps_value"]
assert len(data["latency_ms"]["samples"]) == 10, data["latency_ms"]["samples"]
PY

echo "[smoke_ort] success"
