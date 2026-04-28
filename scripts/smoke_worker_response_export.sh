#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
COMPLETED_JSON="${TMPDIR:-/tmp}/inferedge_runtime_worker_completed_response.json"
FAILED_JSON="${TMPDIR:-/tmp}/inferedge_runtime_worker_failed_response.json"

cmake -S . -B "$BUILD_DIR" >/dev/null
cmake --build "$BUILD_DIR" >/dev/null

"$BUILD_DIR/inferedge-runtime" \
  --lab-worker-request tests/fixtures/lab_worker_request.json \
  --export-worker-response "$COMPLETED_JSON" \
  --worker-response-status completed >/dev/null

"$BUILD_DIR/inferedge-runtime" \
  --lab-worker-request tests/fixtures/lab_worker_request.json \
  --export-worker-response "$FAILED_JSON" \
  --worker-response-status failed \
  --worker-error-message "dry-run failure" >/dev/null

if "$BUILD_DIR/inferedge-runtime" \
  --export-worker-response "$COMPLETED_JSON" \
  --worker-response-status completed >/tmp/inferedge_runtime_missing_worker_request.out 2>&1; then
  echo "expected missing --lab-worker-request to fail" >&2
  exit 1
fi

if "$BUILD_DIR/inferedge-runtime" \
  --lab-worker-request tests/fixtures/lab_worker_request.json \
  --export-worker-response "$COMPLETED_JSON" \
  --worker-response-status unknown >/tmp/inferedge_runtime_invalid_worker_status.out 2>&1; then
  echo "expected invalid --worker-response-status to fail" >&2
  exit 1
fi

python3 - "$COMPLETED_JSON" "$FAILED_JSON" <<'PY'
import json
import sys
from pathlib import Path

completed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
failed = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

assert completed["job_id"] == "job_runtime_worker_smoke"
assert completed["status"] == "completed"
assert isinstance(completed["runtime_result"], dict)
assert completed["runtime_result"]["model_path"]
assert completed["runtime_result"]["engine_backend"] == "tensorrt"
assert completed["runtime_result"]["precision"] == "fp16"
assert completed["runtime_result"]["run_config"]["dry_run"] is True
assert completed["runtime_result"]["extra"]["worker_response_mode"] == "dry_run"
assert "completed_at" in completed

assert failed["job_id"] == "job_runtime_worker_smoke"
assert failed["status"] == "failed"
assert failed["error"]["message"] == "dry-run failure"
assert failed["error"]["stage"] == "runtime"
assert "failed_at" in failed
assert "runtime_result" not in failed
PY

echo "[smoke_worker_response_export] success"
