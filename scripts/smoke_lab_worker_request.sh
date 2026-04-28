#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

cmake -S . -B build
cmake --build build

python3 tests/test_lab_worker_adapter_contract.py

./build/inferedge-runtime \
  --lab-worker-request tests/fixtures/lab_worker_request.json \
  --validate-lab-worker-request \
  > /tmp/inferedge_runtime_lab_worker_request_valid.txt

grep -q "Lab worker request validation" /tmp/inferedge_runtime_lab_worker_request_valid.txt
grep -q "status: ok" /tmp/inferedge_runtime_lab_worker_request_valid.txt
grep -q "job_id: job_runtime_worker_smoke" /tmp/inferedge_runtime_lab_worker_request_valid.txt
grep -q "engine: tensorrt" /tmp/inferedge_runtime_lab_worker_request_valid.txt

./build/inferedge-runtime \
  --lab-worker-request tests/fixtures/forge_summary_worker_request.json \
  --validate-lab-worker-request \
  > /tmp/inferedge_runtime_forge_summary_worker_request_valid.txt

grep -q "Lab worker request validation" /tmp/inferedge_runtime_forge_summary_worker_request_valid.txt
grep -q "status: ok" /tmp/inferedge_runtime_forge_summary_worker_request_valid.txt
grep -q "job_id: job_forge_summary_smoke" /tmp/inferedge_runtime_forge_summary_worker_request_valid.txt
grep -q "source_model_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" /tmp/inferedge_runtime_forge_summary_worker_request_valid.txt
grep -q "artifact_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" /tmp/inferedge_runtime_forge_summary_worker_request_valid.txt
grep -q "preset_name: tensorrt/jetson_fp16" /tmp/inferedge_runtime_forge_summary_worker_request_valid.txt

if ./build/inferedge-runtime \
  --lab-worker-request tests/fixtures/lab_worker_request_missing_input.json \
  --validate-lab-worker-request \
  > /tmp/inferedge_runtime_lab_worker_request_invalid.txt \
  2> /tmp/inferedge_runtime_lab_worker_request_invalid.err; then
  echo "expected invalid Lab worker request validation to fail" >&2
  exit 1
fi

grep -q "model_path or artifact_path" /tmp/inferedge_runtime_lab_worker_request_invalid.err

if ./build/inferedge-runtime \
  --lab-worker-request tests/fixtures/forge_summary_worker_request_missing_input.json \
  --validate-lab-worker-request \
  > /tmp/inferedge_runtime_forge_summary_worker_request_invalid.txt \
  2> /tmp/inferedge_runtime_forge_summary_worker_request_invalid.err; then
  echo "expected invalid Forge summary worker request validation to fail" >&2
  exit 1
fi

grep -q "model_path or artifact_path" /tmp/inferedge_runtime_forge_summary_worker_request_invalid.err

echo "[smoke_lab_worker_request] success"
