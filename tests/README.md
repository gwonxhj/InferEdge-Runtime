# Tests

InferEdgeRuntime has a GitHub Actions default smoke workflow. Use the scripted smoke tests locally first, then fall back to the manual checklist when debugging a specific failure.

## Recommended Smoke Scripts

Default dependency-free build:

```bash
scripts/smoke_default.sh
```

ONNX Runtime linked build with a local package and model:

```bash
scripts/smoke_ort.sh "$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0" /path/to/model.onnx
```

The scripts validate build success, CLI execution, JSON validity, Lab-compatible top-level fields, and benchmark status.

## Manual Smoke Checklist

## Build Smoke Test

```bash
cmake -S . -B build
cmake --build build
```

## CLI Help/Version Test

```bash
./build/inferedge-runtime --help
./build/inferedge-runtime --version
```

## Default Build Skipped Benchmark Test

```bash
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/default_smoke.json
python3 -m json.tool results/default_smoke.json > /tmp/default_smoke_pretty.json
```

Expected:

- benchmark status is `skipped`
- JSON is valid

## ORT Linked Build Test

```bash
cmake -S . -B build-ort -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT=$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0
cmake --build build-ort
./build-ort/inferedge-runtime --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 3 --runs 10 --output results/ort_smoke.json
```

Expected:

- backend availability is `true`
- benchmark status is `success`
- latency and FPS are printed

## Missing Model Error Test

```bash
./build-ort/inferedge-runtime --model models/missing.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/missing.json
```

Expected:

- command exits non-zero
- stderr includes `model file not found`

## Invalid Runs Validation Test

```bash
./build-ort/inferedge-runtime --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 0 --runs 0 --output results/invalid.json
```

Expected:

- command exits non-zero
- stderr includes `minimum: 1`

## Manifest Path Recording Smoke Test

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/manifest_sample_smoke.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/manifest_sample_smoke.json").read_text())
assert data["manifest_path"] == "examples/manifest.sample.json"
assert data["run_config"]["manifest_path"] == "examples/manifest.sample.json"
assert data["extra"]["manifest_recorded"] is True
print("manifest sample smoke ok")
PY
```

## Manifest Parser Smoke Test

Manifest-only default apply:

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --output results/manifest_only.json
```

Expected:

- `model_path` is loaded from `artifact.model_path`
- `engine`, `device`, `batch`, `height`, and `width` are loaded from `runtime`
- `manifest_applied` is `true`

CLI override priority:

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --model models/override.onnx --batch 2 --height 320 --width 320 --output results/manifest_override.json
```

Expected:

- CLI `--model` wins over `artifact.model_path`
- CLI `--batch`, `--height`, and `--width` win over manifest values

## TensorRT Stub Smoke Test

```bash
./build/inferedge-runtime --model models/sample.engine --engine tensorrt --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/tensorrt_stub_smoke.json
python3 -m json.tool results/tensorrt_stub_smoke.json > /tmp/tensorrt_stub_smoke_pretty.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/tensorrt_stub_smoke.json").read_text())
assert data["engine_name"] == "tensorrt"
assert data["engine_backend"] == "tensorrt"
assert data["status"] == "skipped"
assert data["success"] is False
print("tensorrt stub smoke ok")
PY
```

## JSON Validity Test

```bash
python3 -m json.tool results/default_smoke.json > /tmp/default_smoke_pretty.json
python3 -m json.tool results/ort_smoke.json > /tmp/ort_smoke_pretty.json
```
