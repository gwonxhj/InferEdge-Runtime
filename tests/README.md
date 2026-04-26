# Tests

InferEdgeRuntime does not have an automated test suite yet. Use this manual smoke test checklist before opening a PR.

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

## JSON Validity Test

```bash
python3 -m json.tool results/default_smoke.json > /tmp/default_smoke_pretty.json
python3 -m json.tool results/ort_smoke.json > /tmp/ort_smoke_pretty.json
```
