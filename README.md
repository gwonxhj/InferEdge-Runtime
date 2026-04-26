# InferEdge-Runtime

![CI](https://github.com/gwonxhj/InferEdge-Runtime/actions/workflows/ci.yml/badge.svg)

## Project Overview

InferEdgeRuntime is a C++ Edge AI runtime for on-device inference and benchmarking.

It is the Runtime stage of the InferEdge portfolio pipeline. InferEdgeForge prepares model artifacts, InferEdgeRuntime runs and benchmarks those artifacts on target devices, and InferEdgeLab analyzes the exported result JSON files.

## InferEdge Pipeline Position

1. Forge: Build / Convert / Metadata
2. Runtime: Run / Benchmark / Export Result
3. Lab: Analyze / Compare / Report

## Current Capabilities

- C++17 + CMake build
- CLI option validation
- ONNX Runtime external link configuration
- ONNX model metadata loading
- float32 dummy input generation
- ONNX Runtime CPU inference benchmark
- latency mean/min/max/std/p50/p90/p99
- FPS calculation
- JSON result export
- Lab-compatible top-level fields

## Current Limitations

- ONNX Runtime CPU only
- float32 input only
- TensorRT not implemented yet
- OpenCV/CUDA not linked
- no real image preprocessing yet
- no model artifact auto-discovery from Forge yet
- no automated test suite yet

## Requirements

- CMake 3.16+
- C++17 compiler
- Optional: ONNX Runtime C/C++ package
- Apple Silicon users should use the `osx-arm64` ONNX Runtime package

## Smoke Test Scripts

Use the smoke scripts before opening a PR or after changing runtime behavior.

CI runs the default smoke test on every push to `main` and every pull request. The workflow validates build success, CLI execution, and JSON export without external ONNX Runtime dependencies.

Default smoke test:

```bash
scripts/smoke_default.sh
```

This builds the dependency-free target, runs help/version checks, writes `results/smoke_default.json`, validates the JSON, and confirms the benchmark status is `skipped`.

ONNX Runtime linked smoke test:

```bash
scripts/smoke_ort.sh "$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0" /path/to/model.onnx
```

This requires a local ONNX Runtime package and a local ONNX model file outside the repository. If macOS blocks the downloaded ONNX Runtime `.dylib`, use the `xattr` command in the macOS quarantine note below.

## Quickstart: Default Build

The default build does not require ONNX Runtime. It still writes a JSON result, but the benchmark is marked as skipped.

```bash
cmake -S . -B build
cmake --build build
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/default_skipped.json
```

Expected behavior:

- build succeeds without external runtime dependencies
- backend availability is `false`
- benchmark status is `skipped`
- `results/default_skipped.json` is created

## Quickstart: ONNX Runtime Linked Build On Apple Silicon

Keep the ONNX Runtime package outside this repository. Do not vendor ONNX Runtime headers, libraries, or model files into this repo.

Example package location:

```bash
~/onnxruntime/onnxruntime-osx-arm64-1.25.0
```

Build with ONNX Runtime enabled:

```bash
cmake -S . -B build-ort -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT=$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0
cmake --build build-ort
```

Run a benchmark with a local ONNX model:

```bash
./build-ort/inferedge-runtime --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 3 --runs 10 --output results/ort_cpu.json
```

Expected behavior:

- backend availability is `true`
- model input/output metadata is printed
- warmup iterations run before timed runs
- latency and FPS are printed
- `results/ort_cpu.json` is created

## macOS Quarantine Note

Downloaded ONNX Runtime `.dylib` files can be blocked by macOS quarantine policy. If the linked binary fails to load the ONNX Runtime library, remove the quarantine attribute from the external ONNX Runtime package:

```bash
xattr -dr com.apple.quarantine ~/onnxruntime/onnxruntime-osx-arm64-1.25.0
```

## Usage

```bash
./build/inferedge-runtime --help
./build/inferedge-runtime --version
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 5 --runs 50 --output results/sample.json
```

CLI notes:

- `--batch`, `--height`, and `--width` resolve dynamic dummy input dimensions.
- Static model dimensions take precedence over CLI shape overrides.
- `--warmup` controls untimed warmup iterations.
- `--runs` controls timed iterations used for latency and FPS statistics.
- `--output` writes the benchmark result JSON and creates missing output directories.

## JSON Result Schema

Runtime JSON results include nested structured fields for detailed reporting and top-level compatibility fields for quick comparison.

Main nested fields:

- `schema_version`
- `model`
- `engine`
- `device`
- `run_config`
- `latency_ms`
- `fps`
- `benchmark`
- `timestamp`
- `system`
- `model_metadata`
- `extra`

Top-level compatibility fields:

- `model_name`
- `model_path`
- `engine_name`
- `engine_backend`
- `device_name`
- `batch`
- `height`
- `width`
- `mean_ms`
- `p99_ms`
- `fps_value`
- `success`
- `status`

See [examples/README.md](examples/README.md) for command examples and compact JSON field notes.

## InferEdgeLab Compatibility

Runtime JSON results include both nested structured fields and top-level compatibility fields.

The nested fields are intended for detailed reports and future schema expansion. The top-level compatibility fields are intended for quick comparison in InferEdgeLab and EdgeBench-style loaders without deep nested parsing.

Forge -> Runtime -> Lab flow:

1. Forge builds or exports model artifacts.
2. Runtime runs ONNX Runtime benchmark and writes JSON result.
3. Lab reads JSON results and compares/report performance.

## Repository Layout

```text
.
├── CMakeLists.txt
├── include/
│   └── inferedge_runtime/
│       ├── cli.hpp
│       ├── engine.hpp
│       ├── version.hpp
│       └── engines/
│           └── onnxruntime_engine.hpp
├── src/
│   ├── cli.cpp
│   ├── engine.cpp
│   ├── main.cpp
│   └── engines/
│       └── onnxruntime_engine.cpp
├── scripts/
│   ├── smoke_default.sh
│   └── smoke_ort.sh
├── examples/
│   └── README.md
└── tests/
    └── README.md
```

## Roadmap

- [x] CLI skeleton
- [x] Backend interface and ONNX Runtime stub backend
- [x] ONNX Runtime C++ link configuration
- [x] ONNX model metadata loading
- [x] ONNX Runtime dummy inference
- [x] Benchmark runner
- [x] JSON result export
- [x] Lab-compatible JSON fields
- [x] Scripted smoke tests
- [ ] TensorRT backend on Jetson
- [ ] Forge metadata input integration
- [ ] InferEdgeLab direct import workflow
- [ ] GitHub Actions CI smoke tests
