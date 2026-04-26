# InferEdge-Runtime

InferEdgeRuntime is a C++ Edge AI runtime project for on-device inference execution and benchmarking.

This repository is part of the InferEdge portfolio pipeline:

1. InferEdgeForge prepares and exports model artifacts.
2. InferEdgeRuntime executes models on edge targets and records runtime behavior.
3. InferEdgeLab analyzes benchmark results, compares backends, and presents portfolio-ready reports.

## Current Stage

The current stage is **ONNX Runtime benchmark runner**.

This version provides a minimal C++17 and CMake-based command-line interface. It parses runtime options, validates supported engine/device values and numeric ranges, creates an inference engine, prints backend and model metadata, then runs warmup iterations followed by timed ONNX Runtime inference runs when ONNX Runtime is linked.

The CLI reports latency mean, min, max, population standard deviation, p50, p90, p99, and FPS. JSON export is not implemented yet. TensorRT, OpenCV, CUDA, and other external runtime dependencies are intentionally not linked at this stage.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## ONNX Runtime C++ Backend Link Preparation

The default build does not require ONNX Runtime:

```bash
cmake -S . -B build
cmake --build build
```

To validate the ONNX Runtime C++ link path, download the platform-specific ONNX Runtime package from GitHub Releases and keep it outside this repository. Apple Silicon Macs such as M1, M2, and M3 should use the `osx-arm64` package.

```bash
cmake -S . -B build-ort -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT=/path/to/onnxruntime-osx-arm64
cmake --build build-ort
```

When `INFEREDGE_ENABLE_ORT=ON`, `INFEREDGE_ORT_ROOT` must point to an external ONNX Runtime C/C++ package root containing `include/onnxruntime_cxx_api.h` and the `lib/onnxruntime` library. The package must not be vendored into this repository.

This project intentionally uses an external dependency path to reflect real-world deployment environments where runtime libraries are managed outside of the application repository.

At the current stage, a linked ONNX Runtime backend creates an `Ort::Env` and persistent `Ort::Session`, loads the supplied ONNX model file, prints input/output names, element types, and shapes, creates float32 dummy input tensors, executes warmup runs, and measures timed inference latency. It does not export JSON yet.

## Usage

```bash
./build/inferedge-runtime --help
./build/inferedge-runtime --version
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 5 --runs 50 --output results/sample.json
```

The `--batch`, `--height`, and `--width` options are used to resolve dynamic dummy input dimensions. Dynamic or zero dimensions are resolved as batch for dimension 0, `3` for dimension 1, height for dimension 2, width for dimension 3, and `1` for later dimensions. Static model dimensions take precedence over CLI overrides.

The `--warmup` option controls untimed warmup iterations. The `--runs` option controls timed inference iterations used to calculate latency and FPS statistics.

In the default non-ORT build, the CLI does not require the model file to exist, prints empty model metadata with `available: false`, and skips inference.

In an ONNX Runtime linked build, the model file must exist. Missing files fail with an error such as:

```text
error: model file not found: models/missing.onnx
```

The ONNX Runtime linked dummy execution currently supports float32 inputs only. Models with non-float32 inputs fail before inference with a clear error.

## Roadmap

1. CLI skeleton
2. Backend interface and ONNX Runtime stub backend
3. ONNX Runtime C++ link configuration
4. ONNX Runtime model metadata loading
5. ONNX Runtime dummy input allocation and one-shot inference execution
6. Benchmark runner
7. JSON result export
8. TensorRT backend on Jetson
9. Forge/Lab integration
