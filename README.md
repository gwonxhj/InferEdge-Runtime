# InferEdge-Runtime

InferEdgeRuntime is a C++ Edge AI runtime project for on-device inference execution and benchmarking.

This repository is part of the InferEdge portfolio pipeline:

1. InferEdgeForge prepares and exports model artifacts.
2. InferEdgeRuntime executes models on edge targets and records runtime behavior.
3. InferEdgeLab analyzes benchmark results, compares backends, and presents portfolio-ready reports.

## Current Stage

The current stage is **ONNX Runtime model metadata loading**.

This version provides a minimal C++17 and CMake-based command-line interface. It parses runtime options, validates supported engine/device values and numeric ranges, creates an inference engine, then prints the selected benchmark configuration, backend metadata, and model input/output metadata when ONNX Runtime is linked.

Actual inference execution, benchmarking, and JSON export are not implemented yet. TensorRT, OpenCV, CUDA, and other external runtime dependencies are intentionally not linked at this stage.

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

At the current stage, a linked ONNX Runtime backend creates an `Ort::Env` and `Ort::Session`, loads the supplied ONNX model file, and prints input/output names, element types, and shapes. It does not create dummy tensors, run inference, execute warmup/runs benchmarking, or export JSON yet. The next step will extend this into input allocation and inference execution.

## Usage

```bash
./build/inferedge-runtime --help
./build/inferedge-runtime --version
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --warmup 5 --runs 50 --output results/sample.json
```

The model path is accepted as a configuration value only. The CLI does not check model file existence and does not run inference yet.

In the default non-ORT build, the CLI does not require the model file to exist and prints empty model metadata with `available: false`.

In an ONNX Runtime linked build, the model file must exist. Missing files fail with an error such as:

```text
error: model file not found: models/missing.onnx
```

The CLI prints backend metadata for the selected engine. The ONNX Runtime backend reports `available: false` in the default build and can report `available: true` only when an external ONNX Runtime C++ package is explicitly linked. In both cases, inference execution is still disabled.

## Roadmap

1. CLI skeleton
2. Backend interface and ONNX Runtime stub backend
3. ONNX Runtime C++ link configuration
4. ONNX Runtime model metadata loading
5. ONNX Runtime input allocation and inference execution
6. Benchmark runner
7. JSON result export
8. TensorRT backend on Jetson
9. Forge/Lab integration
