# InferEdge-Runtime

InferEdgeRuntime is a C++ Edge AI runtime project for on-device inference execution and benchmarking.

This repository is part of the InferEdge portfolio pipeline:

1. InferEdgeForge prepares and exports model artifacts.
2. InferEdgeRuntime executes models on edge targets and records runtime behavior.
3. InferEdgeLab analyzes benchmark results, compares backends, and presents portfolio-ready reports.

## Current Stage

The current stage is **C++ Runtime CLI Skeleton** with explicit CLI validation policy, a backend interface, an ONNX Runtime stub backend, and ONNX Runtime C++ link preparation.

This version only provides a minimal C++17 and CMake-based command-line interface. It parses runtime options, validates supported engine/device values and numeric ranges, creates a stub inference engine, then prints the selected benchmark configuration and backend metadata.

Actual inference execution is not implemented yet. TensorRT, OpenCV, CUDA, and other external runtime dependencies are intentionally not linked at this stage.

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

At the current stage, a linked ONNX Runtime backend only reports backend availability. It does not load a real model, create tensors, inspect input/output metadata, create a session, or run inference yet. The next step will extend this into model loading, session setup, and input/output metadata discovery.

## Usage

```bash
./build/inferedge-runtime --help
./build/inferedge-runtime --version
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --warmup 5 --runs 50 --output results/sample.json
```

The model path is accepted as a configuration value only. The CLI does not check model file existence and does not run inference yet.

The CLI prints backend metadata for the selected engine. The ONNX Runtime backend reports `available: false` in the default build and can report `available: true` only when an external ONNX Runtime C++ package is explicitly linked. In both cases, inference execution is still disabled.

## Roadmap

1. CLI skeleton
2. Backend interface and ONNX Runtime stub backend
3. ONNX Runtime C++ backend
4. Benchmark runner
5. JSON result export
6. TensorRT backend on Jetson
7. Forge/Lab integration
