# InferEdge-Runtime

InferEdgeRuntime is a C++ Edge AI runtime project for on-device inference execution and benchmarking.

This repository is part of the InferEdge portfolio pipeline:

1. InferEdgeForge prepares and exports model artifacts.
2. InferEdgeRuntime executes models on edge targets and records runtime behavior.
3. InferEdgeLab analyzes benchmark results, compares backends, and presents portfolio-ready reports.

## Current Stage

The current stage is **C++ Runtime CLI Skeleton** with explicit CLI validation policy, a backend interface, and an ONNX Runtime stub backend.

This version only provides a minimal C++17 and CMake-based command-line interface. It parses runtime options, validates supported engine/device values and numeric ranges, creates a stub inference engine, then prints the selected benchmark configuration and backend metadata.

Actual inference execution is not implemented yet. ONNX Runtime, TensorRT, OpenCV, CUDA, and other external runtime dependencies are intentionally not linked at this stage.

## Build

```bash
cmake -S . -B build
cmake --build build
```

The `INFEREDGE_ENABLE_ORT` option is available as a preparation switch for the future ONNX Runtime backend:

```bash
cmake -S . -B build-ort -DINFEREDGE_ENABLE_ORT=ON
cmake --build build-ort
```

At the current stage, this option only enables the `INFEREDGE_ENABLE_ORT=1` compile definition. It does not find, include, or link the real ONNX Runtime C++ library yet.

## Usage

```bash
./build/inferedge-runtime --help
./build/inferedge-runtime --version
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --warmup 5 --runs 50 --output results/sample.json
```

The model path is accepted as a configuration value only. The CLI does not check model file existence and does not run inference yet.

The CLI prints backend metadata for the selected engine. The ONNX Runtime backend is currently a stub and reports `available: false` until the real C++ backend is integrated.

## Roadmap

1. CLI skeleton
2. Backend interface and ONNX Runtime stub backend
3. ONNX Runtime C++ backend
4. Benchmark runner
5. JSON result export
6. TensorRT backend on Jetson
7. Forge/Lab integration
