# InferEdge-Runtime

InferEdgeRuntime is a C++ Edge AI runtime project for on-device inference execution and benchmarking.

This repository is part of the InferEdge portfolio pipeline:

1. InferEdgeForge prepares and exports model artifacts.
2. InferEdgeRuntime executes models on edge targets and records runtime behavior.
3. InferEdgeLab analyzes benchmark results, compares backends, and presents portfolio-ready reports.

## Current Stage

The current stage is **C++ Runtime CLI Skeleton** with explicit CLI validation policy.

This version only provides a minimal C++17 and CMake-based command-line interface. It parses runtime options, validates supported engine/device values and numeric ranges, then prints the selected benchmark configuration.

Actual inference execution is not implemented yet. ONNX Runtime, TensorRT, OpenCV, and other external runtime dependencies are intentionally not linked at this stage.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Usage

```bash
./build/inferedge-runtime --help
./build/inferedge-runtime --version
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --warmup 5 --runs 50 --output results/sample.json
```

The model path is accepted as a configuration value only. The CLI does not check model file existence and does not run inference yet.

## Roadmap

1. CLI skeleton
2. ONNX Runtime C++ backend
3. Benchmark runner
4. JSON result export
5. TensorRT backend on Jetson
6. Forge/Lab integration
