# TensorRT Backend Implementation Plan

## Purpose

The TensorRT backend is the next Jetson-oriented extension stage for InferEdgeRuntime. Its purpose is to load serialized `.engine` files on Jetson Orin Nano, execute inference through TensorRT, measure benchmark latency, and export the same Runtime JSON schema already used by the ONNX Runtime backend.

On Mac development environments, InferEdgeRuntime should continue to avoid TensorRT/CUDA linkage and keep the TensorRT backend as a stub. This keeps the default build dependency-free while allowing Jetson-specific implementation work to be introduced explicitly later.

## Current State

- The ONNX Runtime backend supports CPU model loading, metadata extraction, dummy float32 inference, benchmark execution, JSON export, auto output naming, and `results/latest.json`.
- The TensorRT backend supports Jetson linked-build `.engine` deserialization and input/output metadata extraction.
- `--engine tensorrt` and `--engine trt` are accepted by the CLI.
- Mac/default TensorRT stub metadata reports `available=false`.
- Jetson linked TensorRT metadata reports `available=true`.
- TensorRT benchmark execution produces real latency/FPS JSON results on Jetson linked builds.
- Jetson TensorRT Runtime JSON can be imported and compared in InferEdgeLab and replayed through Local Studio demo evidence.

Progress note:

- TensorRT linked builds can now deserialize `.engine` files, extract metadata, allocate float32 dummy buffers, and run one-shot inference.
- TensorRT linked builds can now produce real benchmark JSON from Forge-generated `.engine` files on Jetson.
- InferEdgeLab now has direct import/compare and Local Studio demo evidence for TensorRT vs ONNX Runtime result inspection.

## Target Runtime Flow

The target TensorRT flow should reuse the existing runtime orchestration and benchmark/result schema wherever possible.

1. Runtime receives a `.engine` path through `--model` or manifest `artifact.model_path`.
2. `TensorRTEngine` loads the serialized engine file.
3. Create the TensorRT runtime.
4. Deserialize `ICudaEngine`.
5. Create `IExecutionContext`.
6. Resolve input/output binding metadata.
7. Allocate host/device buffers.
8. Generate dummy input buffer.
9. Copy input host buffer to device.
10. Enqueue inference.
11. Copy output device buffer to host if needed.
12. Reuse the existing warmup/timed benchmark loop structure.
13. Export the same JSON schema as the ONNX Runtime backend.

## Class/Module Design

`TensorRTEngine` should keep TensorRT and CUDA types hidden inside the `.cpp` implementation or a pImpl holder. The public header must not expose TensorRT or CUDA headers, because the default Mac build should remain dependency-free.

The generic engine interface should remain stable:

- `EngineMetadata` describes backend availability and status.
- `ModelMetadata` represents engine input/output bindings.
- `BenchmarkResult` is reused for warmup/timed benchmark statistics.
- `IInferenceEngine` remains the shared abstraction for ONNX Runtime, TensorRT, and future backends.

Recommended structure:

- `include/inferedge_runtime/engines/tensorrt_engine.hpp`
- `src/engines/tensorrt_engine.cpp`

Optional future helpers:

- `src/engines/tensorrt_buffers.cpp`
- `include/inferedge_runtime/engines/tensorrt_buffers.hpp`

## Build Configuration Plan

TensorRT support should be explicitly enabled in a Jetson-oriented build. The default build must remain dependency-free, and the Mac default build must keep the TensorRT stub.

Implemented CMake options:

- `INFEREDGE_ENABLE_TENSORRT`
- `INFEREDGE_TENSORRT_ROOT`
- `INFEREDGE_CUDA_ROOT`

Expected CMake policy:

- If `INFEREDGE_ENABLE_TENSORRT=ON` and TensorRT/CUDA headers/libs are missing, configure fails with a clear error.
- Jetson default paths are supported first:
  - `/usr/include/aarch64-linux-gnu`
  - `/usr/lib/aarch64-linux-gnu`
  - `/usr/local/cuda/include`
  - `/usr/local/cuda/lib64`
- TensorRT and CUDA files must not be vendored into this repository.
- Jetson builds enable TensorRT explicitly.
- Mac builds keep the TensorRT stub unless a future supported TensorRT environment is intentionally configured.

## TensorRT Metadata Mapping

- `engine.name = tensorrt`
- `engine.backend = tensorrt`
- `device.name = jetson` or `cuda`
- Precision should come from the manifest or a future CLI field.
- `model_metadata.inputs` and `model_metadata.outputs` should be derived from TensorRT bindings/tensors.
- Dynamic shape handling should follow the existing `batch`/`height`/`width` policy where applicable.

## Benchmark Policy

- Warmup runs are untimed.
- Timed runs collect latency samples.
- InferEdgeRuntime TensorRT benchmark latency is end-to-end latency, not pure GPU compute latency.
- Current measurements include host/device transfer and synchronization overhead.
- Do not directly compare InferEdgeRuntime latency with `trtexec` GPU latency.
- This policy is intentional so ONNX Runtime and TensorRT can be compared from the same Runtime perspective.
- TensorRT should populate the same `BenchmarkResult` fields as ONNX Runtime:
  - `mean_ms`
  - `min_ms`
  - `max_ms`
  - `std_ms`
  - `p50_ms`
  - `p90_ms`
  - `p95_ms`
  - `p99_ms`
  - `fps`
- JSON schema compatibility must remain stable.
- `results/latest.json` must still be written after each successful result export.
- `--output auto` naming must work with `.engine` files.

## Error Handling Policy

TensorRT implementation errors should be explicit and actionable. Example messages:

- `TensorRT backend is not enabled in this build`
- `TensorRT engine file not found: <path>`
- `Failed to create TensorRT runtime`
- `Failed to deserialize TensorRT engine`
- `Failed to create TensorRT execution context`
- `Unsupported TensorRT input data type`
- `Failed to allocate CUDA buffer`
- `Failed to enqueue TensorRT inference`

## Jetson Validation Checklist

- [x] Verify Jetson Orin Nano environment
- [x] Verify TensorRT version
- [x] Verify CUDA runtime availability
- [x] Build with TensorRT enabled
- [x] Run TensorRT stub regression
- [x] Run real `.engine` metadata load
- [x] Run one-shot inference
- [x] Run warmup/runs benchmark
- [x] Validate JSON with `python3 -m json.tool`
- [x] Confirm `results/latest.json`
- [x] Compare Runtime JSON in InferEdgeLab

## Non-goals

- TensorRT benchmark implementation is now available in the linked Jetson build; this plan keeps the earlier one-shot context for design history.
- No OpenCV preprocessing
- No real image/video pipeline
- No RKNN/Hailo backend yet
- No Forge repo changes yet

## Implementation Milestones

- [x] TensorRT backend stub
- [x] Add TensorRT CMake option and link validation
- [x] Add TensorRT pImpl runtime holder
- [x] Load serialized `.engine` file
- [x] Deserialize `ICudaEngine`
- [x] Create execution context
- [x] Extract binding/tensor metadata
- [x] Allocate buffers
- [x] Run one-shot inference
- [x] Reuse benchmark runner
- [x] Export TensorRT benchmark JSON
- [x] Validate on Jetson Orin Nano
- [x] Compare TensorRT result in InferEdgeLab
- [x] Local Studio demo evidence integration through InferEdgeLab
- [x] Jetson power-mode evidence fields and tegrastats summary
- [ ] Output post-processing
- [x] Lab comparison workflow
