# InferEdgeRuntime MVP Validation

> Status note: this document records the early ONNX Runtime MVP validation snapshot.
> Current Runtime documentation in `README.md` and `tests/README.md` also covers Jetson TensorRT linked-build smoke evidence, Forge manifest handoff, Lab worker request validation, worker response dry-run export, and manifest source-model identity preservation for compare keys.

## Validation Environment

- Host: macOS (Apple Silicon)
- Compiler: AppleClang
- Build system: CMake
- Runtime: ONNX Runtime (external package)

## Build Validation

- Default build (no ORT): success
- ORT-linked build: success

## CLI Validation

- `--help` / `--version`: success
- Argument validation: success
- Invalid inputs produce expected errors

## Model Handling

- ONNX model metadata extraction: success
- Dynamic shape resolution: success

## Inference Execution

- Dummy input generation: success
- ONNX Runtime inference execution: success

## Benchmark Validation

- warmup runs: success
- timed runs: success
- latency statistics:
  - mean
  - min/max
  - std
  - p50/p90/p99
- FPS calculation: success

## JSON Result Validation

- JSON export: success
- `schema_version` present
- nested + compatibility fields present
- `latest.json` handoff: success
- python `json.tool` validation: success

## Manifest Validation

- manifest path recording: success
- manifest default application: success
- CLI override priority: verified

## CI Validation

- GitHub Actions `build-and-smoke`: success
- artifact upload: success

## Early MVP Limitations

- default macOS build uses ONNX Runtime/stub paths unless optional backends are explicitly linked
- float32 input only
- TensorRT execution requires an explicit Jetson linked build
- image preprocessing requires an OpenCV-enabled build
- broader backend integration tests remain future work

## Conclusion

InferEdgeRuntime v0.1.0 provides a working C++ edge inference runtime with reproducible benchmarking and structured result export suitable for integration with InferEdgeLab.
