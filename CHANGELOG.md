# Changelog

## v0.1.0 - Initial MVP Release

### Added

- C++17 CMake-based runtime CLI
- ONNX Runtime CPU inference backend
- Model metadata extraction (inputs/outputs)
- Dummy input generation for float32 models
- Warmup + timed benchmark runner
- Latency statistics: mean, min, max, std, p50, p90, p99
- FPS calculation
- JSON result export
- Lab-compatible top-level JSON fields
- Automatic result naming (`--output auto`)
- `results/latest.json` handoff
- Manifest path recording (`--manifest`)
- Limited manifest default application
- TensorRT backend stub
- Smoke test scripts
- GitHub Actions CI (default build smoke test)

### Notes

- ONNX Runtime must be provided externally
- TensorRT backend is currently a stub
- JSON schema is designed for InferEdgeLab integration
