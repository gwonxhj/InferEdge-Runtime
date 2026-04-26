# InferEdge-Runtime

![CI](https://github.com/gwonxhj/InferEdge-Runtime/actions/workflows/ci.yml/badge.svg)

## Project Overview

InferEdgeRuntime is a C++ Edge AI runtime for on-device inference and benchmarking.

It is the Runtime stage of the InferEdge portfolio pipeline. InferEdgeForge prepares model artifacts, InferEdgeRuntime runs and benchmarks those artifacts on target devices, and InferEdgeLab analyzes the exported result JSON files.

## Release Status

InferEdgeRuntime v0.1.0 is a validated MVP release.

- ONNX Runtime CPU backend: fully functional
- Benchmark + JSON export: stable
- Forge/Lab pipeline: partially integrated (manifest + JSON handoff)
- TensorRT backend: benchmark execution on Jetson

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
- automatic result naming and `results/latest.json` handoff
- limited manifest default apply for Forge handoff preparation
- TensorRT backend stub for future Jetson integration
- TensorRT engine deserialization and metadata extraction on Jetson linked builds
- TensorRT one-shot dummy inference on Jetson linked builds
- TensorRT benchmark runner on Jetson linked builds
- documented benchmark measurement policy

## Current Limitations

- ONNX Runtime CPU only
- float32 input only
- no real image preprocessing yet
- no TensorRT output post-processing yet
- float32 TensorRT buffers only at current stage
- no multi-input advanced dynamic shape support yet
- OpenCV/CUDA not linked
- manifest parsing is limited to the sample Forge handoff schema
- no full general-purpose JSON parser yet
- no full unit test suite yet (CI smoke test only)
- GitHub Actions currently runs default smoke test only
- ORT linked smoke test remains local/manual because it requires external ONNX Runtime and model files

TensorRT backend execution is implemented for Jetson-oriented linked builds. The current Mac/default build keeps TensorRT as a stub and does not link TensorRT or CUDA. See [docs/tensorrt_backend_plan.md](docs/tensorrt_backend_plan.md) for the Jetson Orin Nano implementation plan.

## Requirements

- CMake 3.16+
- C++17 compiler
- Optional: ONNX Runtime C/C++ package
- Apple Silicon users should use the `osx-arm64` ONNX Runtime package
- Optional for Jetson TensorRT link validation:
  - Jetson Orin Nano
  - TensorRT 10.x
  - CUDA runtime
  - `NvInfer.h`
  - `libnvinfer.so`
  - `libcudart.so`

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

Record a Forge/build manifest path:

```bash
./build-ort/inferedge-runtime --manifest /path/to/manifest.json --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 3 --runs 10 --output auto
```

Auto-named output:

```bash
./build-ort/inferedge-runtime --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 3 --runs 10 --output auto
```

Expected behavior:

- backend availability is `true`
- model input/output metadata is printed
- warmup iterations run before timed runs
- latency and FPS are printed
- `results/ort_cpu.json` is created
- `--output auto` writes a structured filename under `results/`
- every run also writes `results/latest.json`

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
./build-ort/inferedge-runtime --manifest /path/to/manifest.json --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 3 --runs 10 --output auto
```

CLI notes:

- `--manifest` loads limited defaults from the Forge/build manifest schema.
- CLI-provided values always take priority over manifest defaults.
- `--batch`, `--height`, and `--width` resolve dynamic dummy input dimensions.
- Static model dimensions take precedence over CLI shape overrides.
- `--warmup` controls untimed warmup iterations.
- `--runs` controls timed iterations used for latency and FPS statistics.
- `--run-once` runs one inference without benchmark timing.
- `--output` writes the benchmark result JSON and creates missing output directories.

TensorRT stub example:

```bash
./build/inferedge-runtime --model models/sample.engine --engine tensorrt --device jetson --batch 1 --height 640 --width 640 --warmup 1 --runs 1 --output results/tensorrt_stub.json
```

This command does not execute TensorRT. In the default build, the TensorRT stub reports `available=false` and creates a skipped benchmark JSON result.

Jetson TensorRT one-shot check build:

```bash
cmake -S . -B build-trt -DINFEREDGE_ENABLE_TENSORRT=ON
cmake --build build-trt
./build-trt/inferedge-runtime --model /home/risenano01/InferEdgeForge/builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine --engine tensorrt --device jetson --batch 1 --height 640 --width 640 --run-once --output results/tensorrt_run_once.json
```

When TensorRT and CUDA headers/libraries are found, the TensorRT backend reports `available=true`, deserializes the `.engine` file, records input/output metadata, allocates float32 dummy host/device buffers, and executes one inference through TensorRT. Expected metadata for the current Forge YOLOv8n TensorRT engine includes input `images` and output `output0`.

Jetson TensorRT benchmark:

```bash
./build-trt/inferedge-runtime --model /home/risenano01/InferEdgeForge/builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine --engine tensorrt --device jetson --batch 1 --height 640 --width 640 --warmup 10 --runs 50 --output results/tensorrt_benchmark.json
```

Expected benchmark behavior:

- `engine.available=true`
- `status=success`
- `mean_ms > 0`
- `p99_ms > 0`
- `fps_value > 0`
- `model_metadata.inputs` contains `images`
- `model_metadata.outputs` contains `output0`

## Benchmark Interpretation

InferEdgeRuntime measures end-to-end inference latency. The reported `latency_ms` values include memory transfer and synchronization overhead in addition to backend execution.

Do not directly compare InferEdgeRuntime TensorRT latency with `trtexec` GPU latency. `trtexec` reports lower-level metrics such as GPU latency, Host latency, enqueue time, and H2D/D2H latency separately. InferEdgeRuntime currently reports a deployment-oriented wall-clock latency, so it is normal for Runtime latency to be larger than `trtexec` GPU latency.

This makes InferEdgeRuntime results more representative of the simple runtime path used for deployment and downstream InferEdgeLab comparison. See [docs/benchmark_policy.md](docs/benchmark_policy.md) for the full measurement policy.

Output modes:

- `--output results/foo.json`: writes to an explicit path.
- `--output auto`: writes to an auto-generated filename under `results/`.

Auto filename rule:

```text
{model}__{engine}__{device}__{precision}__b{batch}__h{height}w{width}__{timestamp}.json
```

Example:

```text
toy224__onnxruntime__cpu__fp32__b1__h224w224__20260426T115825Z.json
```

Every run also writes the same JSON content to `results/latest.json`. This stable file is useful for quick handoff to InferEdgeLab or small scripts that only need the most recent result.

## JSON Result Schema

Runtime JSON results include nested structured fields for detailed reporting and top-level compatibility fields for quick comparison.

Main nested fields:

- `schema_version`
- `manifest_path`
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

The `extra` object includes:

- `runtime`
- `json_export`
- `output_mode`: `auto` or `explicit`
- `latest_path`: currently `results/latest.json`
- `manifest_recorded`: `true` when `--manifest` was provided, otherwise `false`
- `manifest_precision`: recorded from `artifact.precision`
- `manifest_format`: recorded from `artifact.format`
- `compare_ready`: currently `true`
- `compare_key`
- `backend_key`
- `compare_model_source`: `manifest_model_name` or `model_path`
- `compare_model_name`: normalized model component used by `compare_key`

Top-level compatibility fields:

- `compare_key`
- `backend_key`
- `runtime_role`
- `model_name`
- `manifest_path`
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

## Forge Manifest Handoff Preparation

Runtime can now record a manifest path produced by Forge or another build stage and apply a limited set of manifest values as default runtime config. CLI-provided values always take priority over manifest defaults.

Sample manifest:

- `examples/manifest.sample.json`

Current behavior:

- Runtime records the `--manifest` path in the result JSON.
- Runtime reads limited defaults from `examples/manifest.sample.json` style manifests.
- Runtime applies manifest defaults only when the same value was not provided directly by CLI.

Applied manifest fields:

- `artifact.model_path`
- `runtime.engine`
- `runtime.device`
- `runtime.batch`
- `runtime.height`
- `runtime.width`

Recorded-only manifest fields:

- `artifact.precision`
- `artifact.format`

Compare-key manifest fields:

- `artifact.model_name`

Not applied yet:

- `warmup`
- `runs`
- `output`
- arbitrary metadata

Default build example:

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --output auto
```

The sample manifest uses `/path/to/model.onnx` as a placeholder. For a real run, either edit a local manifest outside the repository to point at a real model or override the model path from the CLI.

CLI override example:

```bash
./build-ort/inferedge-runtime --manifest examples/manifest.sample.json --model /Users/GwonHyeokJun/Desktop/edgebench/models/toy224.onnx --batch 1 --height 224 --width 224 --output auto
```

Draft manifest schema direction:

    schema_version: inferedge-forge-manifest-v1
    artifact:
      model_path: /path/to/model.onnx
      model_name: toy224.onnx
      precision: fp32
      format: onnx
    runtime:
      engine: onnxruntime
      device: cpu
      batch: 1
      height: 224
      width: 224
    metadata:
      source: InferEdgeForge
      created_at: 2026-04-26T12:00:00Z
      notes: optional build notes

## InferEdgeLab Compatibility

Runtime JSON results include both nested structured fields and top-level compatibility fields.

The nested fields are intended for detailed reports and future schema expansion. The top-level compatibility fields are intended for quick comparison in InferEdgeLab and EdgeBench-style loaders without deep nested parsing.

Runtime does not perform comparison calculations. It only writes compare-ready metadata that Lab can consume:

- `compare_key`: groups results from the same model and input condition, such as `toy224__b1__h224w224__fp32`
- `backend_key`: identifies the backend/device pair, such as `onnxruntime__cpu` or `tensorrt__jetson`
- `runtime_role`: fixed to `runtime-result`

The model component of `compare_key` prefers manifest `artifact.model_name` when available, then falls back to the CLI `--model` path stem. This lets TensorRT artifacts with generic filenames such as `model.engine` still produce a model-specific key like `yolov8n__b1__h640w640__fp32` when Forge supplies `artifact.model_name`.

InferEdgeLab can compare results that share the same `compare_key` and use `backend_key` to distinguish backend/device variants.

Forge -> Runtime -> Lab flow:

1. Forge builds or exports model artifacts.
2. Runtime runs ONNX Runtime benchmark and writes JSON result.
3. Lab reads JSON results and compares/report performance.

## Repository Layout

```text
.
├── CMakeLists.txt
├── CHANGELOG.md
├── include/
│   └── inferedge_runtime/
│       ├── cli.hpp
│       ├── engine.hpp
│       ├── manifest.hpp
│       ├── result_writer.hpp
│       ├── version.hpp
│       └── engines/
│           ├── onnxruntime_engine.hpp
│           └── tensorrt_engine.hpp
├── src/
│   ├── cli.cpp
│   ├── engine.cpp
│   ├── main.cpp
│   ├── manifest.cpp
│   ├── result_writer.cpp
│   └── engines/
│       ├── onnxruntime_engine.cpp
│       └── tensorrt_engine.cpp
├── scripts/
│   ├── smoke_default.sh
│   └── smoke_ort.sh
├── docs/
│   ├── benchmark_policy.md
│   ├── mvp_validation.md
│   └── tensorrt_backend_plan.md
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
- [x] GitHub Actions CI smoke tests
- [x] Auto result naming and latest.json handoff
- [x] Manifest path recording for Forge handoff preparation
- [x] Example Forge manifest
- [x] Forge manifest parsing and config default apply
- [ ] Robust manifest parser or external JSON dependency decision
- [x] TensorRT backend stub
- [x] TensorRT backend implementation plan
- [x] TensorRT CMake link validation
- [x] TensorRT engine deserialization on Jetson
- [x] TensorRT metadata extraction
- [x] TensorRT one-shot inference
- [x] TensorRT benchmark runner on Jetson
- [ ] TensorRT output post-processing
- [ ] TensorRT/ONNX Runtime comparison through InferEdgeLab
- [ ] InferEdgeLab direct import workflow

## Version

Current version: v0.1.0 (MVP)

See [CHANGELOG.md](CHANGELOG.md) for details.
