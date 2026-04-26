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
- automatic result naming and `results/latest.json` handoff
- limited manifest default apply for Forge handoff preparation

## Current Limitations

- ONNX Runtime CPU only
- float32 input only
- TensorRT not implemented yet
- OpenCV/CUDA not linked
- no real image preprocessing yet
- manifest parsing is limited to the sample Forge handoff schema
- no full general-purpose JSON parser yet
- no full unit test suite yet
- GitHub Actions currently runs default smoke test only
- ORT linked smoke test remains local/manual because it requires external ONNX Runtime and model files

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
- `--output` writes the benchmark result JSON and creates missing output directories.

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

Top-level compatibility fields:

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
- [x] GitHub Actions CI smoke tests
- [x] Auto result naming and latest.json handoff
- [x] Manifest path recording for Forge handoff preparation
- [x] Example Forge manifest
- [x] Forge manifest parsing and config default apply
- [ ] Robust manifest parser or external JSON dependency decision
- [ ] TensorRT backend on Jetson
- [ ] InferEdgeLab direct import workflow
