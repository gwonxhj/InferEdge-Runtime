# InferEdge-Runtime

C++ runtime execution and result export layer  
(ONNX Runtime · TensorRT Jetson · latency statistics · Lab-compatible JSON)

Language: English | [한국어](README.ko.md)

![CI](https://github.com/gwonxhj/InferEdge-Runtime/actions/workflows/ci.yml/badge.svg)

**GitHub description:** C++ runtime execution and result export layer for ONNX Runtime/TensorRT edge inference validation.

## Summary

- C++ execution layer for the InferEdge validation pipeline
- Runs ONNX Runtime CPU and Jetson TensorRT benchmark paths
- Measures latency statistics and FPS from real Runtime executions
- Exports Lab-compatible result JSON for compare/report/deployment decision flows
- Preserves Forge manifest source model identity when running built artifacts
- Optionally records Forge `agent_manifest.json` context as an additive `agent` result block

## What Makes InferEdge-Runtime Different?

InferEdge-Runtime is not a benchmark wrapper.

It is an execution evidence layer that:

- validates or runs model/artifact inputs at the Runtime boundary
- records latency, FPS, system, and provenance context
- exports structured evidence that Lab can compare and review
- keeps runtime execution separate from Lab's deployment decision policy

## Project Overview

InferEdgeRuntime is a C++ Edge AI runtime for on-device inference and benchmarking.

It is the Runtime stage of the InferEdge portfolio pipeline. InferEdgeForge prepares model artifacts, InferEdgeRuntime runs and benchmarks those artifacts on target devices, and InferEdgeLab analyzes the exported result JSON files.

## Release Status

InferEdgeRuntime v0.1.0 is a validated MVP release.

- ONNX Runtime CPU backend: fully functional
- Benchmark + JSON export: stable
- Forge/Lab pipeline: integrated through manifest/result JSON and worker boundary contracts
- TensorRT backend: benchmark execution on Jetson linked builds

## InferEdge Pipeline Position

InferEdgeRuntime is the C++ execution/result export layer of the larger InferEdge validation pipeline:

```text
ONNX model
-> InferEdgeForge build
-> metadata / manifest / worker runtime summary
-> InferEdgeRuntime validation / result export
-> InferEdgeLab compare / API / job workflow / deployment_decision
-> optional InferEdgeAIGuard provenance diagnosis
-> deploy / review / blocked decision

Experiment hygiene / comparability layer:
InferEdgeEnv -> v0.1.5 v1-complete local-first run evidence registry / comparability checker
```

In that pipeline, Runtime is responsible for the execution boundary: it validates or runs model/artifact inputs, measures latency, exports Lab-compatible result JSON, and can emit dry-run worker response payloads for Lab integration smoke tests.

Implemented today:

- ONNX Runtime C++ MVP path and benchmark/result JSON export
- Jetson TensorRT linked-build benchmark/result JSON export
- Lab-compatible result fields for compare/report/deployment decision flows
- Forge metadata/manifest handoff validation
- manifest source model identity preservation for compare-ready TensorRT engine results
- Lab `worker_request` dry-run validation
- Lab worker completed/failed response dry-run export
- optional `agent` result block for agent workload metadata without changing the base Runtime schema

Planned later:

- full worker daemon integration
- real Lab-triggered Forge/Runtime execution
- production queue or job runner infrastructure
- production hardening beyond the current manual/dev linked-build validation path

Runtime does not own comparison policy or final deployment judgement. InferEdgeLab owns `deployment_decision`, while Runtime supplies trustworthy execution and profiling evidence.

Portfolio boundary: InferEdgeLab is the validation / decision layer. InferEdgeEnv is the v0.1.5 v1-complete experiment hygiene / comparability layer; it records whether benchmark evidence can be trusted and compared without replacing Runtime execution or Lab deployment decisions.

## Current Capabilities

- C++17 + CMake build
- CLI option validation
- ONNX Runtime external link configuration
- ONNX model metadata loading
- float32 dummy input generation
- optional OpenCV-based real image input preprocessing
- ONNX Runtime CPU inference benchmark
- latency mean/min/max/std/p50/p90/p95/p99
- FPS calculation
- JSON result export
- Lab-compatible top-level fields
- automatic result naming and `results/latest.json` handoff
- limited manifest default apply for Forge handoff preparation
- Lab worker adapter contract fixture/test coverage
- Lab worker response dry-run export for contract smoke testing
- TensorRT backend stub for default/non-linked builds
- TensorRT engine deserialization and metadata extraction on Jetson linked builds
- TensorRT one-shot dummy inference on Jetson linked builds
- TensorRT benchmark runner on Jetson linked builds
- Jetson Evidence Track fields for power mode, jetson_clocks, tegrastats summary, and Lab-compatible result import
- optional Forge `agent_manifest.json` ingestion for `agent_id`, `task_id`, priority, latency budget, fallback, and telemetry context
- documented benchmark measurement policy

## Current Limitations

- default macOS build uses ONNX Runtime/stub paths unless optional backends are explicitly linked
- float32 input only
- real image preprocessing requires `INFEREDGE_ENABLE_OPENCV=ON`
- no TensorRT output post-processing yet
- float32 TensorRT buffers only at current stage
- no multi-input advanced dynamic shape support yet
- OpenCV and CUDA are not linked in the default build
- manifest parsing is limited to the sample Forge handoff schema
- no full general-purpose JSON parser yet
- `agent` task context is additive metadata only; scheduling/policy decisions remain outside Runtime
- contract tests and smoke tests cover the current handoff/result schemas; broader backend integration tests remain future work
- GitHub Actions currently runs default smoke test only
- ORT linked smoke test remains local/manual because it requires external ONNX Runtime and model files

TensorRT backend execution is implemented for Jetson-oriented linked builds. The current Mac/default build keeps TensorRT as a stub and does not link TensorRT or CUDA. See [docs/tensorrt_backend_plan.md](docs/tensorrt_backend_plan.md) for the Jetson Orin Nano implementation plan.

## Jetson Evidence Track

The current Jetson Evidence Track has been validated on Jetson Orin Nano through a TensorRT FP16 linked build and Lab-compatible Runtime JSON export.
These records are deployment validation evidence, not a production inference server or a `trtexec` GPU-only benchmark.

| Evidence | Backend | Precision | Power Mode | Mean ms | P95 ms | P99 ms | FPS |
|---|---|---|---|---:|---:|---:|---:|
| TensorRT short smoke | tensorrt__jetson | FP16 | 25W | 10.066401 | 15.476641 | 15.548438 | 99.340373 |
| TensorRT power-mode evidence | tensorrt__jetson | FP16 | 15W | 10.799106 | 15.438690 | 15.529218 | 92.600262 |

The 15W and 25W outputs include tegrastats-derived context and should be interpreted as different run configurations.
InferEdgeLab owns comparison and deployment decision interpretation.
Current committed Jetson snapshots are `short_smoke` evidence. The Markdown report path now records `capture_depth` so future 5-10 minute Jetson runs can be distinguished from quick contract/device smoke evidence without changing the Runtime JSON contract.

## Requirements

- CMake 3.16+
- C++17 compiler
- Optional: ONNX Runtime C/C++ package
- Optional: OpenCV for `--input <image_path>` real image preprocessing
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

Build with ONNX Runtime and OpenCV real image input enabled:

```bash
cmake -S . -B build-ort-opencv -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT=$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0 -DINFEREDGE_ENABLE_OPENCV=ON
cmake --build build-ort-opencv
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
- Lab worker adapter planning is documented in [docs/lab_worker_adapter_contract.md](docs/lab_worker_adapter_contract.md).
- `--lab-worker-request <path> --validate-lab-worker-request` validates Lab worker request JSON and exits without inference.
- Forge summary-origin worker request compatibility is covered by `tests/fixtures/forge_summary_worker_request.json`.
- `--lab-worker-request <path> --export-worker-response <path> --worker-response-status completed|failed` writes a Lab worker response contract payload without inference.
- CLI-provided values always take priority over manifest defaults.
- `--batch`, `--height`, and `--width` resolve dynamic dummy input dimensions.
- `--input` uses a real image input instead of dummy zeros when OpenCV support is enabled.
- Static model dimensions take precedence over CLI shape overrides.
- `--warmup` controls untimed warmup iterations.
- `--runs` controls timed iterations used for latency and FPS statistics.
- `--run-once` runs one inference without benchmark timing.
- `--output` writes the benchmark result JSON and creates missing output directories.

## Real Inference Mode

`--input <image_path>` enables real image input mode. In this mode, Runtime loads the image with OpenCV, converts BGR to RGB, resizes it to the resolved model input size, normalizes values to `0.0..1.0`, and writes a float32 NCHW tensor with shape `[batch, 3, height, width]`.

The default build remains dependency-free. Real image input requires an OpenCV-enabled build:

```bash
cmake -S . -B build-ort-opencv -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT=$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0 -DINFEREDGE_ENABLE_OPENCV=ON
cmake --build build-ort-opencv
```

ONNX Runtime example:

```bash
./build-ort-opencv/inferedge-runtime --model yolov8n.onnx --input test.jpg --engine onnxruntime --device cpu --batch 1 --height 640 --width 640 --run-once --output results/real_input_onnx.json
```

TensorRT linked builds use the same `--input` path to fill TensorRT input buffers when OpenCV support is enabled. If `--input` is provided without OpenCV support, Runtime fails with a clear configuration error instead of silently falling back to dummy input.

Runtime records input mode metadata under JSON `extra`:

- `input_mode`: `dummy` or `image`
- `input_path`: the provided image path, or an empty string for dummy mode
- `input_preprocess`: `opencv_bgr_to_rgb_resize_float32_nchw` or `dummy_zero_float32`

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
./build-trt/inferedge-runtime \
  --model /home/risenano01/InferEdgeForge/builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine \
  --engine tensorrt \
  --device jetson \
  --power-mode 15W \
  --jetson-clocks on \
  --tegrastats-log results/tegrastats_yolov8n_trt_fp16_15w.log \
  --batch 1 \
  --height 640 \
  --width 640 \
  --warmup 10 \
  --runs 50 \
  --output results/tensorrt_benchmark.json
```

Expected benchmark behavior:

- `engine.available=true`
- `status=success`
- `mean_ms > 0`
- `p99_ms > 0`
- `fps_value > 0`
- `model_metadata.inputs` contains `images`
- `model_metadata.outputs` contains `output0`

Jetson evidence fields are optional CLI inputs used to preserve validation context before importing results into InferEdgeLab:

- `--power-mode`: records the Jetson power mode label, such as `15W`, `25W`, or `MAXN`
- `--jetson-clocks`: records the observed `jetson_clocks` state, such as `on`, `off`, or `unknown`
- `--tegrastats-log`: records and parses a tegrastats log into `jetson_evidence.tegrastats_summary`

These fields prepare the Runtime result for Jetson Evidence Track validation. They do not imply that every TensorRT/GPU benchmark or INT8 calibration path is complete.

Jetson evidence can also be exported as Markdown for portfolio/review handoff:

```bash
./build/inferedge-runtime \
  --report-jetson-evidence \
  --result-json tests/fixtures/jetson_tensorrt_25w_result.json \
  --tegrastats-log tests/fixtures/tegrastats_sample.log \
  --report-output reports/jetson_evidence_summary.md

./build/inferedge-runtime \
  --compare-power-modes \
  --base-result tests/fixtures/jetson_tensorrt_25w_result.json \
  --candidate-result tests/fixtures/jetson_tensorrt_15w_result.json \
  --report-output reports/jetson_power_mode_comparison.md
```

The Markdown reports summarize Runtime JSON and tegrastats evidence only. InferEdgeLab remains responsible for comparison policy and deployment decision interpretation.
They also label evidence depth (`short_smoke`, `sustained_candidate`, or `sustained`) from Runtime run count and tegrastats sample count, preventing short Jetson smoke evidence from being overstated as sustained thermal validation.

Committed report snapshots:

- [Jetson evidence summary](docs/reports/jetson_evidence_summary.md)
- [Jetson power-mode comparison](docs/reports/jetson_power_mode_comparison.md)

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

Runtime JSON results include nested structured fields for detailed reporting and top-level compatibility fields for quick comparison. This output is the Runtime side of the Forge -> Runtime -> Lab contract documented by InferEdgeLab, and is intended to be consumed by Lab compare, report, and deployment decision flows.

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
- `jetson_evidence`
- `agent` (optional, emitted only when `--agent-manifest` is provided)
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
- `power_mode`: optional Jetson power mode label
- `jetson_clocks`: optional `jetson_clocks` state
- `tegrastats_log_path`: optional source log path for thermal/power evidence
- `tegrastats_status`: `not_provided`, `parsed`, `unavailable`, or `no_samples`
- `agent_manifest_recorded`: `true` when `--agent-manifest` was provided, otherwise `false`
- `compare_ready`: currently `true`
- `compare_key`
- `backend_key`
- `compare_model_source`: `manifest_source_model` or `model_path`
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
- `p50_ms`
- `p95_ms`
- `p99_ms`
- `fps_value`
- `success`
- `status`

The schema regression fixture lives at `tests/fixtures/lab_compatible_result.json`, and `tests/test_lab_result_schema.py` validates both the fixture and smoke-generated Runtime JSON.

See [examples/README.md](examples/README.md) for command examples and compact JSON field notes.

## Agent Runtime Result Context

Runtime can optionally read a Forge `agent_manifest.json` and append an additive `agent` block to the result JSON.

This is the first bridge toward the reliable edge agent runtime direction. It records task metadata such as `agent_id`, `task_id`, `agent_type`, priority, latency budget, queue wait, fallback usage, and telemetry context while preserving the base Lab-compatible Runtime result schema.

Example:

```bash
./build/inferedge-runtime \
  --agent-manifest tests/fixtures/agent_manifest.json \
  --agent-task-id task_camera_frame_0001 \
  --agent-queue-wait-ms 7 \
  --agent-fallback-used \
  --warmup 1 \
  --runs 1 \
  --output results/agent_runtime_result.json
```

Contract notes:

- `agent` is optional and absent from existing Runtime result files.
- `agent.schema_version` is `inferedge-runtime-agent-task-v1`.
- `agent.source_contract` is `inferedge-agent-manifest-v1`.
- `agent.deadline_missed` is computed from mean latency and `latency_budget_ms` unless explicitly set with `--agent-deadline-missed`.
- Runtime records task context only; scheduling/policy decisions remain Orchestrator/Lab responsibilities.

See [docs/agent_runtime_result_contract.md](docs/agent_runtime_result_contract.md) for the full contract.

## Forge Handoff Input Preparation

Runtime can now record a manifest path produced by Forge or another build stage and apply a limited set of manifest values as default runtime config. It can also validate Forge `metadata.json` / `manifest.json` handoff fixtures without executing an artifact. CLI-provided values always take priority over handoff defaults.

Sample manifest:

- `examples/manifest.sample.json`
- `tests/fixtures/forge_handoff_manifest.json`
- `tests/fixtures/forge_handoff_metadata.json`

Current behavior:

- Runtime records the `--manifest` path in the result JSON.
- Runtime reads limited defaults from `examples/manifest.sample.json` and Forge `manifest.json` style handoffs.
- Runtime can read Forge `metadata.json` with `--forge-metadata`.
- Runtime applies handoff defaults only when the same value was not provided directly by CLI.
- `--validate-forge-handoff` parses and validates the handoff input, then exits before execution.

Applied manifest fields:

- `artifact.model_path`
- `artifact.path`
- `lab_compat.runtime.runtime_artifact_path`
- `runtime.engine`
- `runtime.device`
- `runtime.precision`
- `runtime.batch`
- `runtime.height`
- `runtime.width`

Recorded-only manifest fields:

- `artifact.precision`
- `artifact.format`
- `artifact.sha256`
- `source_model.sha256`
- `build.preset_name`
- `build.build_id`

Compare-key manifest fields:

- `source_model.path`
- `artifact.model_name` as a fallback when `source_model.path` is absent

Not applied yet:

- `warmup`
- `runs`
- `output`
- arbitrary metadata

Default build example:

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --output auto
```

Forge handoff validation examples:

```bash
./build/inferedge-runtime --forge-manifest tests/fixtures/forge_handoff_manifest.json --validate-forge-handoff
./build/inferedge-runtime --forge-metadata tests/fixtures/forge_handoff_metadata.json --validate-forge-handoff
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
- top-level latency aliases: `mean_ms`, `p50_ms`, `p95_ms`, and `p99_ms`

The model component of `compare_key` prefers manifest `source_model.path` when available, then falls back to the CLI `--model` path stem. This lets TensorRT artifacts with generic filenames such as `model.engine` still produce a source-model-specific key like `yolov8n__b1__h640w640__fp32` when Forge supplies source model identity.

InferEdgeLab can compare results that share the same `compare_key` and use `backend_key` to distinguish backend/device variants.

Forge -> Runtime -> Lab flow:

1. Forge builds or exports model artifacts and provenance.
2. Runtime runs ONNX Runtime or Jetson TensorRT linked-build benchmarks and writes Lab-compatible JSON results.
3. Lab reads JSON results and owns comparison, reporting, API/job workflow, and deployment decision output.

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
- [x] Optional real image input inference mode
- [ ] TensorRT output post-processing
- [x] TensorRT/ONNX Runtime comparison through InferEdgeLab demo evidence
- [x] InferEdgeLab direct import workflow through Local Studio / Lab result ingest

## Version

Current version: v0.1.0 (MVP)

See [CHANGELOG.md](CHANGELOG.md) for details.
