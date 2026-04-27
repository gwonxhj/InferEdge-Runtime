# Tests

InferEdgeRuntime has a GitHub Actions default smoke workflow. Use the scripted smoke tests locally first, then fall back to the manual checklist when debugging a specific failure.

## Recommended Smoke Scripts

Default dependency-free build:

```bash
scripts/smoke_default.sh
```

ONNX Runtime linked build with a local package and model:

```bash
scripts/smoke_ort.sh "$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0" /path/to/model.onnx
```

The scripts validate build success, CLI execution, JSON validity, Lab-compatible top-level fields, and benchmark status.

The Lab-compatible result contract is also covered by `tests/test_lab_result_schema.py` and `tests/fixtures/lab_compatible_result.json`. The default smoke script validates the committed fixture and the generated `results/smoke_default.json` with the same checker.

## Manual Smoke Checklist

## Build Smoke Test

```bash
cmake -S . -B build
cmake --build build
```

## CLI Help/Version Test

```bash
./build/inferedge-runtime --help
./build/inferedge-runtime --version
```

## Default Build Skipped Benchmark Test

```bash
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/default_smoke.json
python3 -m json.tool results/default_smoke.json > /tmp/default_smoke_pretty.json
```

Expected:

- benchmark status is `skipped`
- JSON is valid

## ORT Linked Build Test

```bash
cmake -S . -B build-ort -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT=$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0
cmake --build build-ort
./build-ort/inferedge-runtime --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 3 --runs 10 --output results/ort_smoke.json
```

Expected:

- backend availability is `true`
- benchmark status is `success`
- latency and FPS are printed

## Input Mode Metadata Test

Dummy input metadata should be present in every result JSON:

```bash
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/input_dummy_metadata.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/input_dummy_metadata.json").read_text())
extra = data["extra"]
assert extra["input_mode"] == "dummy"
assert extra["input_path"] == ""
assert extra["input_preprocess"] == "dummy_zero_float32"
print("dummy input metadata ok")
PY
```

`--input` requires an OpenCV-enabled build:

```bash
./build/inferedge-runtime --model models/sample.onnx --input examples/sample.jpg --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/input_without_opencv.json
```

Expected:

- command exits non-zero
- stderr mentions `OpenCV-enabled build`

## OpenCV Real Input Smoke Test

Run this only on a machine where OpenCV provides a CMake package. Some macOS environments do not have `OpenCVConfig.cmake`; in that case, treat configure failure as an environment setup issue and run the check on Jetson or another OpenCV-enabled machine.

```bash
cmake -S . -B build-ort-opencv -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT=$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0 -DINFEREDGE_ENABLE_OPENCV=ON
cmake --build build-ort-opencv
./build-ort-opencv/inferedge-runtime --model /path/to/model.onnx --input /path/to/sample.jpg --engine onnxruntime --device cpu --batch 1 --height 640 --width 640 --warmup 3 --runs 10 --output results/ort_real_input_metadata.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/ort_real_input_metadata.json").read_text())
assert data["status"] == "success"
assert data["success"] is True
assert data["extra"]["input_mode"] == "image"
assert data["extra"]["input_path"]
assert data["extra"]["input_preprocess"] == "opencv_bgr_to_rgb_resize_float32_nchw"
print("ort real image input smoke ok")
PY
```

## Missing Model Error Test

```bash
./build-ort/inferedge-runtime --model models/missing.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/missing.json
```

Expected:

- command exits non-zero
- stderr includes `model file not found`

## Invalid Runs Validation Test

```bash
./build-ort/inferedge-runtime --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 0 --runs 0 --output results/invalid.json
```

Expected:

- command exits non-zero
- stderr includes `minimum: 1`

## Manifest Path Recording Smoke Test

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/manifest_sample_smoke.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/manifest_sample_smoke.json").read_text())
assert data["manifest_path"] == "examples/manifest.sample.json"
assert data["run_config"]["manifest_path"] == "examples/manifest.sample.json"
assert data["extra"]["manifest_recorded"] is True
assert data["runtime_role"] == "runtime-result"
assert "compare_key" in data
assert "backend_key" in data
assert data["extra"]["compare_ready"] is True
assert data["extra"]["compare_key"] == data["compare_key"]
assert data["extra"]["backend_key"] == data["backend_key"]
assert data["extra"]["compare_model_source"] in {"manifest_model_name", "model_path"}
assert data["extra"]["compare_model_name"]
print("manifest sample smoke ok")
PY
```

## Manifest Parser Smoke Test

Manifest-only default apply:

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --output results/manifest_only.json
```

Expected:

- `model_path` is loaded from `artifact.model_path`
- `engine`, `device`, `batch`, `height`, and `width` are loaded from `runtime`
- `manifest_applied` is `true`

Manifest-based compare key:

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --model models/model.engine --engine tensorrt --device jetson --batch 1 --height 640 --width 640 --warmup 1 --runs 1 --output results/manifest_compare_key.json
```

Expected:

- `compare_key` uses `artifact.model_name`, for example `yolov8n__b1__h640w640__fp32`
- `extra.compare_model_source` is `manifest_model_name`
- `extra.compare_model_name` is the normalized model stem

CLI override priority:

```bash
./build/inferedge-runtime --manifest examples/manifest.sample.json --model models/override.onnx --batch 2 --height 320 --width 320 --output results/manifest_override.json
```

Expected:

- CLI `--model` wins over `artifact.model_path`
- CLI `--batch`, `--height`, and `--width` win over manifest values

## TensorRT Stub Smoke Test

```bash
./build/inferedge-runtime --model models/sample.engine --engine tensorrt --device jetson --batch 1 --height 640 --width 640 --warmup 1 --runs 1 --output results/tensorrt_stub_smoke.json
python3 -m json.tool results/tensorrt_stub_smoke.json > /tmp/tensorrt_stub_smoke_pretty.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/tensorrt_stub_smoke.json").read_text())
assert data["engine_name"] == "tensorrt"
assert data["engine_backend"] == "tensorrt"
assert data["device_name"] == "jetson"
assert data["status"] == "skipped"
assert data["success"] is False
assert data["engine"]["available"] is False
print("tensorrt stub smoke ok")
PY
```

## TensorRT Metadata Extraction Smoke Test

Run this on Jetson after TensorRT and CUDA are installed. This validates TensorRT link configuration, `.engine` deserialization, and input/output metadata extraction.

```bash
cmake -S . -B build-trt -DINFEREDGE_ENABLE_TENSORRT=ON
cmake --build build-trt
./build-trt/inferedge-runtime --model /home/risenano01/InferEdgeForge/builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine --engine tensorrt --device jetson --batch 1 --height 640 --width 640 --warmup 1 --runs 1 --output results/tensorrt_metadata_check.json
python3 -m json.tool results/tensorrt_metadata_check.json > /tmp/tensorrt_metadata_check_pretty.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/tensorrt_metadata_check.json").read_text())
assert data["engine_name"] == "tensorrt"
assert data["engine_backend"] == "tensorrt"
assert data["device_name"] == "jetson"
assert data["engine"]["available"] is True
assert data["status"] == "success"
assert data["success"] is True
inputs = data["model_metadata"]["inputs"]
outputs = data["model_metadata"]["outputs"]
assert inputs, inputs
assert outputs, outputs
assert any(t["name"] == "images" for t in inputs)
assert any(t["name"] == "output0" for t in outputs)
print("jetson tensorrt metadata extraction ok")
PY
```

## TensorRT One-shot Inference Smoke Test

Run this on Jetson after TensorRT and CUDA are installed. This validates one dummy inference execution without benchmark timing.

```bash
cmake -S . -B build-trt -DINFEREDGE_ENABLE_TENSORRT=ON
cmake --build build-trt
./build-trt/inferedge-runtime --model /home/risenano01/InferEdgeForge/builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine --engine tensorrt --device jetson --batch 1 --height 640 --width 640 --run-once --output results/tensorrt_run_once.json
python3 -m json.tool results/tensorrt_run_once.json > /tmp/tensorrt_run_once_pretty.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/tensorrt_run_once.json").read_text())
assert data["engine_name"] == "tensorrt"
assert data["engine_backend"] == "tensorrt"
assert data["device_name"] == "jetson"
assert data["engine"]["available"] is True
assert data["success"] is True
assert data["status"] == "success"
assert data["benchmark"]["message"] == "one-shot inference completed"
inputs = data["model_metadata"]["inputs"]
outputs = data["model_metadata"]["outputs"]
assert any(t["name"] == "images" for t in inputs)
assert any(t["name"] == "output0" for t in outputs)
print("jetson tensorrt run_once ok")
PY
```

## TensorRT Benchmark Smoke Test

Run this on Jetson after TensorRT and CUDA are installed. This validates warmup/timed inference execution and TensorRT benchmark JSON export.

```bash
cmake -S . -B build-trt -DINFEREDGE_ENABLE_TENSORRT=ON
cmake --build build-trt
./build-trt/inferedge-runtime --model /home/risenano01/InferEdgeForge/builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine --engine tensorrt --device jetson --batch 1 --height 640 --width 640 --warmup 10 --runs 50 --output results/tensorrt_benchmark.json
python3 -m json.tool results/tensorrt_benchmark.json > /tmp/tensorrt_benchmark_pretty.json
```

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("results/tensorrt_benchmark.json").read_text())
assert data["engine_name"] == "tensorrt"
assert data["engine_backend"] == "tensorrt"
assert data["device_name"] == "jetson"
assert data["engine"]["available"] is True
assert data["success"] is True
assert data["status"] == "success"
assert data["mean_ms"] > 0
assert data["p99_ms"] > 0
assert data["fps_value"] > 0
assert data["latency_ms"]["samples"]
assert len(data["latency_ms"]["samples"]) == 50
inputs = data["model_metadata"]["inputs"]
outputs = data["model_metadata"]["outputs"]
assert any(t["name"] == "images" for t in inputs)
assert any(t["name"] == "output0" for t in outputs)
print("jetson tensorrt benchmark ok")
PY
```

## Benchmark Policy Check

TensorRT benchmark values from InferEdgeRuntime should not be directly compared with `trtexec` GPU latency. Runtime latency can be larger because it measures an end-to-end wall-clock path that includes memory transfers and synchronization overhead.

Normal validation criteria:

- JSON `status` is `success`
- `mean_ms > 0`
- `p99_ms > 0`
- `fps_value > 0`
- `latency_ms.samples` length matches `runs`
- values do not swing wildly under the same model, device, and run configuration

## JSON Validity Test

```bash
python3 -m json.tool results/default_smoke.json > /tmp/default_smoke_pretty.json
python3 -m json.tool results/ort_smoke.json > /tmp/ort_smoke_pretty.json
```
