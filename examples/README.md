# Examples

This directory documents common InferEdgeRuntime command flows. Model files are not included in this repository. Use local model paths such as `/path/to/model.onnx`.

## Example 1: Default Build Skipped Result

The default build does not link ONNX Runtime. It should still create a skipped JSON result.

```bash
cmake -S . -B build
cmake --build build
./build/inferedge-runtime --model models/sample.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 1 --runs 1 --output results/default_skipped.json
python3 -m json.tool results/default_skipped.json
```

Expected result:

- `engine.available` is `false`
- `benchmark.status` is `skipped`
- `latency_ms.samples` is empty

## Example 2: ONNX Runtime Linked Benchmark

Keep ONNX Runtime outside the repository.

```bash
cmake -S . -B build-ort -DINFEREDGE_ENABLE_ORT=ON -DINFEREDGE_ORT_ROOT=$HOME/onnxruntime/onnxruntime-osx-arm64-1.25.0
cmake --build build-ort
./build-ort/inferedge-runtime --model /path/to/model.onnx --engine onnxruntime --device cpu --batch 1 --height 224 --width 224 --warmup 3 --runs 10 --output results/ort_cpu.json
```

Expected result:

- `engine.available` is `true`
- input/output metadata is printed
- benchmark latency and FPS are printed
- `results/ort_cpu.json` is written

## Example 3: JSON Validation

```bash
python3 -m json.tool results/ort_cpu.json > /tmp/ort_cpu_pretty.json
```

## Example JSON Fields

Nested fields for detailed reporting:

- `model.path`
- `engine.backend`
- `run_config.runs`
- `latency_ms.mean`
- `latency_ms.samples`
- `benchmark.status`
- `model_metadata.inputs`
- `model_metadata.outputs`

Top-level compatibility fields for quick comparison:

- `model_name`
- `engine_name`
- `device_name`
- `mean_ms`
- `p99_ms`
- `fps_value`
- `success`
- `status`
