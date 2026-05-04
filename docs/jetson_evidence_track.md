# Jetson Evidence Track Preparation

This document prepares the Runtime side of the InferEdge Jetson Evidence Track.

The goal is not to create a new inference server or benchmark product. The goal is to produce Lab-compatible Runtime JSON that preserves Jetson runtime context for deployment validation.

## Scope

Prepared for the next Jetson smoke step:

- ONNX Runtime CPU baseline
- TensorRT FP16 Jetson candidate
- power mode context, such as `15W` or `25W`
- `jetson_clocks` state
- `tegrastats` log summary
- p50 / p95 / p99 / FPS Runtime JSON fields

Out of scope:

- production inference server
- cloud dashboard
- model zoo runner
- INT8 calibration automation
- CUDA pre/post-processing optimization

## Runtime JSON Contract

Runtime JSON now records Jetson evidence context in:

- `run_config.power_mode`
- `run_config.jetson_clocks`
- `run_config.tegrastats_log_path`
- `system.jetson`
- `jetson_evidence`
- `extra.power_mode`
- `extra.jetson_clocks`
- `extra.tegrastats_status`

`jetson_evidence.tegrastats_summary` contains:

- `status`
- `sample_count`
- `ram_used_mb_avg`
- `ram_used_mb_max`
- `ram_total_mb`
- `max_temp_c`
- `max_temp_name`
- `vdd_in_mw_avg`
- `vdd_in_mw_max`

## Pre-Jetson Local Validation

Run locally before moving to Jetson:

```bash
bash scripts/smoke_default.sh
python3 tests/test_manifest_compare_identity.py
```

The default smoke confirms that the JSON contract exists even when no Jetson context is provided.

## Jetson Smoke Template

On Jetson, capture tegrastats in parallel with the Runtime command:

```bash
tegrastats --interval 1000 --logfile results/tegrastats_yolov8n_trt_fp16_15w.log &
TEGRAPID=$!

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
  --output results/yolov8n_trt_fp16_15w.json

kill "$TEGRAPID"
```

Repeat with `--power-mode 25W` only after the Jetson power mode has actually been changed. Do not compare 15W and 25W results as the same run configuration.

## Validation Notes

- `mean_ms` alone is not enough for deployment evidence.
- `p95_ms`, `p99_ms`, FPS, power mode, and thermal behavior should be reviewed together.
- INT8 calibration remains future work.
- Runtime exports evidence; InferEdgeLab owns interpretation and deployment decision.
