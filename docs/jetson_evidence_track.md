# Jetson Evidence Track

This document records the Runtime side of the InferEdge Jetson Evidence Track.

The goal is not to create a new inference server or benchmark product. The goal is to produce Lab-compatible Runtime JSON that preserves Jetson runtime context for deployment validation.

## Scope

Current evidence:

- ONNX Runtime CPU baseline
- TensorRT FP16 Jetson candidate
- power mode context, such as `15W` or `25W`
- `jetson_clocks` state
- `tegrastats` log summary
- p50 / p95 / p99 / FPS Runtime JSON fields

## Observed Jetson Evidence

| Evidence | Result path | Power Mode | Mean ms | P95 ms | P99 ms | FPS | Notes |
|---|---|---|---:|---:|---:|---:|---|
| TensorRT FP16 25W | `results/jetson_evidence/yolov8n_trt_fp16_25w_20260504T170039Z.json` | 25W | 10.066401 | 15.476641 | 15.548438 | 99.340373 | Local Studio candidate fixture |
| TensorRT FP16 15W | `results/jetson_evidence/yolov8n_trt_fp16_15w_20260504T171959Z.json` | 15W | 10.799106 | 15.438690 | 15.529218 | 92.600262 | Power-mode comparison fixture |

The 25W and 15W runs are not same-run-config regression evidence.
They are system evidence that power mode, tegrastats, p95/p99 latency, and FPS should travel with the Runtime result JSON.
The committed snapshots are currently `short_smoke` evidence. Runtime Markdown reports now label `capture_depth` so a future 5-10 minute Jetson run can be recorded as `sustained_candidate` or `sustained` without changing the Lab-compatible result JSON schema.

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

The same smoke also verifies local Markdown report generation from fixture evidence:

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

These reports are handoff summaries for review/portfolio documentation. They do not change the Runtime JSON contract and do not perform Lab deployment decision logic.
They include a run-depth section derived from `run_config.runs` and `jetson_evidence.tegrastats_summary.sample_count`. Short smoke evidence must not be described as sustained thermal validation.

Committed report snapshots:

- [Jetson evidence summary](reports/jetson_evidence_summary.md)
- [Jetson power-mode comparison](reports/jetson_power_mode_comparison.md)

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
- `capture_depth=short_smoke` means the result is device/contract evidence, not sustained stability evidence.
- INT8 calibration remains future work.
- Runtime exports evidence; InferEdgeLab owns interpretation and deployment decision.
