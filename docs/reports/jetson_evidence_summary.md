# InferEdge Runtime Jetson Evidence Summary

This report summarizes Runtime JSON and optional tegrastats evidence for Lab-compatible deployment validation.
It is not a production inference server report or a TensorRT INT8 calibration workflow.

## Runtime Result

| Field | Value |
|---|---|
| source | `tests/fixtures/jetson_tensorrt_25w_result.json` |
| model | `model.engine` |
| backend_key | `tensorrt__jetson` |
| compare_key | `yolov8n__b1__h640w640__fp16` |
| engine | `tensorrt` |
| device | `jetson` |
| precision | `fp16` |
| power_mode | `25W` |
| jetson_clocks | `unknown` |
| mean_ms | 10.0664 |
| p50_ms | 9.9086 |
| p95_ms | 15.4766 |
| p99_ms | 15.5484 |
| fps | 99.3404 |
| timestamp | `2026-05-04T17:00:41Z` |

## Tegrastats Summary

| Field | Value |
|---|---|
| source | `embedded result JSON` |
| status | `parsed` |
| sample_count | 3 |
| ram_used_mb_avg | 947.6667 |
| ram_used_mb_max | 1072 |
| ram_total_mb | 7620 |
| max_temp_c | 40.656 |
| max_temp_name | `gpu` |
| vdd_in_mw_avg | 4863 |
| vdd_in_mw_max | 5827 |

## Lab Handoff

- Lab-compatible import path: `tests/fixtures/jetson_tensorrt_25w_result.json`
- Runtime supplies execution evidence. InferEdgeLab owns comparison and deployment decision interpretation.
