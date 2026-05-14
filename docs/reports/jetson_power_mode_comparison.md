# InferEdge Runtime Jetson Power Mode Comparison

This report compares two Runtime result JSON files as Jetson system evidence.
Different power modes are not treated as the same run_config regression test.

## Compared Results

| Field | Base | Candidate |
|---|---|---|
| source | `tests/fixtures/jetson_tensorrt_25w_result.json` | `tests/fixtures/jetson_tensorrt_15w_result.json` |
| backend_key | `tensorrt__jetson` | `tensorrt__jetson` |
| compare_key | `yolov8n__b1__h640w640__fp16` | `yolov8n__b1__h640w640__fp16` |
| power_mode | `25W` | `15W` |
| precision | `fp16` | `fp16` |
| jetson_clocks | `unknown` | `unknown` |

## Latency / FPS Comparison

| Metric | 25W | 15W | Delta | Delta % |
|---|---:|---:|---:|---:|
| mean_ms | 10.0664 | 10.7991 | 0.7327 | 7.2787% |
| p50_ms | 9.9086 | 10.3388 | 0.4303 | 4.3426% |
| p95_ms | 15.4766 | 15.4387 | -0.038 | -0.2452% |
| p99_ms | 15.5484 | 15.5292 | -0.0192 | -0.1236% |
| fps | 99.3404 | 92.6003 | -6.7401 | -6.7849% |

## Run Depth Comparison

| Field | 25W | 15W |
|---|---|---|
| warmup | `10` | `10` |
| runs | `50` | `50` |
| tegrastats_sample_count | `3` | `10` |
| capture_depth | `short_smoke` | `short_smoke` |

## Tegrastats Comparison

| Metric | 25W | 15W | Delta | Delta % |
|---|---:|---:|---:|---:|
| sample_count | 3 | 10 | 7 | 233.3333% |
| max_temp_c | 40.656 | 42.437 | 1.781 | 4.3807% |
| vdd_in_mw_avg | 4863 | 5707.8 | 844.8 | 17.372% |
| vdd_in_mw_max | 5827 | 7120 | 1293 | 22.1898% |

## Interpretation Notes

- Power mode changes are deployment validation evidence, not a same-run_config latency regression test.
- `capture_depth=short_smoke` should not be described as a sustained thermal benchmark.
- Runtime exports evidence; InferEdgeLab owns comparison policy and deployment decision.
- TensorRT INT8 automatic calibration is outside this report scope.
