# InferEdge Runtime Jetson Evidence Depth Audit

Date: 2026-06-19

This audit maps the current committed Jetson Runtime evidence to the Core 4
six-month roadmap requirement for deeper real-device evidence. It does not
introduce new benchmark measurements or change the Runtime `result.json`
contract.

## Current Evidence Snapshot

| Artifact | Power mode | Runs | Tegrastats samples | Capture depth |
|---|---:|---:|---:|---|
| `tests/fixtures/jetson_tensorrt_25w_result.json` | `25W` | 50 | 3 | `short_smoke` |
| `tests/fixtures/jetson_tensorrt_15w_result.json` | `15W` | 50 | 10 | `short_smoke` |

Both artifacts are TensorRT FP16 Jetson Orin Nano linked-build Runtime results.
They preserve Lab-compatible `compare_key`, `backend_key`, top-level latency
aliases, `run_config`, and `jetson_evidence.tegrastats_summary`.

## Roadmap Coverage

| Requirement | Current evidence | Status |
|---|---|---|
| p95/p99 latency | `p95_ms`, `p99_ms`, `latency_ms.p95`, `latency_ms.p99` in both fixture results | Covered |
| FPS | `fps_value` and `fps` aliases in both fixture results | Covered |
| Power mode context | `run_config.power_mode` and `jetson_evidence.power_mode` record `25W` / `15W` | Covered |
| Thermal behavior starter evidence | `tegrastats_summary.max_temp_c`, `max_temp_name`, sample count | Covered as short-smoke context |
| Memory behavior starter evidence | `ram_used_mb_avg`, `ram_used_mb_max`, `ram_total_mb` | Covered as short-smoke context |
| Power draw starter evidence | `vdd_in_mw_avg`, `vdd_in_mw_max` | Covered as short-smoke context |
| Sustained thermal/power validation | `capture_depth=short_smoke`, runs < 500, samples < 300 | Not yet sustained |
| 15W/25W interpretation | `docs/reports/jetson_power_mode_comparison.md` states different power modes are not same-run_config regression | Covered |

## Interpretation Boundary

- Current committed Jetson results are valid device and contract evidence.
- Current committed Jetson results are not a 5-10 minute sustained thermal or
  power stability claim.
- 15W and 25W results are system evidence for different run configurations; they
  should not be treated as same-condition latency regression.
- Runtime exports execution and device evidence. InferEdgeLab owns comparison
  policy and final deployment decision interpretation.
- TensorRT INT8 automatic calibration remains outside this evidence track.

## Next Evidence Step

To upgrade the current evidence from `short_smoke` to sustained review evidence,
collect a Jetson TensorRT run with either:

- `runs >= 500`, or
- `tegrastats sample_count >= 300`.

The resulting Markdown report should move from `capture_depth=short_smoke` to
`capture_depth=sustained` without changing Runtime `result.json` compatibility.

Jetson hardware is required only for collecting that new sustained evidence. This
audit itself is a documentation and fixture traceability check.
