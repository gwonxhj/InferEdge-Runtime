# InferEdge-Runtime

C++ runtime execution and result export layer  
(ONNX Runtime · TensorRT Jetson · latency statistics · Lab-compatible JSON)

언어: [English](README.md) | 한국어

## 요약

- InferEdge validation pipeline의 C++ execution layer입니다.
- ONNX Runtime CPU와 Jetson TensorRT benchmark path를 실행합니다.
- 실제 Runtime 실행에서 latency statistics와 FPS를 기록합니다.
- Lab compare/report/deployment decision 흐름에서 사용할 result JSON을 export합니다.
- built artifact 실행 시 Forge manifest의 source model identity를 보존합니다.
- Forge `agent_manifest.json` context를 optional `agent` result block으로 기록할 수 있습니다.

## InferEdge-Runtime의 차별점

InferEdge-Runtime은 단순한 benchmark wrapper가 아닙니다.

이 레포는 Runtime boundary에서:

- model/artifact input을 검증하거나 실행하고
- latency, FPS, system, provenance context를 기록하며
- Lab이 비교/검토할 수 있는 structured evidence를 export하고
- 실행 evidence와 deployment decision policy를 분리합니다.

InferEdge는 build provenance, C++ Runtime 실행, Lab 분석/deployment decision, optional AIGuard deterministic diagnosis evidence를 연결하는 end-to-end Edge AI inference validation pipeline입니다.

```text
ONNX model
-> InferEdgeForge build/provenance
-> InferEdge-Runtime C++ execution/result export
-> InferEdgeLab compare/API/job/deployment_decision
-> optional InferEdgeAIGuard diagnosis evidence

Experiment hygiene / comparability layer:
InferEdgeEnv -> v0.1.5 v1-complete local-first run evidence registry / comparability checker
```

## 이 레포의 역할

- ONNX Runtime CPU와 TensorRT Jetson 실행 경계를 제공합니다.
- C++ CLI로 inference/benchmark를 실행하고 Lab-compatible result JSON을 export합니다.
- Forge `metadata.json` / `manifest.json` handoff를 읽고 Runtime 실행/provenance로 연결합니다.
- Lab `worker_request` dry-run validation과 worker completed/failed response dry-run export를 제공합니다.
- agent workload metadata를 기존 Runtime result schema를 깨지 않는 optional `agent` block으로 연결합니다.
- Runtime은 production worker daemon이 아닙니다. 실제 queue/DB/worker orchestration은 future work입니다.

## 구현된 주요 기능

- C++17 + CMake 기반 Runtime CLI
- ONNX Runtime CPU benchmark/result JSON export
- Jetson TensorRT linked build 실행 evidence
- mean, p50, p95, p99, FPS 등 latency/profiling 결과 export
- Jetson Evidence Track용 power mode / jetson_clocks / tegrastats summary context export
- Lab-compatible result schema fixture/test
- Forge manifest source model identity preservation
- optional `agent_manifest.json` ingestion과 `agent_id` / `task_id` / priority / latency budget / fallback context export

Identity preservation:

```text
manifest.source_model.path = models/onnx/yolov8n.onnx
explicit model path = .../model.engine
compare_model_name = yolov8n
compare_key = yolov8n__b1__h640w640__fp32
```

즉 TensorRT engine artifact 경로가 `model.engine`이어도, manifest가 제공하는 원본 모델 identity를 우선해 Lab compare readiness를 유지합니다.

## Agent Runtime Result Context

Runtime은 Forge `agent_manifest.json`을 선택적으로 읽어 기존 Lab-compatible result JSON에 `agent` block을 추가할 수 있습니다.

이 기능은 reliable edge agent runtime 방향의 첫 Runtime-side contract입니다. `agent_id`, `task_id`, `agent_type`, priority, latency budget, queue wait, fallback usage, telemetry context를 기록하지만 기존 `result.json`의 top-level compare/report 필드는 변경하지 않습니다.

Runtime result JSON에는 `runtime_health_snapshot`, `runtime_error_classification`, `runtime_events`도 additive evidence로 기록됩니다. `--timeout-ms`는 latency timeout 관측 기준을 남기는 옵션이며, production request cancellation을 의미하지 않습니다.

예시:

```bash
./build/inferedge-runtime \
  --agent-manifest tests/fixtures/agent_manifest.json \
  --agent-task-id task_camera_frame_0001 \
  --agent-queue-wait-ms 7 \
  --agent-fallback-used \
  --timeout-ms 1 \
  --warmup 1 \
  --runs 1 \
  --output results/agent_runtime_result.json
```

계약 기준:

- `agent` block은 optional입니다.
- 기존 Runtime result는 `agent` block 없이도 계속 유효합니다.
- `agent.schema_version`은 `inferedge-runtime-agent-task-v1`입니다.
- Runtime은 task context를 기록할 뿐 scheduling/policy/deployment decision owner가 아닙니다.
- 상세 계약은 [Agent Runtime Result Contract](docs/agent_runtime_result_contract.md)에 정리되어 있습니다.

## 빠른 실행

기본 smoke:

```bash
bash scripts/smoke_default.sh
```

manifest identity test:

```bash
python3 tests/test_manifest_compare_identity.py
```

기본 빌드:

```bash
cmake -S . -B build
cmake --build build -j
./build/inferedge-runtime --help
```

## 다른 InferEdge 레포와의 관계

- **InferEdgeForge:** Runtime이 실행할 artifact와 manifest/metadata provenance를 생성합니다.
- **InferEdgeLab:** Runtime result JSON을 분석해 compare/report/API/job/deployment decision을 생성합니다.
- **InferEdgeAIGuard:** Runtime provenance와 Forge provenance를 비교해 optional diagnosis evidence를 제공합니다.
- **InferEdgeEnv:** `v0.1.5` v1-complete 상태의 experiment hygiene / comparability layer로, benchmark run evidence를 local artifact와 SQLite registry로 고정하고 비교 가능성을 판정합니다.

포트폴리오 경계: InferEdgeLab은 validation / decision layer이고, InferEdgeEnv는 `v0.1.5` v1-complete experiment hygiene / comparability layer입니다. Runtime은 execution/result export를 소유하고, Env는 benchmark evidence가 신뢰 가능하고 비교 가능한 형태인지 관리합니다.

## Jetson Evidence Track

Runtime JSON은 Jetson Orin Nano 실측 validation context를 보존합니다.

지원 context:
- `--power-mode`: `15W`, `25W`, `MAXN` 같은 power mode label 기록
- `--jetson-clocks`: `on`, `off`, `unknown` 같은 jetson_clocks 상태 기록
- `--tegrastats-log`: tegrastats log를 읽어 temperature / memory / VDD_IN summary를 `jetson_evidence`에 기록

현재 기록된 evidence:

| Evidence | Backend | Precision | Power Mode | Mean ms | P95 ms | P99 ms | FPS |
|---|---|---|---|---:|---:|---:|---:|
| TensorRT short smoke | tensorrt__jetson | FP16 | 25W | 10.066401 | 15.476641 | 15.548438 | 99.340373 |
| TensorRT power-mode evidence | tensorrt__jetson | FP16 | 15W | 10.799106 | 15.438690 | 15.529218 | 92.600262 |

15W와 25W는 power mode가 다르므로 같은 run configuration의 latency regression으로 해석하지 않고, deployment validation을 위한 system evidence로 해석합니다.

이 기능은 TensorRT/GPU benchmark 전체 완료나 INT8 calibration 완료를 의미하지 않습니다. 목적은 InferEdgeLab이 p95/p99 latency, FPS, power mode, thermal behavior를 deployment validation evidence로 해석할 수 있게 하는 것입니다.

Jetson evidence는 Markdown 리포트로도 export할 수 있습니다.

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

Markdown 리포트는 Runtime JSON과 tegrastats evidence를 사람이 읽기 쉽게 정리하는 용도입니다. 비교 정책과 deployment decision 해석은 InferEdgeLab이 담당합니다.

제출/검토용으로 고정한 report snapshot:

- [Jetson evidence summary](docs/reports/jetson_evidence_summary.md)
- [Jetson power-mode comparison](docs/reports/jetson_power_mode_comparison.md)

## 현재 범위와 future work

현재는 C++ execution/result export와 contract validation 중심입니다. ONNX Runtime CPU와 Jetson TensorRT smoke evidence가 있지만, production worker daemon이나 persistent queue/DB는 포함하지 않습니다.

Future work:

- production worker daemon integration
- production queue/job runner
- broader TensorRT output post-processing
- production hardening for automated Lab-triggered execution
