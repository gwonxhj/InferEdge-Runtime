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
```

## 이 레포의 역할

- ONNX Runtime CPU와 TensorRT Jetson 실행 경계를 제공합니다.
- C++ CLI로 inference/benchmark를 실행하고 Lab-compatible result JSON을 export합니다.
- Forge `metadata.json` / `manifest.json` handoff를 읽고 Runtime 실행/provenance로 연결합니다.
- Lab `worker_request` dry-run validation과 worker completed/failed response dry-run export를 제공합니다.
- Runtime은 production worker daemon이 아닙니다. 실제 queue/DB/worker orchestration은 future work입니다.

## 구현된 주요 기능

- C++17 + CMake 기반 Runtime CLI
- ONNX Runtime CPU benchmark/result JSON export
- Jetson TensorRT linked build 실행 evidence
- mean, p50, p95, p99, FPS 등 latency/profiling 결과 export
- Jetson Evidence Track용 power mode / jetson_clocks / tegrastats summary context export
- Lab-compatible result schema fixture/test
- Forge manifest source model identity preservation

Identity preservation:

```text
manifest.source_model.path = models/onnx/yolov8n.onnx
explicit model path = .../model.engine
compare_model_name = yolov8n
compare_key = yolov8n__b1__h640w640__fp32
```

즉 TensorRT engine artifact 경로가 `model.engine`이어도, manifest가 제공하는 원본 모델 identity를 우선해 Lab compare readiness를 유지합니다.

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

## Jetson Evidence Track 준비

Runtime JSON은 Jetson Orin Nano 실측 전후로 validation context를 보존할 수 있습니다.

지원 context:
- `--power-mode`: `15W`, `25W`, `MAXN` 같은 power mode label 기록
- `--jetson-clocks`: `on`, `off`, `unknown` 같은 jetson_clocks 상태 기록
- `--tegrastats-log`: tegrastats log를 읽어 temperature / memory / VDD_IN summary를 `jetson_evidence`에 기록

이 기능은 TensorRT/GPU benchmark 전체 완료나 INT8 calibration 완료를 의미하지 않습니다. 목적은 InferEdgeLab이 p95/p99 latency, FPS, power mode, thermal behavior를 deployment validation evidence로 해석할 수 있게 하는 것입니다.

## 현재 범위와 future work

현재는 C++ execution/result export와 contract validation 중심입니다. ONNX Runtime CPU와 Jetson TensorRT smoke evidence가 있지만, production worker daemon이나 persistent queue/DB는 포함하지 않습니다.

Future work:

- production worker daemon integration
- production queue/job runner
- broader TensorRT output post-processing
- production hardening for automated Lab-triggered execution
