# InferEdge-Runtime

언어: [English](README.md) | 한국어

InferEdge-Runtime은 InferEdge 전체 파이프라인에서 **C++ runtime execution/result export layer** 역할을 맡는 레포입니다.

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

## 현재 범위와 future work

현재는 C++ execution/result export와 contract validation 중심입니다. ONNX Runtime CPU와 Jetson TensorRT smoke evidence가 있지만, production worker daemon이나 persistent queue/DB는 포함하지 않습니다.

Future work:

- production worker daemon integration
- production queue/job runner
- broader TensorRT output post-processing
- production hardening for automated Lab-triggered execution
