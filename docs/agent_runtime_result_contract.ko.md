# Agent Runtime Result Contract 한국어 Quick Guide

언어: [English](agent_runtime_result_contract.md) | 한국어

이 문서는 한국어 빠른 안내서입니다. 대표/canonical 문서는
[Agent Runtime Result Contract](agent_runtime_result_contract.md)입니다.

InferEdge-Runtime은 기존 Lab-compatible `result.json`에 optional `agent` block과
runtime operation evidence를 additive하게 붙일 수 있습니다. 기존 top-level
contract인 `compare_key`, `backend_key`, `run_config`, `latency_ms`,
`jetson_evidence`, `extra`의 shape를 바꾸지 않는 것이 핵심입니다.

## 핵심 역할

Runtime은 execution/result evidence producer입니다. Runtime은 task context,
latency/FPS, backend/device 상태, timeout observation, telemetry seed를 기록하지만
deployment decision, scheduler decision, regression judgement를 소유하지 않습니다.

```text
Forge agent_manifest.json
-> Runtime optional agent result block
-> Runtime health / telemetry / operation evidence
-> Orchestrator operation context
-> EdgeEnv registry / comparability / runtime regression evidence
-> AIGuard deterministic warning evidence
-> Lab-owned deployment decision
```

## Additive evidence

| Block | 의미 | Owner boundary |
|---|---|---|
| `agent` | Forge `agent_manifest.json`에서 온 task metadata | Runtime은 기록만 수행 |
| `runtime_health_snapshot` | backend availability, latency/deadline/timeout observation | production request cancellation 아님 |
| `runtime_error_classification` | skipped/timeout/error classification | AIGuard root cause 확정 아님 |
| `runtime_events` | `event_index` 기반 lifecycle trace | Orchestrator scheduler timeline 대체 아님 |
| `runtime_operation_summary` | Lab/Orchestrator/AIGuard handoff index | `decision_owner: lab`, `scheduler_owner: orchestrator` 유지 |
| `runtime_telemetry.history_seed` | EdgeEnv telemetry history로 넘길 single-result replay point | `registry_owner: edgeenv`, `production_monitoring: false` 유지 |

## 반드시 유지할 경계

- `agent` block은 optional입니다.
- 기존 Runtime result는 `agent` block 없이도 유효해야 합니다.
- Runtime은 production worker daemon이 아닙니다.
- Runtime은 multi-workload scheduler가 아닙니다.
- Runtime은 queue/drop/fallback owner가 아닙니다.
- Runtime은 EdgeEnv registry나 runtime regression owner가 아닙니다.
- Runtime은 production observability platform이 아닙니다.
- Runtime은 Lab-owned deployment decision을 대체하지 않습니다.
- `--timeout-ms`는 observation threshold이며 production request cancellation이 아닙니다.

## Jetson / telemetry 해석

Jetson power mode, `jetson_clocks`, `tegrastats` context는 run-configuration
evidence입니다. 15W와 25W처럼 power mode가 다른 실행은 same-condition
regression으로 해석하지 않습니다. 즉 다른 power mode/clock state를 같은
same-condition regression evidence로 취급하지 않습니다.

Telemetry history seed는 EdgeEnv가 history/replay/comparability evidence로
이어받기 위한 single-result seed입니다. Runtime이 telemetry store나 production
monitoring stream이 되는 것은 아닙니다.

## Jetson 필요 여부

이 문서를 읽거나 링크를 검증하는 작업에는 Jetson 기기가 필요 없습니다. 새로운
TensorRT linked build, live `tegrastats`, device-local replay evidence를 수집할
때만 Jetson 기기가 필요합니다.
