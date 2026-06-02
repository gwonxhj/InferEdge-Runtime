# Agent Runtime Result Contract

Language: English | [한국어](agent_runtime_result_contract.ko.md)

InferEdge-Runtime can attach optional agent task context to the existing Lab-compatible Runtime result JSON.

This contract is intentionally additive. Existing Runtime results remain valid without an `agent` block, and Lab-compatible top-level fields such as `compare_key`, `backend_key`, `run_config`, `latency_ms`, `jetson_evidence`, and `extra` must not change shape.

Runtime may also append additive operation evidence blocks:

- `runtime_health_snapshot`
- `runtime_error_classification`
- `runtime_events`
- `runtime_operation_summary`

These blocks support downstream runtime operation reporting without turning Runtime into a scheduler or deployment decision owner.

## Scope

The agent result block is the Runtime-side bridge from Forge `agent_manifest.json` to later Orchestrator, AIGuard, and Lab agent workflow analysis.

It records task metadata only. Runtime does not become a scheduler, deployment decision owner, or production worker daemon.

## CLI

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

Relevant options:

| Option | Purpose |
|---|---|
| `--agent-manifest <path>` | Reads Forge `agent_manifest.json` and records task context. |
| `--agent-task-id <id>` | Overrides the generated task id. Default is `task_<agent_id>`. |
| `--agent-queue-wait-ms <n>` | Records queue wait telemetry when supplied by an orchestrator/dev scenario. |
| `--agent-deadline-missed` | Explicitly marks the task as deadline missed. |
| `--agent-fallback-used` | Records fallback policy usage. |
| `--agent-execution-status <status>` | Overrides the agent execution status recorded in `agent.execution_status`. |
| `--timeout-ms <n>` | Records a latency timeout observation threshold in Runtime health/error evidence. This is not production request cancellation. |

Committed observation fixture:

- `tests/fixtures/runtime_timeout_observed_result.json`

This fixture shows the downstream Lab/AIGuard handoff case where Runtime
completed a benchmark, but mean latency crossed the configured observation
threshold. It records:

- `runtime_health_snapshot.timeout_policy: "latency_threshold"`
- `runtime_health_snapshot.timeout_budget_ms: 10`
- `runtime_health_snapshot.timeout_observed: true`
- `runtime_error_classification.category: "runtime_timeout_observed"`
- `runtime_error_classification.retryable: true`
- `runtime_operation_summary.recommended_action: "review_latency_budget_or_degrade"`

Lab treats this as deployment review evidence. Runtime still only records the
observation; it does not cancel production requests or make deployment
decisions.

## Result JSON Shape

When `--agent-manifest` is not provided, no top-level `agent` block is emitted.

When provided, Runtime appends:

```json
{
  "schema_version": "inferedge-runtime-result-v1",
  "compare_key": "yolov8n__b1__h224w224__fp32",
  "backend_key": "onnxruntime__cpu",
  "runtime_health_snapshot": {
    "schema_version": "inferedge-runtime-health-v1",
    "status": "ok",
    "engine_backend": "onnxruntime",
    "engine_available": true,
    "engine_status_message": "",
    "device": "cpu",
    "input_mode": "synthetic",
    "input_preprocess": "synthetic",
    "warmup": 1,
    "runs": 1,
    "run_once": false,
    "success": true,
    "health_reason": "benchmark_completed",
    "latency_mean_ms": 0.0,
    "latency_p95_ms": 0.0,
    "latency_p99_ms": 0.0,
    "fps": 0.0,
    "latency_budget_ms": 33,
    "latency_budget_exceeded": false,
    "deadline_missed": false,
    "power_mode": "unknown",
    "jetson_clocks": "unknown",
    "tegrastats_status": "not_provided",
    "tegrastats_sample_count": 0,
    "thermal_memory_evidence_available": false,
    "timeout_policy": "latency_threshold",
    "timeout_budget_ms": 1,
    "timeout_observed": false
  },
  "runtime_error_classification": {
    "schema_version": "inferedge-runtime-error-v1",
    "status": "none",
    "category": "none",
    "severity": "none",
    "message": "",
    "observed_mean_ms": 0.0,
    "timeout_budget_ms": 1,
    "timeout_observed": false,
    "retryable": false,
    "retry_hint": "none"
  },
  "runtime_events": [
    {
      "schema_version": "inferedge-runtime-event-v1",
      "event_index": 0,
      "type": "runtime_configured",
      "status": "ok",
      "engine_backend": "onnxruntime",
      "engine_available": true,
      "engine_status_message": "",
      "device": "cpu",
      "input_mode": "synthetic",
      "timeout_policy": "latency_threshold"
    },
    {
      "schema_version": "inferedge-runtime-event-v1",
      "event_index": 1,
      "type": "benchmark_completed",
      "status": "success",
      "success": true,
      "warmup": 1,
      "runs": 1,
      "mean_ms": 0.0,
      "p95_ms": 0.0,
      "p99_ms": 0.0,
      "fps": 0.0,
      "latency_budget_ms": 33,
      "latency_budget_exceeded": false,
      "deadline_missed": false
    },
    {
      "schema_version": "inferedge-runtime-event-v1",
      "event_index": 2,
      "type": "runtime_error_classified",
      "status": "none",
      "category": "none",
      "severity": "none",
      "health_reason": "benchmark_completed",
      "timeout_policy": "latency_threshold",
      "timeout_budget_ms": 1,
      "observed_mean_ms": 0.0,
      "timeout_observed": false,
      "retryable": false,
      "retry_hint": "none"
    },
    {
      "schema_version": "inferedge-runtime-event-v1",
      "event_index": 3,
      "type": "runtime_operation_summary_recorded",
      "status": "ok",
      "health_reason": "benchmark_completed",
      "recommended_action": "none",
      "risk_labels": [],
      "evidence_gaps": ["thermal_memory_evidence_missing"]
    }
  ],
  "runtime_operation_summary": {
    "schema_version": "inferedge-runtime-operation-summary-v1",
    "observation_scope": "single_runtime_result",
    "decision_owner": "lab",
    "scheduler_owner": "orchestrator",
    "production_cancellation": false,
    "health_status": "ok",
    "health_reason": "benchmark_completed",
    "error_category": "none",
    "retryable": false,
    "recommended_action": "none",
    "risk_labels": [],
    "evidence_gaps": ["thermal_memory_evidence_missing"],
    "timeout_observed": false,
    "latency_budget_exceeded": false,
    "deadline_missed": false,
    "thermal_memory_evidence_available": false
  },
  "agent": {
    "schema_version": "inferedge-runtime-agent-task-v1",
    "source_contract": "inferedge-agent-manifest-v1",
    "manifest_path": "tests/fixtures/agent_manifest.json",
    "manifest_applied": true,
    "agent_id": "vision_detector",
    "task_id": "task_camera_frame_0001",
    "agent_type": "vision",
    "input_type": "frame",
    "output_type": "detections",
    "scheduled_priority": 90,
    "latency_budget_ms": 33,
    "deadline_ms": 40,
    "deadline_missed": false,
    "queue_wait_ms": 7,
    "execution_status": "skipped",
    "fallback_used": true,
    "fallback_policy": {
      "mode": "drop_stale"
    },
    "runtime_artifact_path": "models/sample.onnx",
    "required_backend": "onnxruntime",
    "device_target": "cpu",
    "precision": "fp32",
    "telemetry_contract_version": "inferedge-agent-telemetry-v1",
    "telemetry_snapshot": {
      "latency_mean_ms": 0.0,
      "latency_p95_ms": 0.0,
      "latency_p99_ms": 0.0,
      "fps": 0.0,
      "power_mode": "unknown",
      "jetson_clocks": "unknown"
    }
  }
}
```

`extra` also records a small discoverability index:

```json
{
  "extra": {
    "agent_manifest_recorded": true,
    "agent_manifest_path": "tests/fixtures/agent_manifest.json",
    "agent_id": "vision_detector",
    "agent_task_id": "task_camera_frame_0001",
    "agent_type": "vision"
  }
}
```

## Compatibility Rules

- `agent` is optional.
- Existing Runtime result consumers must ignore `agent` if they do not support agent workflows yet.
- `agent.schema_version` is `inferedge-runtime-agent-task-v1`.
- `agent.source_contract` is `inferedge-agent-manifest-v1`.
- `agent.deadline_missed` is computed from mean latency and `latency_budget_ms` when possible, unless explicitly overridden by `--agent-deadline-missed`.
- `queue_wait_ms` is `null` unless supplied.
- `execution_status` defaults to the Runtime benchmark status unless overridden.
- `runtime_health_snapshot`, `runtime_error_classification`, and `runtime_events` are additive and safe for existing consumers to ignore.
- `runtime_health_snapshot` includes backend availability, latency-budget/deadline observation, timeout observation, and tegrastats evidence availability when those values are known.
- `runtime_health_snapshot.health_reason` gives a compact reason such as `benchmark_completed`, `backend_unavailable_or_not_enabled`, `runtime_execution_skipped`, or `timeout_threshold_exceeded`.
- `runtime_events` uses additive `inferedge-runtime-event-v1` entries with sequential `event_index` values so Lab/Orchestrator reports can show a compact lifecycle trace.
- `runtime_operation_summary` is an additive handoff index for Lab/Orchestrator/AIGuard. It repeats the health reason, retryability, risk labels, evidence gaps, and a conservative `recommended_action` without making the deployment decision itself.
- `runtime_operation_summary.decision_owner` must remain `lab`, and `scheduler_owner` must remain `orchestrator`.
- `runtime_operation_summary.production_cancellation` is always `false`; Runtime records observations only.
- `runtime_telemetry.history_seed` is an additive `inferedge-runtime-telemetry-history-seed-v1` block for EdgeEnv telemetry history/replay. It keeps `registry_owner: edgeenv`, `decision_owner: lab`, `production_monitoring: false`, `replay_scope: single_result_to_history`, and one single-result telemetry point so downstream tools can accumulate history without Runtime becoming a telemetry store. The replay point mirrors Runtime telemetry timestamp, sequence id, latency values, operation flags, power mode, and telemetry source. The seed also carries a compact `run_config` snapshot for replay/comparability context.
- Runtime does not claim production request cancellation. `--timeout-ms` is an observation threshold: if a successful benchmark mean latency exceeds the configured threshold, Runtime records `timeout_observed: true`, `runtime_error_classification.category: runtime_timeout_observed`, and `retryable: true` for downstream reliability reporting.
- If execution is skipped because Runtime cannot complete the configured benchmark, Runtime records `runtime_error_classification.category: runtime_execution_skipped`, `severity: warning`, `retryable: true`, and `retry_hint: check_backend_availability`. This is failure-handling evidence for Lab/Orchestrator reporting, not a production worker retry loop.
- Without `--timeout-ms`, results record `timeout_policy: not_configured`, `timeout_budget_ms: null`, and `timeout_observed: false`.

## Current Boundary

Runtime records agent task execution context. It does not:

- schedule multiple tasks
- own overload/drop policy
- make deployment decisions
- provide production worker daemon behavior

Those responsibilities remain with Orchestrator and Lab in later phases.
