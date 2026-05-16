# Agent Runtime Result Contract

InferEdge-Runtime can attach optional agent task context to the existing Lab-compatible Runtime result JSON.

This contract is intentionally additive. Existing Runtime results remain valid without an `agent` block, and Lab-compatible top-level fields such as `compare_key`, `backend_key`, `run_config`, `latency_ms`, `jetson_evidence`, and `extra` must not change shape.

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

## Result JSON Shape

When `--agent-manifest` is not provided, no top-level `agent` block is emitted.

When provided, Runtime appends:

```json
{
  "schema_version": "inferedge-runtime-result-v1",
  "compare_key": "yolov8n__b1__h224w224__fp32",
  "backend_key": "onnxruntime__cpu",
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

## Current Boundary

Runtime records agent task execution context. It does not:

- schedule multiple tasks
- own overload/drop policy
- make deployment decisions
- provide production worker daemon behavior

Those responsibilities remain with Orchestrator and Lab in later phases.
