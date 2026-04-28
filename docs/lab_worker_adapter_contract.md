# Lab Worker Adapter Contract

## Purpose

This document defines how InferEdgeRuntime should interpret an InferEdgeLab `worker_request` as a future Runtime invocation config.

The contract prepares Runtime for Lab-driven worker execution without adding a queue, daemon, database, Redis, Celery, TensorRT expansion, or any change to the existing CLI and Forge handoff parsing behavior.

## Lab Worker Request Input

InferEdgeLab sends a `worker_request` after `/api/analyze` creates a queued job.

Required fields:

| Field | Purpose |
|---|---|
| `job_id` | Stable Lab job identifier. |
| `input_summary` | Original analyze input summary from Lab. |
| `requested_at` | Timestamp for the Lab-to-worker handoff. |
| `model_path` or `artifact_path` | Source model or already-built artifact to execute. |
| `metadata_path` | Optional Forge metadata path. |
| `manifest_path` | Optional Forge manifest path. |
| `options` | Runtime execution options such as backend, target, precision, shape, warmup, and runs. |

Runtime must reject requests that provide neither `model_path` nor `artifact_path`.

## Runtime Invocation Config

The worker adapter should project a Lab request into a Runtime invocation config before any execution happens.

Minimum config fields:

| Field | Source |
|---|---|
| `job_id` | `worker_request.job_id` |
| `model_path` | `worker_request.artifact_path` or `worker_request.model_path` |
| `source_model_path` | `worker_request.model_path` |
| `artifact_path` | `worker_request.artifact_path` |
| `metadata_path` | `worker_request.metadata_path` |
| `manifest_path` | `worker_request.manifest_path` |
| `engine` | `worker_request.options.backend` |
| `device` | `worker_request.options.target` |
| `precision` | `worker_request.options.precision` |
| `batch` | `worker_request.options.batch` |
| `height` | `worker_request.options.height` |
| `width` | `worker_request.options.width` |
| `warmup` | `worker_request.options.warmup` |
| `runs` | `worker_request.options.runs` |

This config is a contract boundary. It is not a new CLI path in this step.

## Forge Metadata/Manifest Interaction

When `metadata_path` or `manifest_path` is provided, Runtime should preserve those paths in config and later in result provenance.

The existing Forge handoff parser remains authoritative for Forge metadata/manifest parsing. The Lab worker adapter contract only defines how Lab request fields point to those inputs.

## Runtime Result Output

Runtime execution should eventually emit a Lab-compatible result JSON with:

- model or model path
- backend/engine
- target/device
- precision
- batch, height, width
- mean, p50, p95, p99 latency
- run config
- timestamp
- provenance fields such as artifact path/hash and source model hash when available

This matches the existing `tests/fixtures/lab_compatible_result.json` schema guard.

## Worker Completed Response Mapping

After execution, Runtime worker output should be wrapped as:

```json
{
  "job_id": "job_...",
  "status": "completed",
  "runtime_result": {},
  "completed_at": "2026-04-28T00:00:00Z"
}
```

`runtime_result` must remain consumable by InferEdgeLab compare/report/deployment decision flows. If Forge metadata is available to the worker, a concise `forge_metadata` summary should be included so Lab and AIGuard can preserve provenance.

## Failure Response Mapping

Failures should use:

```json
{
  "job_id": "job_...",
  "status": "failed",
  "error": {
    "code": "runtime_execution_failed",
    "message": "Runtime execution failed.",
    "stage": "runtime"
  },
  "failed_at": "2026-04-28T00:00:00Z"
}
```

Failed responses must include `error` and must not claim a completed `runtime_result`.

## Non-Goals

This contract does not introduce:

- worker daemon implementation
- queue, database, Redis, or Celery
- file upload handling
- TensorRT execution expansion
- changes to existing Runtime CLI behavior
- changes to existing Forge handoff parsing behavior
- real Forge build execution
- real Runtime inference execution

## Next Steps

- Add a Runtime CLI dry-run option that accepts a Lab worker request and validates the projected invocation config.
- Later, add a worker adapter implementation behind this contract.
- Keep real backend execution behind the existing Runtime execution layer.
