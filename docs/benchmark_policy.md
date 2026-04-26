# InferEdge Runtime Benchmark Policy

InferEdgeRuntime measures end-to-end inference latency from the Runtime point of view.

The `latency_ms` values in Runtime JSON are per-inference wall-clock times. They are intended to represent the cost paid by a simple deployment-oriented runtime path, not pure GPU compute time or isolated kernel execution time.

## Measurement Scope

For TensorRT linked builds, `latency_ms` includes:

- Host to Device memory copy
- TensorRT enqueue / execution launch
- GPU compute
- Device to Host memory copy
- synchronization overhead

This means InferEdgeRuntime latency is broader than a pure GPU kernel metric.

## trtexec Comparison

`trtexec` reports several metrics separately, such as GPU latency, Host latency, enqueue time, and H2D/D2H latency. Those metrics are useful for low-level TensorRT profiling and optimization.

InferEdgeRuntime currently reports one deployment-oriented end-to-end latency value. Because it includes memory transfer and synchronization overhead, InferEdgeRuntime latency is generally expected to be larger than `trtexec` GPU latency.

Therefore, InferEdgeRuntime `mean_ms` and `p99_ms` should not be directly compared to `trtexec` GPU latency as if they measured the same thing.

## Intended Use

InferEdgeRuntime benchmark results are designed to answer:

- How much end-to-end inference cost does this runtime path pay?
- What latency and FPS should downstream tools such as InferEdgeLab compare under the same Runtime measurement policy?
- How do different runtime backends behave when measured through the same application-level interface?

This makes the metric useful for deployment-oriented evaluation, even when it differs from lower-level TensorRT profiler numbers.

## Cross-backend Comparison Policy

When comparing ONNX Runtime and TensorRT results, compare only results with the same `compare_key`.

The `compare_key` groups model and input condition, while `backend_key` identifies the backend/device pair. InferEdgeLab should use `compare_key + backend_key` as the basic comparison unit.

Runtime latency remains backend-specific end-to-end wall-clock latency. TensorRT `trtexec` GPU latency should not be directly compared with Runtime JSON `mean_ms`.

InferEdgeRuntime does not calculate cross-backend comparison results. It only emits compare-ready metadata so InferEdgeLab can own comparison, reporting, and visualization.

## Current Non-goals

The current benchmark policy does not include:

- asynchronous pipeline optimization
- overlapped H2D/D2H transfers
- multi-stream execution
- zero-copy input pipeline
- pure GPU compute-only measurement mode

## Future Improvements

Potential future benchmark modes:

- GPU compute-only latency mode
- optional copy exclusion benchmark mode
- TensorRT profiling integration
- `trtexec`-compatible metric export

These modes should be added as explicit options so existing Runtime JSON semantics remain stable.
