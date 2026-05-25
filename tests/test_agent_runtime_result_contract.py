import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "inferedge-runtime"
FIXTURES = ROOT / "tests" / "fixtures"


class AgentRuntimeResultContractTest(unittest.TestCase):
    def test_agent_manifest_fixture_contains_runtime_task_fields(self):
        manifest = load_json(FIXTURES / "agent_manifest.json")

        self.assertEqual(manifest["schema_version"], "inferedge-agent-manifest-v1")
        self.assertEqual(manifest["agent_id"], "vision_detector")
        self.assertEqual(manifest["agent_type"], "vision")
        self.assertEqual(manifest["priority"], 90)
        self.assertEqual(manifest["latency_budget_ms"], 33)
        self.assertEqual(manifest["required_backend"], "onnxruntime")
        self.assertEqual(manifest["device_target"], "cpu")

    def test_existing_lab_compatible_fixture_keeps_agent_block_optional(self):
        result = load_json(FIXTURES / "lab_compatible_result.json")

        self.assertNotIn("agent", result)
        self.assertNotIn("agent_manifest_path", result.get("extra", {}))

    def test_runtime_output_records_optional_agent_block_when_manifest_is_provided(self):
        skip_without_runtime_binary(self)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "agent_runtime_result.json"
            subprocess.run(
                [
                    str(BINARY),
                    "--agent-manifest",
                    str(FIXTURES / "agent_manifest.json"),
                    "--agent-task-id",
                    "task_camera_frame_0001",
                    "--agent-queue-wait-ms",
                    "7",
                    "--agent-fallback-used",
                    "--timeout-ms",
                    "1",
                    "--warmup",
                    "1",
                    "--runs",
                    "1",
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            result = load_json(output_path)

        agent = result.get("agent")
        if not isinstance(agent, dict):
            raise AssertionError("result.agent must be present when --agent-manifest is provided")

        self.assertEqual(agent["schema_version"], "inferedge-runtime-agent-task-v1")
        self.assertEqual(agent["source_contract"], "inferedge-agent-manifest-v1")
        self.assertEqual(agent["agent_id"], "vision_detector")
        self.assertEqual(agent["task_id"], "task_camera_frame_0001")
        self.assertEqual(agent["agent_type"], "vision")
        self.assertEqual(agent["scheduled_priority"], 90)
        self.assertEqual(agent["latency_budget_ms"], 33)
        self.assertEqual(agent["deadline_ms"], 40)
        self.assertEqual(agent["queue_wait_ms"], 7)
        self.assertEqual(agent["fallback_policy"]["mode"], "drop_stale")
        self.assertTrue(agent["fallback_used"])
        self.assertEqual(agent["runtime_artifact_path"], "models/sample.onnx")
        self.assertEqual(agent["telemetry_contract_version"], "inferedge-agent-telemetry-v1")
        self.assertEqual(agent["telemetry_snapshot"]["power_mode"], "unknown")

        health = result["runtime_health_snapshot"]
        self.assertEqual(health["schema_version"], "inferedge-runtime-health-v1")
        self.assertIn(health["status"], {"ok", "degraded", "error"})
        self.assertIn("health_reason", health)
        self.assertEqual(health["engine_backend"], "onnxruntime")
        self.assertIn("engine_available", health)
        self.assertIn("engine_status_message", health)
        self.assertEqual(health["device"], "cpu")
        self.assertEqual(health["timeout_policy"], "latency_threshold")
        self.assertEqual(health["timeout_budget_ms"], 1)
        self.assertFalse(health["timeout_observed"])
        self.assertEqual(health["latency_budget_ms"], 33)
        self.assertIn("latency_budget_exceeded", health)
        self.assertIn("deadline_missed", health)
        self.assertEqual(health["tegrastats_status"], "not_provided")

        error = result["runtime_error_classification"]
        self.assertEqual(error["schema_version"], "inferedge-runtime-error-v1")
        self.assertIn(error["severity"], {"none", "warning", "error"})
        self.assertIn("retry_hint", error)
        self.assertEqual(error["timeout_budget_ms"], 1)
        self.assertFalse(error["timeout_observed"])
        if result["success"]:
            self.assertEqual(error["status"], "none")
            self.assertEqual(error["category"], "none")
            self.assertFalse(error["retryable"])
        elif result["status"] == "skipped":
            self.assertEqual(health["status"], "degraded")
            self.assertEqual(error["status"], "classified")
            self.assertEqual(error["category"], "runtime_execution_skipped")
            self.assertEqual(error["severity"], "warning")
            self.assertTrue(error["retryable"])
            self.assertEqual(error["retry_hint"], "check_backend_availability")
        else:
            self.assertEqual(error["status"], "classified")
            self.assertNotEqual(error["category"], "none")

        runtime_events = result["runtime_events"]
        self.assertIsInstance(runtime_events, list)
        self.assertEqual([event["event_index"] for event in runtime_events], list(range(len(runtime_events))))
        self.assertTrue(all(event["schema_version"] == "inferedge-runtime-event-v1" for event in runtime_events))
        event_types = {event["type"] for event in runtime_events}
        self.assertIn("runtime_configured", event_types)
        self.assertIn("benchmark_completed", event_types)
        self.assertIn("runtime_error_classified", event_types)
        self.assertIn("runtime_operation_summary_recorded", event_types)
        self.assertIn("agent_context_recorded", event_types)
        benchmark_event = next(event for event in runtime_events if event["type"] == "benchmark_completed")
        self.assertEqual(benchmark_event["latency_budget_ms"], 33)
        self.assertIn("latency_budget_exceeded", benchmark_event)
        self.assertIn("deadline_missed", benchmark_event)
        error_event = next(event for event in runtime_events if event["type"] == "runtime_error_classified")
        self.assertEqual(error_event["timeout_policy"], "latency_threshold")
        self.assertEqual(error_event["timeout_budget_ms"], 1)
        self.assertEqual(error_event["health_reason"], health["health_reason"])
        self.assertIn("retry_hint", error_event)
        self.assertFalse(error_event["timeout_observed"])
        self.assertEqual(error_event["retryable"], error["retryable"])
        operation_event = next(
            event for event in runtime_events if event["type"] == "runtime_operation_summary_recorded"
        )
        self.assertEqual(operation_event["health_reason"], health["health_reason"])
        self.assertIn("recommended_action", operation_event)
        self.assertIsInstance(operation_event["risk_labels"], list)
        self.assertIsInstance(operation_event["evidence_gaps"], list)

        operation_summary = result["runtime_operation_summary"]
        self.assertEqual(
            operation_summary["schema_version"],
            "inferedge-runtime-operation-summary-v1",
        )
        self.assertEqual(operation_summary["observation_scope"], "single_runtime_result")
        self.assertEqual(operation_summary["decision_owner"], "lab")
        self.assertEqual(operation_summary["scheduler_owner"], "orchestrator")
        self.assertFalse(operation_summary["production_cancellation"])
        self.assertEqual(operation_summary["health_status"], health["status"])
        self.assertEqual(operation_summary["health_reason"], health["health_reason"])
        self.assertEqual(operation_summary["retryable"], error["retryable"])
        self.assertIn("recommended_action", operation_summary)
        self.assertIsInstance(operation_summary["risk_labels"], list)
        self.assertIsInstance(operation_summary["evidence_gaps"], list)
        self.assertEqual(operation_summary["timeout_observed"], health["timeout_observed"])
        self.assertEqual(
            operation_summary["latency_budget_exceeded"],
            health["latency_budget_exceeded"],
        )
        self.assertEqual(operation_summary["deadline_missed"], health["deadline_missed"])

        telemetry = result["runtime_telemetry"]
        self.assertEqual(telemetry["schema_version"], "inferedge-runtime-telemetry-v1")
        self.assertEqual(telemetry["evidence_role"], "runtime_telemetry_seed")
        self.assertEqual(telemetry["collection_mode"], "single_result_export")
        self.assertEqual(telemetry["source_result_schema_version"], "inferedge-runtime-result-v1")
        self.assertEqual(telemetry["engine_backend"], "onnxruntime")
        self.assertEqual(telemetry["device"], "cpu")
        self.assertEqual(telemetry["operation"]["timeout_observed"], health["timeout_observed"])
        self.assertEqual(
            telemetry["operation"]["latency_budget_exceeded"],
            health["latency_budget_exceeded"],
        )
        self.assertEqual(telemetry["operation"]["deadline_missed"], health["deadline_missed"])
        self.assertEqual(telemetry["latency"]["mean_ms"], result["mean_ms"])
        self.assertEqual(telemetry["latency"]["p99_ms"], result["p99_ms"])
        self.assertFalse(telemetry["production_monitoring"])
        self.assertIn("queue_depth", telemetry["missing_fields"])
        coverage = telemetry["coverage"]
        self.assertEqual(
            coverage["schema_version"],
            "inferedge-runtime-telemetry-coverage-v1",
        )
        self.assertEqual(coverage["comparability_owner"], "edgeenv")
        self.assertFalse(coverage["missing_telemetry_is_failure"])
        self.assertIn("queue_depth", coverage["expected_fields"])
        self.assertIn("queue_depth", coverage["missing_fields"])
        self.assertIn("telemetry_timestamp", coverage["observed_fields"])
        self.assertEqual(coverage["missing_fields"], telemetry["missing_fields"])
        self.assertEqual(
            coverage["observed_field_count"],
            len(coverage["observed_fields"]),
        )
        self.assertEqual(
            coverage["missing_field_count"],
            len(coverage["missing_fields"]),
        )
        history_seed = telemetry["history_seed"]
        self.assertEqual(
            history_seed["schema_version"],
            "inferedge-runtime-telemetry-history-seed-v1",
        )
        self.assertEqual(history_seed["evidence_role"], "runtime_telemetry_history_seed")
        self.assertEqual(history_seed["registry_owner"], "edgeenv")
        self.assertEqual(history_seed["decision_owner"], "lab")
        self.assertEqual(
            history_seed["source_result_schema_version"],
            telemetry["source_result_schema_version"],
        )
        self.assertEqual(
            history_seed["source_telemetry_schema_version"],
            telemetry["schema_version"],
        )
        self.assertEqual(history_seed["replay_scope"], "single_result_to_history")
        self.assertFalse(history_seed["production_monitoring"])
        self.assertFalse(history_seed["missing_telemetry_is_failure"])
        self.assertTrue(history_seed["replay_ready"])
        self.assertIn("compare_key", history_seed["recommended_registry_key_fields"])
        self.assertIn("latency.mean_ms", history_seed["time_series_fields"])
        self.assertEqual(
            history_seed["source_result"]["compare_key"],
            result["compare_key"],
        )
        self.assertEqual(
            history_seed["source_result"]["backend_key"],
            result["backend_key"],
        )
        self.assertEqual(history_seed["source_result"]["precision"], result["precision"])
        self.assertEqual(history_seed["source_result"]["power_mode"], result["run_config"]["power_mode"])
        self.assertEqual(history_seed["run_config"]["batch"], result["run_config"]["batch"])
        self.assertEqual(history_seed["run_config"]["height"], result["run_config"]["height"])
        self.assertEqual(history_seed["run_config"]["width"], result["run_config"]["width"])
        self.assertEqual(history_seed["run_config"]["warmup"], result["run_config"]["warmup"])
        self.assertEqual(history_seed["run_config"]["runs"], result["run_config"]["runs"])
        self.assertEqual(history_seed["run_config"]["timeout_ms"], result["run_config"]["timeout_ms"])
        self.assertEqual(history_seed["run_config"]["input_mode"], result["runtime_health_snapshot"]["input_mode"])
        self.assertEqual(history_seed["run_config"]["power_mode"], telemetry["power_mode"])
        point = history_seed["points"][0]
        self.assertEqual(point["execution_sequence_id"], telemetry["execution_sequence_id"])
        self.assertEqual(point["telemetry_timestamp"], telemetry["telemetry_timestamp"])
        self.assertEqual(point["mean_ms"], telemetry["latency"]["mean_ms"])
        self.assertEqual(point["p99_ms"], telemetry["latency"]["p99_ms"])
        self.assertEqual(point["timeout_observed"], telemetry["operation"]["timeout_observed"])
        self.assertEqual(point["deadline_missed"], telemetry["operation"]["deadline_missed"])
        self.assertEqual(point["power_mode"], telemetry["power_mode"])
        self.assertEqual(point["telemetry_source"], telemetry["resource"]["telemetry_source"])

        extra = result["extra"]
        self.assertTrue(extra["agent_manifest_recorded"])
        self.assertEqual(extra["agent_id"], "vision_detector")
        self.assertEqual(extra["agent_task_id"], "task_camera_frame_0001")
        self.assertEqual(extra["agent_type"], "vision")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return data


def skip_without_runtime_binary(test_case: unittest.TestCase) -> None:
    if not BINARY.exists():
        test_case.skipTest("inferedge-runtime binary was not built")


if __name__ == "__main__":
    unittest.main()
