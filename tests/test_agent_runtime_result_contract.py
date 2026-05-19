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
        self.assertEqual(health["engine_backend"], "onnxruntime")
        self.assertEqual(health["device"], "cpu")
        self.assertEqual(health["timeout_policy"], "latency_threshold")
        self.assertEqual(health["timeout_budget_ms"], 1)
        self.assertFalse(health["timeout_observed"])

        error = result["runtime_error_classification"]
        self.assertEqual(error["schema_version"], "inferedge-runtime-error-v1")
        self.assertFalse(error["timeout_observed"])
        if result["success"]:
            self.assertEqual(error["status"], "none")
            self.assertEqual(error["category"], "none")
        else:
            self.assertEqual(error["status"], "classified")
            self.assertNotEqual(error["category"], "none")

        runtime_events = result["runtime_events"]
        self.assertIsInstance(runtime_events, list)
        event_types = {event["type"] for event in runtime_events}
        self.assertIn("runtime_configured", event_types)
        self.assertIn("benchmark_completed", event_types)
        self.assertIn("runtime_error_classified", event_types)
        self.assertIn("agent_context_recorded", event_types)
        error_event = next(event for event in runtime_events if event["type"] == "runtime_error_classified")
        self.assertEqual(error_event["timeout_policy"], "latency_threshold")
        self.assertFalse(error_event["timeout_observed"])

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
