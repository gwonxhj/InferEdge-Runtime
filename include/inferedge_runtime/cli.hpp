#pragma once

#include <string>

namespace inferedge_runtime {

struct RuntimeConfig {
    std::string manifest_path;
    std::string forge_metadata_path;
    std::string forge_manifest_path;
    std::string lab_worker_request_path;
    std::string manifest_model_name;
    std::string manifest_source_model_path;
    std::string manifest_precision;
    std::string manifest_format;
    std::string manifest_artifact_sha256;
    std::string manifest_source_sha256;
    std::string manifest_preset_name;
    std::string manifest_build_id;
    std::string agent_manifest_path;
    std::string agent_schema_version;
    std::string agent_id;
    std::string agent_task_id;
    std::string agent_type;
    std::string agent_input_type;
    std::string agent_output_type;
    std::string agent_required_backend;
    std::string agent_device_target;
    std::string agent_precision;
    std::string agent_fallback_policy_mode;
    std::string agent_runtime_artifact_path;
    std::string agent_telemetry_contract_version;
    std::string agent_execution_status;
    std::string model_path;
    std::string input_path;
    std::string result_json_path;
    std::string base_result_json_path;
    std::string candidate_result_json_path;
    std::string report_output_path;
    std::string worker_response_output_path;
    std::string worker_response_status = "completed";
    std::string worker_error_message = "Runtime worker dry-run failure.";
    std::string engine = "onnxruntime";
    std::string device = "cpu";
    std::string power_mode = "unknown";
    std::string jetson_clocks = "unknown";
    std::string tegrastats_log_path;
    int batch = 1;
    int height = 224;
    int width = 224;
    int agent_scheduled_priority = 0;
    int agent_latency_budget_ms = 0;
    int agent_deadline_ms = 0;
    int agent_queue_wait_ms = -1;
    int warmup = 5;
    int runs = 50;
    std::string output_path = "results/runtime_result.json";
    bool run_once = false;
    bool manifest_applied = false;
    bool agent_manifest_applied = false;
    bool agent_deadline_missed = false;
    bool agent_deadline_missed_overridden = false;
    bool agent_fallback_used = false;
    bool model_path_overridden = false;
    bool engine_overridden = false;
    bool device_overridden = false;
    bool batch_overridden = false;
    bool height_overridden = false;
    bool width_overridden = false;
    bool show_help = false;
    bool show_version = false;
    bool validate_forge_handoff = false;
    bool validate_lab_worker_request = false;
    bool export_worker_response = false;
    bool report_jetson_evidence = false;
    bool compare_power_modes = false;

    bool has_real_input() const {
        return !input_path.empty();
    }

    std::string input_mode() const {
        return has_real_input() ? "image" : "dummy";
    }

    std::string input_preprocess() const {
        return has_real_input() ? "opencv_bgr_to_rgb_resize_float32_nchw" : "dummy_zero_float32";
    }
};

void print_help();
void print_version();
RuntimeConfig parse_args(int argc, char** argv);
int run_cli(const RuntimeConfig& config);

}  // namespace inferedge_runtime
