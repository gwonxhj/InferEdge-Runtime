#include "inferedge_runtime/cli.hpp"

#include "inferedge_runtime/agent_manifest.hpp"
#include "inferedge_runtime/engine.hpp"
#include "inferedge_runtime/jetson_report.hpp"
#include "inferedge_runtime/lab_worker_request.hpp"
#include "inferedge_runtime/manifest.hpp"
#include "inferedge_runtime/result_writer.hpp"
#include "inferedge_runtime/version.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace inferedge_runtime {
namespace {

std::string require_value(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for option: " + option);
    }

    std::string value = argv[++index];
    if (value.empty() || value.rfind("--", 0) == 0) {
        throw std::invalid_argument("missing value for option: " + option);
    }

    return value;
}

bool is_supported_engine(const std::string& engine) {
    return engine == "onnxruntime" || engine == "ort" || engine == "tensorrt" || engine == "trt";
}

bool is_supported_device(const std::string& device) {
    return device == "cpu" || device == "jetson" || device == "cuda";
}

bool is_supported_worker_response_status(const std::string& status) {
    return status == "completed" || status == "failed";
}

void validate_engine(const std::string& engine) {
    if (!is_supported_engine(engine)) {
        throw std::invalid_argument(
            "unsupported engine: " + engine + " (supported: onnxruntime, ort, tensorrt, trt)");
    }
}

void validate_device(const std::string& device) {
    if (!is_supported_device(device)) {
        throw std::invalid_argument("unsupported device: " + device + " (supported: cpu, jetson, cuda)");
    }
}

int parse_int_with_minimum(const std::string& value, const std::string& option, int min_value) {
    try {
        std::size_t parsed_chars = 0;
        const int parsed_value = std::stoi(value, &parsed_chars);
        if (parsed_chars != value.size() || parsed_value < min_value) {
            throw std::invalid_argument("invalid integer value for option: " + option);
        }
        return parsed_value;
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "invalid integer value for option: " + option + " (" + value + ", minimum: " +
            std::to_string(min_value) + ")");
    }
}

std::string format_shape(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << shape[i];
    }
    stream << ']';
    return stream.str();
}

void print_tensor_metadata_list(const std::vector<TensorMetadata>& tensors) {
    if (tensors.empty()) {
        std::cout << " []\n";
        return;
    }

    std::cout << '\n';
    for (const TensorMetadata& tensor : tensors) {
        std::cout
            << "    - name: " << tensor.name << '\n'
            << "      type: " << tensor.element_type << '\n'
            << "      shape: " << format_shape(tensor.shape) << '\n';
    }
}

}  // namespace

void print_help() {
    std::cout
        << "InferEdgeRuntime CLI\n"
        << "\n"
        << "Usage:\n"
        << "  inferedge-runtime [options]\n"
        << "\n"
        << "Options:\n"
        << "  -h, --help             Show this help message\n"
        << "  --version              Show version information\n"
        << "  --manifest <path>      Optional Forge/build manifest path used for default runtime config\n"
        << "  --forge-manifest <path> Alias for --manifest using Forge manifest.json handoff\n"
        << "  --forge-metadata <path> Optional Forge metadata.json handoff path used for default runtime config\n"
        << "  --agent-manifest <path> Optional Forge agent_manifest.json used for agent task result context\n"
        << "  --agent-task-id <id>   Optional task id recorded inside result.agent (default: task_<agent_id>)\n"
        << "  --agent-queue-wait-ms <n> Optional queue wait in milliseconds recorded in result.agent\n"
        << "  --agent-deadline-missed Mark the agent task as deadline missed in result.agent\n"
        << "  --agent-fallback-used Mark fallback policy use in result.agent\n"
        << "  --agent-execution-status <status> Optional agent execution status override\n"
        << "  --validate-forge-handoff Validate Forge handoff input and exit without execution\n"
        << "  --lab-worker-request <path> Optional Lab worker_request JSON for dry-run validation\n"
        << "  --validate-lab-worker-request Validate Lab worker_request input and exit without execution\n"
        << "  --export-worker-response <path> Write a dry-run Lab worker_response JSON and exit\n"
        << "  --worker-response-status <completed|failed> Dry-run worker response status (default: completed)\n"
        << "  --worker-error-message <message> Optional failed dry-run worker response message\n"
        << "  --model <path>         Path to an input model file\n"
        << "  --input <image_path>   Optional real image input path (requires OpenCV-enabled build)\n"
        << "  --engine <name>        Runtime engine name (supported: onnxruntime, ort, tensorrt, trt; default: onnxruntime)\n"
        << "  --device <name>        Target device name (supported: cpu, jetson, cuda; default: cpu)\n"
        << "  --power-mode <name>    Optional Jetson power mode label recorded in result JSON (example: 15W, 25W)\n"
        << "  --jetson-clocks <state> Optional jetson_clocks state recorded in result JSON (on, off, unknown)\n"
        << "  --tegrastats-log <path> Optional tegrastats log path parsed into Jetson evidence summary\n"
        << "  --report-jetson-evidence Generate a Markdown Jetson evidence report from Runtime JSON and exit\n"
        << "  --result-json <path> Runtime result JSON path used by --report-jetson-evidence\n"
        << "  --compare-power-modes Generate a Markdown 15W/25W-style power-mode comparison report and exit\n"
        << "  --base-result <path> Base Runtime result JSON used by --compare-power-modes\n"
        << "  --candidate-result <path> Candidate Runtime result JSON used by --compare-power-modes\n"
        << "  --report-output <path> Markdown report output path for report-only modes\n"
        << "  --batch <n>            Dummy input batch size, n >= 1 (default: 1)\n"
        << "  --height <n>           Dummy input height, n >= 1 (default: 224)\n"
        << "  --width <n>            Dummy input width, n >= 1 (default: 224)\n"
        << "  --warmup <n>           Number of warmup runs, n >= 0 (default: 5)\n"
        << "  --runs <n>             Number of benchmark runs, n >= 1 (default: 50)\n"
        << "  --timeout-ms <n>       Optional latency timeout observation threshold, n >= 1\n"
        << "  --run-once             Run one inference without benchmark timing\n"
        << "  --output <path|auto>   Output result path or auto-generated results filename (default: results/runtime_result.json)\n";
}

void print_version() {
    std::cout << "inferedge-runtime " << INFEREDGE_RUNTIME_VERSION << '\n';
}

RuntimeConfig parse_args(int argc, char** argv) {
    RuntimeConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];

        if (option == "-h" || option == "--help") {
            config.show_help = true;
        } else if (option == "--version") {
            config.show_version = true;
        } else if (option == "--manifest") {
            config.manifest_path = require_value(argc, argv, i, option);
        } else if (option == "--forge-manifest") {
            config.forge_manifest_path = require_value(argc, argv, i, option);
        } else if (option == "--forge-metadata") {
            config.forge_metadata_path = require_value(argc, argv, i, option);
        } else if (option == "--agent-manifest") {
            config.agent_manifest_path = require_value(argc, argv, i, option);
        } else if (option == "--agent-task-id") {
            config.agent_task_id = require_value(argc, argv, i, option);
        } else if (option == "--agent-queue-wait-ms") {
            config.agent_queue_wait_ms = parse_int_with_minimum(require_value(argc, argv, i, option), option, 0);
        } else if (option == "--agent-deadline-missed") {
            config.agent_deadline_missed = true;
            config.agent_deadline_missed_overridden = true;
        } else if (option == "--agent-fallback-used") {
            config.agent_fallback_used = true;
        } else if (option == "--agent-execution-status") {
            config.agent_execution_status = require_value(argc, argv, i, option);
        } else if (option == "--validate-forge-handoff") {
            config.validate_forge_handoff = true;
        } else if (option == "--lab-worker-request") {
            config.lab_worker_request_path = require_value(argc, argv, i, option);
        } else if (option == "--validate-lab-worker-request") {
            config.validate_lab_worker_request = true;
        } else if (option == "--export-worker-response") {
            config.worker_response_output_path = require_value(argc, argv, i, option);
            config.export_worker_response = true;
        } else if (option == "--worker-response-status") {
            config.worker_response_status = require_value(argc, argv, i, option);
            if (!is_supported_worker_response_status(config.worker_response_status)) {
                throw std::invalid_argument(
                    "unsupported worker response status: " + config.worker_response_status +
                    " (supported: completed, failed)");
            }
        } else if (option == "--worker-error-message") {
            config.worker_error_message = require_value(argc, argv, i, option);
        } else if (option == "--model") {
            config.model_path = require_value(argc, argv, i, option);
            config.model_path_overridden = true;
        } else if (option == "--input") {
            config.input_path = require_value(argc, argv, i, option);
        } else if (option == "--engine") {
            config.engine = require_value(argc, argv, i, option);
            config.engine_overridden = true;
            validate_engine(config.engine);
        } else if (option == "--device") {
            config.device = require_value(argc, argv, i, option);
            config.device_overridden = true;
            validate_device(config.device);
        } else if (option == "--power-mode") {
            config.power_mode = require_value(argc, argv, i, option);
        } else if (option == "--jetson-clocks") {
            config.jetson_clocks = require_value(argc, argv, i, option);
        } else if (option == "--tegrastats-log") {
            config.tegrastats_log_path = require_value(argc, argv, i, option);
        } else if (option == "--report-jetson-evidence") {
            config.report_jetson_evidence = true;
        } else if (option == "--result-json") {
            config.result_json_path = require_value(argc, argv, i, option);
        } else if (option == "--compare-power-modes") {
            config.compare_power_modes = true;
        } else if (option == "--base-result") {
            config.base_result_json_path = require_value(argc, argv, i, option);
        } else if (option == "--candidate-result") {
            config.candidate_result_json_path = require_value(argc, argv, i, option);
        } else if (option == "--report-output") {
            config.report_output_path = require_value(argc, argv, i, option);
        } else if (option == "--batch") {
            config.batch = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
            config.batch_overridden = true;
        } else if (option == "--height") {
            config.height = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
            config.height_overridden = true;
        } else if (option == "--width") {
            config.width = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
            config.width_overridden = true;
        } else if (option == "--warmup") {
            config.warmup = parse_int_with_minimum(require_value(argc, argv, i, option), option, 0);
        } else if (option == "--runs") {
            config.runs = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
        } else if (option == "--timeout-ms") {
            config.timeout_ms = parse_int_with_minimum(require_value(argc, argv, i, option), option, 1);
        } else if (option == "--run-once") {
            config.run_once = true;
        } else if (option == "--output") {
            config.output_path = require_value(argc, argv, i, option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    const int handoff_count =
        (config.manifest_path.empty() ? 0 : 1) +
        (config.forge_manifest_path.empty() ? 0 : 1) +
        (config.forge_metadata_path.empty() ? 0 : 1);
    if (handoff_count > 1) {
        throw std::invalid_argument(
            "use only one Forge handoff option: --manifest, --forge-manifest, or --forge-metadata");
    }
    if (config.validate_forge_handoff && config.validate_lab_worker_request) {
        throw std::invalid_argument(
            "use only one validation mode: --validate-forge-handoff or --validate-lab-worker-request");
    }
    if (config.validate_forge_handoff && config.export_worker_response) {
        throw std::invalid_argument(
            "use only one dry-run mode: --validate-forge-handoff or --export-worker-response");
    }
    if (config.validate_lab_worker_request && config.export_worker_response) {
        throw std::invalid_argument(
            "use only one Lab worker request mode: --validate-lab-worker-request or --export-worker-response");
    }
    if (config.export_worker_response && config.lab_worker_request_path.empty()) {
        throw std::invalid_argument("--lab-worker-request is required with --export-worker-response");
    }
    if (!is_supported_worker_response_status(config.worker_response_status)) {
        throw std::invalid_argument(
            "unsupported worker response status: " + config.worker_response_status +
            " (supported: completed, failed)");
    }

    const int report_mode_count =
        (config.report_jetson_evidence ? 1 : 0) +
        (config.compare_power_modes ? 1 : 0) +
        (config.export_worker_response ? 1 : 0) +
        (config.validate_forge_handoff ? 1 : 0) +
        (config.validate_lab_worker_request ? 1 : 0);
    if (report_mode_count > 1) {
        throw std::invalid_argument(
            "use only one dry-run/report mode: --report-jetson-evidence, --compare-power-modes, "
            "--validate-forge-handoff, --validate-lab-worker-request, or --export-worker-response");
    }
    if (config.report_jetson_evidence && config.result_json_path.empty()) {
        throw std::invalid_argument("--result-json is required with --report-jetson-evidence");
    }
    if (config.compare_power_modes &&
        (config.base_result_json_path.empty() || config.candidate_result_json_path.empty())) {
        throw std::invalid_argument("--base-result and --candidate-result are required with --compare-power-modes");
    }

    ManifestConfig manifest;
    if (!config.forge_metadata_path.empty()) {
        config.manifest_path = config.forge_metadata_path;
        manifest = load_forge_metadata_config(config.forge_metadata_path);
    } else if (!config.forge_manifest_path.empty()) {
        config.manifest_path = config.forge_manifest_path;
        manifest = load_manifest_config(config.forge_manifest_path);
    } else if (!config.manifest_path.empty()) {
        manifest = load_manifest_config(config.manifest_path);
    }

    if (!config.manifest_path.empty()) {
        apply_manifest_defaults(config, manifest);
        config.manifest_applied = true;
    }

    if (!config.agent_manifest_path.empty()) {
        const AgentManifestConfig agent_manifest = load_agent_manifest_config(config.agent_manifest_path);
        apply_agent_manifest_defaults(config, agent_manifest);
        config.agent_manifest_applied = true;
    }

    validate_engine(config.engine);
    validate_device(config.device);

    return config;
}

int run_cli(const RuntimeConfig& config) {
    if (config.report_jetson_evidence) {
        const std::string output_path =
            config.report_output_path.empty() ? "reports/jetson_evidence_summary.md" : config.report_output_path;
        const std::filesystem::path report_path = write_jetson_evidence_markdown_report(
            config.result_json_path,
            config.tegrastats_log_path,
            output_path);
        std::cout
            << "Jetson evidence Markdown report\n"
            << "  result: " << config.result_json_path << '\n'
            << "  tegrastats_log: " << (config.tegrastats_log_path.empty() ? "embedded result JSON" : config.tegrastats_log_path) << '\n'
            << "  output: " << report_path.string() << '\n'
            << "  status: written\n";
        return 0;
    }

    if (config.compare_power_modes) {
        const std::string output_path =
            config.report_output_path.empty() ? "reports/jetson_power_mode_comparison.md" : config.report_output_path;
        const std::filesystem::path report_path = write_power_mode_comparison_markdown_report(
            config.base_result_json_path,
            config.candidate_result_json_path,
            output_path);
        std::cout
            << "Jetson power-mode Markdown comparison\n"
            << "  base: " << config.base_result_json_path << '\n'
            << "  candidate: " << config.candidate_result_json_path << '\n'
            << "  output: " << report_path.string() << '\n'
            << "  status: written\n";
        return 0;
    }

    if (config.export_worker_response) {
        if (config.lab_worker_request_path.empty()) {
            std::cerr << "Lab worker request path is required for --export-worker-response\n";
            return 1;
        }

        const LabWorkerRequestConfig request = load_lab_worker_request_config(config.lab_worker_request_path);
        const std::filesystem::path output_path = write_worker_response_dry_run(
            request,
            config.worker_response_status,
            config.worker_response_output_path,
            config.worker_error_message);

        std::cout
            << "Worker response dry-run export\n"
            << "  request: " << config.lab_worker_request_path << '\n'
            << "  output: " << output_path.string() << '\n'
            << "  job_id: " << request.job_id << '\n'
            << "  status: " << config.worker_response_status << '\n'
            << "  inference: skipped\n";
        return 0;
    }

    if (config.validate_lab_worker_request) {
        if (config.lab_worker_request_path.empty()) {
            std::cerr << "Lab worker request path is required for --validate-lab-worker-request\n";
            return 1;
        }

        const LabWorkerRequestConfig request = load_lab_worker_request_config(config.lab_worker_request_path);
        std::cout
            << "Lab worker request validation\n"
            << "  path: " << config.lab_worker_request_path << '\n'
            << "  job_id: " << request.job_id << '\n'
            << "  workflow: " << request.workflow << '\n'
            << "  model: " << (request.artifact_path.empty() ? request.model_path : request.artifact_path) << '\n'
            << "  source_model: " << (request.model_path.empty() ? "none" : request.model_path) << '\n'
            << "  artifact: " << (request.artifact_path.empty() ? "none" : request.artifact_path) << '\n'
            << "  metadata: " << (request.metadata_path.empty() ? "none" : request.metadata_path) << '\n'
            << "  manifest: " << (request.manifest_path.empty() ? "none" : request.manifest_path) << '\n'
            << "  source_model_sha256: " << (request.source_model_sha256.empty() ? "none" : request.source_model_sha256) << '\n'
            << "  artifact_sha256: " << (request.artifact_sha256.empty() ? "none" : request.artifact_sha256) << '\n'
            << "  artifact_type: " << (request.artifact_type.empty() ? "none" : request.artifact_type) << '\n'
            << "  preset_name: " << (request.preset_name.empty() ? "none" : request.preset_name) << '\n'
            << "  build_id: " << (request.build_id.empty() ? "none" : request.build_id) << '\n'
            << "  engine: " << request.engine << '\n'
            << "  device: " << request.device << '\n'
            << "  precision: " << request.precision << '\n'
            << "  batch: " << request.batch << '\n'
            << "  height: " << request.height << '\n'
            << "  width: " << request.width << '\n'
            << "  warmup: " << request.warmup << '\n'
            << "  runs: " << request.runs << '\n'
            << "  status: ok\n";
        return 0;
    }

    if (config.validate_forge_handoff) {
        if (config.manifest_path.empty()) {
            std::cerr << "Forge handoff path is required for --validate-forge-handoff\n";
            return 1;
        }

        std::cout
            << "Forge handoff validation\n"
            << "  path: " << config.manifest_path << '\n'
            << "  model: " << config.model_path << '\n'
            << "  engine: " << config.engine << '\n'
            << "  device: " << config.device << '\n'
            << "  precision: " << config.manifest_precision << '\n'
            << "  format: " << config.manifest_format << '\n'
            << "  batch: " << config.batch << '\n'
            << "  height: " << config.height << '\n'
            << "  width: " << config.width << '\n'
            << "  preset: " << config.manifest_preset_name << '\n'
            << "  build_id: " << config.manifest_build_id << '\n'
            << "  artifact_sha256: " << config.manifest_artifact_sha256 << '\n'
            << "  source_sha256: " << config.manifest_source_sha256 << '\n'
            << "  status: ok\n";
        return 0;
    }

    if (config.model_path.empty()) {
        std::cerr << "model path is required for benchmark execution\n";
        return 1;
    }

    if (config.has_real_input()) {
#ifndef INFEREDGE_OPENCV_LINKED
        throw std::runtime_error(
            "real image input requires OpenCV-enabled build: reconfigure with -DINFEREDGE_ENABLE_OPENCV=ON");
#endif
    }

    const std::unique_ptr<IInferenceEngine> engine = create_engine(config);
    engine->load_model(config.model_path);
    const EngineMetadata metadata = engine->metadata();
    const ModelMetadata model_metadata = engine->model_metadata();

    std::cout
        << "InferEdgeRuntime benchmark configuration\n"
        << "  manifest: " << (config.manifest_path.empty() ? "none" : config.manifest_path) << '\n'
        << "  manifest_applied: " << (config.manifest_applied ? "true" : "false") << '\n'
        << "  agent_manifest: " << (config.agent_manifest_path.empty() ? "none" : config.agent_manifest_path) << '\n'
        << "  agent_manifest_applied: " << (config.agent_manifest_applied ? "true" : "false") << '\n'
        << "  model:  " << config.model_path << '\n'
        << "  input:  " << (config.input_path.empty() ? "none" : config.input_path) << '\n'
        << "  input_mode: " << config.input_mode() << '\n'
        << "  engine: " << config.engine << '\n'
        << "  device: " << config.device << '\n'
        << "  power_mode: " << config.power_mode << '\n'
        << "  jetson_clocks: " << config.jetson_clocks << '\n'
        << "  tegrastats_log: " << (config.tegrastats_log_path.empty() ? "none" : config.tegrastats_log_path) << '\n'
        << "  batch:  " << config.batch << '\n'
        << "  height: " << config.height << '\n'
        << "  width:  " << config.width << '\n'
        << "  warmup: " << config.warmup << '\n'
        << "  runs:   " << config.runs << '\n'
        << "  timeout_ms: " << (config.timeout_ms > 0 ? std::to_string(config.timeout_ms) : "not_configured") << '\n'
        << "  run_once: " << (config.run_once ? "true" : "false") << '\n'
        << "  output: " << config.output_path << '\n'
        << "\n"
        << "Engine metadata\n"
        << "  name:      " << metadata.name << '\n'
        << "  backend:   " << metadata.backend << '\n'
        << "  device:    " << metadata.device << '\n'
        << "  available: " << (metadata.available ? "true" : "false") << '\n'
        << "  status:    " << metadata.status_message << '\n'
        << "\n"
        << "Model metadata\n"
        << "  inputs:";
    print_tensor_metadata_list(model_metadata.inputs);
    std::cout << "  outputs:";
    print_tensor_metadata_list(model_metadata.outputs);

    BenchmarkResult result;
    if (config.run_once) {
        std::cout << "\nInference\n";
        if (metadata.available) {
            try {
                engine->run_once();
                result.success = true;
                result.status = "success";
                result.message = "one-shot inference completed";
                result.warmup_runs = 0;
                result.timed_runs = 1;
                std::cout
                    << "  status: " << result.status << '\n'
                    << "  runs: 1\n";
            } catch (const std::runtime_error& error) {
                result.success = false;
                result.status = "skipped";
                result.message = error.what();
                result.warmup_runs = 0;
                result.timed_runs = 1;
                std::cout
                    << "  status: " << result.status << '\n'
                    << "  reason: " << result.message << '\n';
            }
        } else {
            result.success = false;
            result.status = "skipped";
            result.message = "backend is not available in this build";
            result.warmup_runs = 0;
            result.timed_runs = 1;
            std::cout
                << "  status: " << result.status << '\n'
                << "  reason: " << result.message << '\n';
        }
    } else if (metadata.available) {
        std::cout << "\nBenchmark\n";
        try {
            result = engine->benchmark(config.warmup, config.runs);
            std::cout
                << "  status: success\n"
                << "  warmup: " << result.warmup_runs << '\n'
                << "  runs: " << result.timed_runs << '\n'
                << std::fixed << std::setprecision(3)
                << "  latency_ms:\n"
                << "    mean: " << result.mean_ms << '\n'
                << "    min: " << result.min_ms << '\n'
                << "    max: " << result.max_ms << '\n'
                << "    std: " << result.std_ms << '\n'
                << "    p50: " << result.p50_ms << '\n'
                << "    p90: " << result.p90_ms << '\n'
                << "    p95: " << result.p95_ms << '\n'
                << "    p99: " << result.p99_ms << '\n'
                << "  fps: " << result.fps << '\n';
        } catch (const std::runtime_error& error) {
            result.success = false;
            result.status = "skipped";
            result.message = error.what();
            result.warmup_runs = config.warmup;
            result.timed_runs = config.runs;
            std::cout
                << "  status: " << result.status << '\n'
                << "  reason: " << result.message << '\n';
        }
    } else {
        std::cout << "\nBenchmark\n";
        result.success = false;
        result.status = "skipped";
        result.message = "backend is not available in this build";
        result.warmup_runs = config.warmup;
        result.timed_runs = config.runs;
        std::cout
            << "  status: " << result.status << '\n'
            << "  reason: " << result.message << '\n';
    }

    const std::filesystem::path output_path = write_result_json(config, metadata, model_metadata, result);

    std::cout
        << "\nResult JSON\n"
        << "  path: " << output_path.string() << '\n'
        << "  latest: results/latest.json\n"
        << "  status: written\n";

    return 0;
}

}  // namespace inferedge_runtime
