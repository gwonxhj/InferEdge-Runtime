#include "inferedge_runtime/result_writer.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace inferedge_runtime {
namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream stream;
    for (const char ch : value) {
        switch (ch) {
            case '"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                stream << ch;
                break;
        }
    }
    return stream.str();
}

std::string json_string(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string current_utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_time{};
#if defined(_WIN32)
    gmtime_s(&utc_time, &time);
#else
    gmtime_r(&time, &utc_time);
#endif

    std::ostringstream stream;
    stream << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::string timestamp_for_filename(const std::string& timestamp) {
    std::string value;
    value.reserve(timestamp.size());
    for (const char ch : timestamp) {
        if (ch != '-' && ch != ':') {
            value.push_back(ch);
        }
    }
    return value;
}

std::string sanitize_filename_component(const std::string& value) {
    std::string sanitized;
    sanitized.reserve(value.size());
    for (const unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
            sanitized.push_back(static_cast<char>(ch));
        } else {
            sanitized.push_back('_');
        }
    }
    return sanitized.empty() ? "unknown" : sanitized;
}

std::filesystem::path resolve_output_path(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const std::string& timestamp) {
    if (config.output_path != "auto") {
        return std::filesystem::path(config.output_path);
    }

    const std::string model_stem = sanitize_filename_component(
        std::filesystem::path(config.model_path).stem().string());
    const std::string engine_name = sanitize_filename_component(engine_metadata.name);
    const std::string device_name = sanitize_filename_component(engine_metadata.device);
    const std::string filename =
        model_stem + "__" + engine_name + "__" + device_name + "__fp32__b" +
        std::to_string(config.batch) + "__h" + std::to_string(config.height) + "w" +
        std::to_string(config.width) + "__" + timestamp_for_filename(timestamp) + ".json";

    return std::filesystem::path("results") / filename;
}

std::string stem_from_path_like_value(const std::string& value) {
    return std::filesystem::path(value).stem().string();
}

std::string compare_model_name(const RuntimeConfig& config) {
    if (!config.manifest_model_name.empty()) {
        return sanitize_filename_component(stem_from_path_like_value(config.manifest_model_name));
    }

    return sanitize_filename_component(std::filesystem::path(config.model_path).stem().string());
}

std::string compare_model_source(const RuntimeConfig& config) {
    return config.manifest_model_name.empty() ? "model_path" : "manifest_source_model";
}

std::string make_compare_key(const RuntimeConfig& config) {
    const std::string model_stem = compare_model_name(config);
    const std::string precision = config.manifest_precision.empty() ? "fp32" : config.manifest_precision;
    return model_stem + "__b" + std::to_string(config.batch) + "__h" +
           std::to_string(config.height) + "w" + std::to_string(config.width) + "__" +
           sanitize_filename_component(precision);
}

std::string make_backend_key(const EngineMetadata& engine_metadata, const RuntimeConfig& config) {
    return sanitize_filename_component(engine_metadata.backend) + "__" +
           sanitize_filename_component(config.device);
}

std::string system_os_name() {
#if defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "linux";
#elif defined(_WIN32)
    return "windows";
#else
    return "unknown";
#endif
}

std::string compiler_name() {
#if defined(__apple_build_version__)
    return "AppleClang";
#elif defined(__clang__)
    return "Clang";
#elif defined(__GNUC__)
    return "GCC";
#elif defined(_MSC_VER)
    return "MSVC";
#else
    return "unknown";
#endif
}

struct TegrastatsSummary {
    std::string status = "not_provided";
    int sample_count = 0;
    double ram_used_mb_avg = 0.0;
    double ram_used_mb_max = 0.0;
    double ram_total_mb = 0.0;
    double max_temp_c = 0.0;
    std::string max_temp_name;
    double vdd_in_mw_avg = 0.0;
    double vdd_in_mw_max = 0.0;
};

TegrastatsSummary parse_tegrastats_log(const std::string& path) {
    TegrastatsSummary summary;
    if (path.empty()) {
        return summary;
    }

    std::ifstream input(path);
    if (!input) {
        summary.status = "unavailable";
        return summary;
    }

    summary.status = "parsed";
    const std::regex ram_pattern(R"(RAM\s+([0-9]+)/([0-9]+)MB)");
    const std::regex temp_pattern(R"(([A-Za-z0-9_]+)@([0-9]+(?:\.[0-9]+)?)C)");
    const std::regex vdd_in_pattern(R"(VDD_IN\s+([0-9]+)mW(/([0-9]+)mW)?)");

    std::string line;
    double ram_used_sum = 0.0;
    int ram_samples = 0;
    double vdd_in_sum = 0.0;
    int vdd_in_samples = 0;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        summary.sample_count += 1;

        std::smatch match;
        if (std::regex_search(line, match, ram_pattern)) {
            const double used = std::stod(match[1].str());
            const double total = std::stod(match[2].str());
            ram_used_sum += used;
            ram_samples += 1;
            summary.ram_used_mb_max = std::max(summary.ram_used_mb_max, used);
            summary.ram_total_mb = total;
        }

        for (auto it = std::sregex_iterator(line.begin(), line.end(), temp_pattern);
             it != std::sregex_iterator();
             ++it) {
            const std::smatch temp_match = *it;
            const double temp_c = std::stod(temp_match[2].str());
            if (temp_c > summary.max_temp_c) {
                summary.max_temp_c = temp_c;
                summary.max_temp_name = temp_match[1].str();
            }
        }

        if (std::regex_search(line, match, vdd_in_pattern)) {
            const double vdd_in = std::stod(match[1].str());
            vdd_in_sum += vdd_in;
            vdd_in_samples += 1;
            summary.vdd_in_mw_max = std::max(summary.vdd_in_mw_max, vdd_in);
        }
    }

    if (summary.sample_count == 0) {
        summary.status = "no_samples";
    }
    if (ram_samples > 0) {
        summary.ram_used_mb_avg = ram_used_sum / static_cast<double>(ram_samples);
    }
    if (vdd_in_samples > 0) {
        summary.vdd_in_mw_avg = vdd_in_sum / static_cast<double>(vdd_in_samples);
    }

    return summary;
}

void write_tegrastats_summary_json(std::ostream& output, const TegrastatsSummary& summary, int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    output
        << "{\n"
        << indent << "  \"status\": " << json_string(summary.status) << ",\n"
        << indent << "  \"sample_count\": " << summary.sample_count << ",\n"
        << indent << "  \"ram_used_mb_avg\": " << summary.ram_used_mb_avg << ",\n"
        << indent << "  \"ram_used_mb_max\": " << summary.ram_used_mb_max << ",\n"
        << indent << "  \"ram_total_mb\": " << summary.ram_total_mb << ",\n"
        << indent << "  \"max_temp_c\": " << summary.max_temp_c << ",\n"
        << indent << "  \"max_temp_name\": " << json_string(summary.max_temp_name) << ",\n"
        << indent << "  \"vdd_in_mw_avg\": " << summary.vdd_in_mw_avg << ",\n"
        << indent << "  \"vdd_in_mw_max\": " << summary.vdd_in_mw_max << "\n"
        << indent << "}";
}

void write_string_array_json(std::ostream& output, const std::vector<std::string>& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            output << ", ";
        }
        output << json_string(values[i]);
    }
    output << ']';
}

bool contains_string(const std::vector<std::string>& values, const std::string& target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}

bool should_mark_deadline_missed(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    if (config.agent_deadline_missed_overridden) {
        return config.agent_deadline_missed;
    }
    return benchmark_result.success &&
           config.agent_latency_budget_ms > 0 &&
           benchmark_result.mean_ms > static_cast<double>(config.agent_latency_budget_ms);
}

std::string agent_execution_status(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    if (!config.agent_execution_status.empty()) {
        return config.agent_execution_status;
    }
    return benchmark_result.status.empty() ? "unknown" : benchmark_result.status;
}

void write_agent_task_json(
    std::ostream& output,
    const RuntimeConfig& config,
    const BenchmarkResult& benchmark_result,
    int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const bool deadline_missed = should_mark_deadline_missed(config, benchmark_result);

    output
        << "{\n"
        << indent << "  \"schema_version\": \"inferedge-runtime-agent-task-v1\",\n"
        << indent << "  \"source_contract\": \"inferedge-agent-manifest-v1\",\n"
        << indent << "  \"manifest_path\": " << json_string(config.agent_manifest_path) << ",\n"
        << indent << "  \"manifest_applied\": " << (config.agent_manifest_applied ? "true" : "false") << ",\n"
        << indent << "  \"agent_id\": " << json_string(config.agent_id) << ",\n"
        << indent << "  \"task_id\": " << json_string(config.agent_task_id) << ",\n"
        << indent << "  \"agent_type\": " << json_string(config.agent_type) << ",\n"
        << indent << "  \"input_type\": " << json_string(config.agent_input_type) << ",\n"
        << indent << "  \"output_type\": " << json_string(config.agent_output_type) << ",\n"
        << indent << "  \"scheduled_priority\": " << config.agent_scheduled_priority << ",\n"
        << indent << "  \"latency_budget_ms\": " << config.agent_latency_budget_ms << ",\n"
        << indent << "  \"deadline_ms\": " << config.agent_deadline_ms << ",\n"
        << indent << "  \"deadline_missed\": " << (deadline_missed ? "true" : "false") << ",\n"
        << indent << "  \"queue_wait_ms\": ";
    if (config.agent_queue_wait_ms < 0) {
        output << "null";
    } else {
        output << config.agent_queue_wait_ms;
    }
    output
        << ",\n"
        << indent << "  \"execution_status\": " << json_string(agent_execution_status(config, benchmark_result)) << ",\n"
        << indent << "  \"fallback_used\": " << (config.agent_fallback_used ? "true" : "false") << ",\n"
        << indent << "  \"fallback_policy\": {\n"
        << indent << "    \"mode\": " << json_string(config.agent_fallback_policy_mode) << "\n"
        << indent << "  },\n"
        << indent << "  \"runtime_artifact_path\": " << json_string(config.agent_runtime_artifact_path) << ",\n"
        << indent << "  \"required_backend\": " << json_string(config.agent_required_backend) << ",\n"
        << indent << "  \"device_target\": " << json_string(config.agent_device_target) << ",\n"
        << indent << "  \"precision\": " << json_string(config.agent_precision) << ",\n"
        << indent << "  \"telemetry_contract_version\": " << json_string(config.agent_telemetry_contract_version) << ",\n"
        << indent << "  \"telemetry_snapshot\": {\n"
        << indent << "    \"latency_mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "    \"latency_p95_ms\": " << benchmark_result.p95_ms << ",\n"
        << indent << "    \"latency_p99_ms\": " << benchmark_result.p99_ms << ",\n"
        << indent << "    \"fps\": " << benchmark_result.fps << ",\n"
        << indent << "    \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << indent << "    \"jetson_clocks\": " << json_string(config.jetson_clocks) << "\n"
        << indent << "  }\n"
        << indent << "}";
}

bool timeout_observed(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    return config.timeout_ms > 0 &&
           benchmark_result.success &&
           benchmark_result.mean_ms > static_cast<double>(config.timeout_ms);
}

std::string runtime_health_status(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    if (timeout_observed(config, benchmark_result)) {
        return "degraded";
    }
    if (benchmark_result.success) {
        return "ok";
    }
    if (benchmark_result.status == "skipped") {
        return "degraded";
    }
    return "error";
}

std::string runtime_error_category(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    if (timeout_observed(config, benchmark_result)) {
        return "runtime_timeout_observed";
    }
    if (benchmark_result.success) {
        return "none";
    }
    if (benchmark_result.status == "skipped") {
        return "runtime_execution_skipped";
    }
    if (!benchmark_result.status.empty()) {
        return "runtime_" + sanitize_filename_component(benchmark_result.status);
    }
    return "runtime_error";
}

std::string runtime_error_severity(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    if (timeout_observed(config, benchmark_result)) {
        return "warning";
    }
    if (benchmark_result.success) {
        return "none";
    }
    if (benchmark_result.status == "skipped") {
        return "warning";
    }
    return "error";
}

std::string runtime_retry_hint(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    if (timeout_observed(config, benchmark_result)) {
        return "retry_or_degrade";
    }
    if (benchmark_result.success) {
        return "none";
    }
    if (benchmark_result.status == "skipped") {
        return "check_backend_availability";
    }
    return "check_runtime_error";
}

bool runtime_retryable(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    return timeout_observed(config, benchmark_result) || benchmark_result.status == "skipped";
}

bool latency_budget_exceeded(const RuntimeConfig& config, const BenchmarkResult& benchmark_result) {
    return benchmark_result.success &&
           config.agent_latency_budget_ms > 0 &&
           benchmark_result.mean_ms > static_cast<double>(config.agent_latency_budget_ms);
}

std::string runtime_health_reason(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const BenchmarkResult& benchmark_result) {
    if (timeout_observed(config, benchmark_result)) {
        return "timeout_threshold_exceeded";
    }
    if (benchmark_result.success) {
        return "benchmark_completed";
    }
    if (benchmark_result.status == "skipped" && !engine_metadata.available) {
        return "backend_unavailable_or_not_enabled";
    }
    if (benchmark_result.status == "skipped") {
        return "runtime_execution_skipped";
    }
    return "runtime_execution_error";
}

std::vector<std::string> runtime_operation_risk_labels(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const BenchmarkResult& benchmark_result) {
    std::vector<std::string> labels;
    if (!benchmark_result.success) {
        if (benchmark_result.status == "skipped") {
            labels.push_back("runtime_execution_skipped");
        } else {
            labels.push_back("runtime_execution_error");
        }
        if (!engine_metadata.available) {
            labels.push_back("backend_unavailable");
        }
    }
    if (timeout_observed(config, benchmark_result)) {
        labels.push_back("runtime_timeout_observed");
    }
    if (latency_budget_exceeded(config, benchmark_result)) {
        labels.push_back("latency_budget_exceeded");
    }
    if (should_mark_deadline_missed(config, benchmark_result)) {
        labels.push_back("deadline_missed");
    }
    return labels;
}

std::vector<std::string> runtime_operation_evidence_gaps(
    const RuntimeConfig& config,
    const TegrastatsSummary& tegrastats_summary) {
    std::vector<std::string> gaps;
    if (config.timeout_ms <= 0) {
        gaps.push_back("timeout_policy_not_configured");
    }
    if (tegrastats_summary.status != "parsed") {
        gaps.push_back("thermal_memory_evidence_missing");
    }
    return gaps;
}

std::vector<std::string> runtime_telemetry_missing_fields(
    const TegrastatsSummary& tegrastats_summary) {
    std::vector<std::string> fields;
    if (tegrastats_summary.status != "parsed") {
        fields.push_back("ram_used_mb");
        fields.push_back("thermal_max_temperature_c");
        fields.push_back("power_vdd_in_mw");
    }
    fields.push_back("gpu_memory_used_mb");
    fields.push_back("cpu_temperature_c");
    fields.push_back("gpu_temperature_c");
    fields.push_back("throttling_detected");
    fields.push_back("queue_depth");
    fields.push_back("runtime_uptime_sec");
    return fields;
}

std::vector<std::string> runtime_telemetry_expected_fields() {
    return {
        "gpu_temperature_c",
        "cpu_temperature_c",
        "thermal_max_temperature_c",
        "gpu_memory_used_mb",
        "ram_used_mb",
        "power_mode",
        "throttling_detected",
        "queue_depth",
        "inference_interval_ms",
        "runtime_uptime_sec",
        "rolling_latency_mean_ms",
        "rolling_latency_std_ms",
        "telemetry_timestamp",
        "execution_sequence_id",
    };
}

std::vector<std::string> runtime_telemetry_observed_fields(
    const TegrastatsSummary& tegrastats_summary) {
    const std::vector<std::string> missing_fields =
        runtime_telemetry_missing_fields(tegrastats_summary);
    std::vector<std::string> observed_fields;
    for (const std::string& field : runtime_telemetry_expected_fields()) {
        if (!contains_string(missing_fields, field)) {
            observed_fields.push_back(field);
        }
    }
    return observed_fields;
}

void write_runtime_telemetry_coverage_json(
    std::ostream& output,
    const TegrastatsSummary& tegrastats_summary,
    int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const std::vector<std::string> expected_fields = runtime_telemetry_expected_fields();
    const std::vector<std::string> observed_fields =
        runtime_telemetry_observed_fields(tegrastats_summary);
    const std::vector<std::string> missing_fields =
        runtime_telemetry_missing_fields(tegrastats_summary);
    const double coverage_ratio = expected_fields.empty()
        ? 0.0
        : static_cast<double>(observed_fields.size()) /
              static_cast<double>(expected_fields.size());

    output
        << "{\n"
        << indent << "  \"schema_version\": \"inferedge-runtime-telemetry-coverage-v1\",\n"
        << indent << "  \"coverage_scope\": \"single_result_export\",\n"
        << indent << "  \"comparability_owner\": \"edgeenv\",\n"
        << indent << "  \"missing_telemetry_is_failure\": false,\n"
        << indent << "  \"expected_fields\": ";
    write_string_array_json(output, expected_fields);
    output
        << ",\n"
        << indent << "  \"observed_fields\": ";
    write_string_array_json(output, observed_fields);
    output
        << ",\n"
        << indent << "  \"missing_fields\": ";
    write_string_array_json(output, missing_fields);
    output
        << ",\n"
        << indent << "  \"observed_field_count\": " << observed_fields.size() << ",\n"
        << indent << "  \"missing_field_count\": " << missing_fields.size() << ",\n"
        << indent << "  \"coverage_ratio\": " << coverage_ratio << "\n"
        << indent << "}";
}

void write_runtime_telemetry_history_seed_json(
    std::ostream& output,
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const BenchmarkResult& benchmark_result,
    const TegrastatsSummary& tegrastats_summary,
    const std::string& timestamp,
    int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const bool has_tegrastats = tegrastats_summary.status == "parsed";
    const std::string precision = config.manifest_precision.empty() ? "fp32" : config.manifest_precision;
    output
        << "{\n"
        << indent << "  \"schema_version\": \"inferedge-runtime-telemetry-history-seed-v1\",\n"
        << indent << "  \"evidence_role\": \"runtime_telemetry_history_seed\",\n"
        << indent << "  \"registry_owner\": \"edgeenv\",\n"
        << indent << "  \"decision_owner\": \"lab\",\n"
        << indent << "  \"source_result_schema_version\": \"inferedge-runtime-result-v1\",\n"
        << indent << "  \"source_telemetry_schema_version\": \"inferedge-runtime-telemetry-v1\",\n"
        << indent << "  \"replay_scope\": \"single_result_to_history\",\n"
        << indent << "  \"replay_ready\": true,\n"
        << indent << "  \"production_monitoring\": false,\n"
        << indent << "  \"missing_telemetry_is_failure\": false,\n"
        << indent << "  \"source_result\": {\n"
        << indent << "    \"compare_key\": " << json_string(make_compare_key(config)) << ",\n"
        << indent << "    \"backend_key\": " << json_string(make_backend_key(engine_metadata, config)) << ",\n"
        << indent << "    \"engine_backend\": " << json_string(engine_metadata.backend) << ",\n"
        << indent << "    \"device\": " << json_string(config.device) << ",\n"
        << indent << "    \"precision\": " << json_string(precision) << ",\n"
        << indent << "    \"power_mode\": " << json_string(config.power_mode) << "\n"
        << indent << "  },\n"
        << indent << "  \"run_config\": {\n"
        << indent << "    \"batch\": " << config.batch << ",\n"
        << indent << "    \"height\": " << config.height << ",\n"
        << indent << "    \"width\": " << config.width << ",\n"
        << indent << "    \"warmup\": " << config.warmup << ",\n"
        << indent << "    \"runs\": " << config.runs << ",\n"
        << indent << "    \"timeout_ms\": ";
    if (config.timeout_ms > 0) {
        output << config.timeout_ms;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "    \"input_mode\": " << json_string(config.input_mode()) << ",\n"
        << indent << "    \"input_preprocess\": " << json_string(config.input_preprocess()) << ",\n"
        << indent << "    \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << indent << "    \"jetson_clocks\": " << json_string(config.jetson_clocks) << "\n"
        << indent << "  },\n"
        << indent << "  \"recommended_registry_key_fields\": ";
    write_string_array_json(output, {
        "compare_key",
        "backend_key",
        "device",
        "precision",
        "power_mode",
        "run_config",
    });
    output
        << ",\n"
        << indent << "  \"time_series_fields\": ";
    write_string_array_json(output, {
        "telemetry_timestamp",
        "execution_sequence_id",
        "latency.mean_ms",
        "latency.p95_ms",
        "latency.p99_ms",
        "latency.fps",
        "latency.inference_interval_ms",
        "latency.rolling_latency_mean_ms",
        "latency.rolling_latency_std_ms",
        "resource.ram_used_mb",
        "resource.max_temperature_c",
        "resource.vdd_in_mw_avg",
        "operation.queue_depth",
        "operation.runtime_uptime_sec",
        "operation.timeout_observed",
        "operation.latency_budget_exceeded",
        "operation.deadline_missed",
    });
    output
        << ",\n"
        << indent << "  \"points\": [\n"
        << indent << "    {\n"
        << indent << "      \"execution_sequence_id\": 0,\n"
        << indent << "      \"telemetry_timestamp\": " << json_string(timestamp) << ",\n"
        << indent << "      \"mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "      \"p95_ms\": " << benchmark_result.p95_ms << ",\n"
        << indent << "      \"p99_ms\": " << benchmark_result.p99_ms << ",\n"
        << indent << "      \"fps\": " << benchmark_result.fps << ",\n"
        << indent << "      \"inference_interval_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "      \"rolling_latency_mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "      \"rolling_latency_std_ms\": " << benchmark_result.std_ms << ",\n"
        << indent << "      \"ram_used_mb\": ";
    if (has_tegrastats) {
        output << tegrastats_summary.ram_used_mb_max;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "      \"max_temperature_c\": ";
    if (has_tegrastats) {
        output << tegrastats_summary.max_temp_c;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "      \"vdd_in_mw_avg\": ";
    if (has_tegrastats) {
        output << tegrastats_summary.vdd_in_mw_avg;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "      \"queue_depth\": null,\n"
        << indent << "      \"runtime_uptime_sec\": null,\n"
        << indent << "      \"timeout_observed\": "
        << (timeout_observed(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "      \"latency_budget_exceeded\": "
        << (latency_budget_exceeded(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "      \"deadline_missed\": "
        << (should_mark_deadline_missed(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "      \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << indent << "      \"telemetry_source\": "
        << json_string(has_tegrastats ? "tegrastats" : "not_available") << ",\n"
        << indent << "      \"tegrastats_status\": " << json_string(tegrastats_summary.status) << "\n"
        << indent << "    }\n"
        << indent << "  ]\n"
        << indent << "}";
}

std::string runtime_operation_recommended_action(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const BenchmarkResult& benchmark_result) {
    if (timeout_observed(config, benchmark_result) || latency_budget_exceeded(config, benchmark_result)) {
        return "review_latency_budget_or_degrade";
    }
    if (benchmark_result.success) {
        return "none";
    }
    if (benchmark_result.status == "skipped" && !engine_metadata.available) {
        return "check_backend_availability";
    }
    if (benchmark_result.status == "skipped") {
        return "review_runtime_configuration";
    }
    return "inspect_runtime_error";
}

void write_runtime_health_snapshot_json(
    std::ostream& output,
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const BenchmarkResult& benchmark_result,
    const TegrastatsSummary& tegrastats_summary,
    int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const bool observed_timeout = timeout_observed(config, benchmark_result);
    const bool exceeded_latency_budget = latency_budget_exceeded(config, benchmark_result);
    output
        << "{\n"
        << indent << "  \"schema_version\": \"inferedge-runtime-health-v1\",\n"
        << indent << "  \"status\": " << json_string(runtime_health_status(config, benchmark_result)) << ",\n"
        << indent << "  \"engine_backend\": " << json_string(engine_metadata.backend) << ",\n"
        << indent << "  \"engine_available\": " << (engine_metadata.available ? "true" : "false") << ",\n"
        << indent << "  \"engine_status_message\": " << json_string(engine_metadata.status_message) << ",\n"
        << indent << "  \"device\": " << json_string(config.device) << ",\n"
        << indent << "  \"input_mode\": " << json_string(config.input_mode()) << ",\n"
        << indent << "  \"input_preprocess\": " << json_string(config.input_preprocess()) << ",\n"
        << indent << "  \"warmup\": " << config.warmup << ",\n"
        << indent << "  \"runs\": " << config.runs << ",\n"
        << indent << "  \"run_once\": " << (config.run_once ? "true" : "false") << ",\n"
        << indent << "  \"success\": " << (benchmark_result.success ? "true" : "false") << ",\n"
        << indent << "  \"health_reason\": "
        << json_string(runtime_health_reason(config, engine_metadata, benchmark_result)) << ",\n"
        << indent << "  \"latency_mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "  \"latency_p95_ms\": " << benchmark_result.p95_ms << ",\n"
        << indent << "  \"latency_p99_ms\": " << benchmark_result.p99_ms << ",\n"
        << indent << "  \"fps\": " << benchmark_result.fps << ",\n"
        << indent << "  \"latency_budget_ms\": ";
    if (config.agent_latency_budget_ms > 0) {
        output << config.agent_latency_budget_ms;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "  \"latency_budget_exceeded\": " << (exceeded_latency_budget ? "true" : "false") << ",\n"
        << indent << "  \"deadline_missed\": " << (should_mark_deadline_missed(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "  \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << indent << "  \"jetson_clocks\": " << json_string(config.jetson_clocks) << ",\n"
        << indent << "  \"tegrastats_status\": " << json_string(tegrastats_summary.status) << ",\n"
        << indent << "  \"tegrastats_sample_count\": " << tegrastats_summary.sample_count << ",\n"
        << indent << "  \"thermal_memory_evidence_available\": " << ((tegrastats_summary.status == "parsed") ? "true" : "false") << ",\n"
        << indent << "  \"timeout_policy\": "
        << json_string(config.timeout_ms > 0 ? "latency_threshold" : "not_configured") << ",\n"
        << indent << "  \"timeout_budget_ms\": ";
    if (config.timeout_ms > 0) {
        output << config.timeout_ms;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "  \"timeout_observed\": " << (observed_timeout ? "true" : "false") << "\n"
        << indent << "}";
}

void write_runtime_operation_summary_json(
    std::ostream& output,
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const BenchmarkResult& benchmark_result,
    const TegrastatsSummary& tegrastats_summary,
    int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const std::vector<std::string> risk_labels =
        runtime_operation_risk_labels(config, engine_metadata, benchmark_result);
    const std::vector<std::string> evidence_gaps =
        runtime_operation_evidence_gaps(config, tegrastats_summary);
    output
        << "{\n"
        << indent << "  \"schema_version\": \"inferedge-runtime-operation-summary-v1\",\n"
        << indent << "  \"observation_scope\": \"single_runtime_result\",\n"
        << indent << "  \"decision_owner\": \"lab\",\n"
        << indent << "  \"scheduler_owner\": \"orchestrator\",\n"
        << indent << "  \"production_cancellation\": false,\n"
        << indent << "  \"health_status\": "
        << json_string(runtime_health_status(config, benchmark_result)) << ",\n"
        << indent << "  \"health_reason\": "
        << json_string(runtime_health_reason(config, engine_metadata, benchmark_result)) << ",\n"
        << indent << "  \"error_category\": "
        << json_string(runtime_error_category(config, benchmark_result)) << ",\n"
        << indent << "  \"retryable\": "
        << (runtime_retryable(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "  \"recommended_action\": "
        << json_string(runtime_operation_recommended_action(config, engine_metadata, benchmark_result)) << ",\n"
        << indent << "  \"risk_labels\": ";
    write_string_array_json(output, risk_labels);
    output
        << ",\n"
        << indent << "  \"evidence_gaps\": ";
    write_string_array_json(output, evidence_gaps);
    output
        << ",\n"
        << indent << "  \"timeout_observed\": "
        << (timeout_observed(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "  \"latency_budget_exceeded\": "
        << (latency_budget_exceeded(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "  \"deadline_missed\": "
        << (should_mark_deadline_missed(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "  \"thermal_memory_evidence_available\": "
        << ((tegrastats_summary.status == "parsed") ? "true" : "false") << "\n"
        << indent << "}";
}

void write_runtime_telemetry_json(
    std::ostream& output,
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const BenchmarkResult& benchmark_result,
    const TegrastatsSummary& tegrastats_summary,
    const std::string& timestamp,
    int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const bool has_tegrastats = tegrastats_summary.status == "parsed";
    output
        << "{\n"
        << indent << "  \"schema_version\": \"inferedge-runtime-telemetry-v1\",\n"
        << indent << "  \"evidence_role\": \"runtime_telemetry_seed\",\n"
        << indent << "  \"collection_mode\": \"single_result_export\",\n"
        << indent << "  \"source_result_schema_version\": \"inferedge-runtime-result-v1\",\n"
        << indent << "  \"telemetry_timestamp\": " << json_string(timestamp) << ",\n"
        << indent << "  \"execution_sequence_id\": 0,\n"
        << indent << "  \"sequence_scope\": \"single_runtime_result\",\n"
        << indent << "  \"engine_backend\": " << json_string(engine_metadata.backend) << ",\n"
        << indent << "  \"device\": " << json_string(config.device) << ",\n"
        << indent << "  \"input_mode\": " << json_string(config.input_mode()) << ",\n"
        << indent << "  \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << indent << "  \"jetson_clocks\": " << json_string(config.jetson_clocks) << ",\n"
        << indent << "  \"latency\": {\n"
        << indent << "    \"mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "    \"p95_ms\": " << benchmark_result.p95_ms << ",\n"
        << indent << "    \"p99_ms\": " << benchmark_result.p99_ms << ",\n"
        << indent << "    \"fps\": " << benchmark_result.fps << ",\n"
        << indent << "    \"inference_interval_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "    \"rolling_latency_mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "    \"rolling_latency_std_ms\": " << benchmark_result.std_ms << ",\n"
        << indent << "    \"sample_count\": " << benchmark_result.samples_ms.size() << "\n"
        << indent << "  },\n"
        << indent << "  \"resource\": {\n"
        << indent << "    \"ram_used_mb\": ";
    if (has_tegrastats) {
        output << tegrastats_summary.ram_used_mb_max;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "    \"ram_total_mb\": ";
    if (has_tegrastats) {
        output << tegrastats_summary.ram_total_mb;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "    \"gpu_memory_used_mb\": null,\n"
        << indent << "    \"max_temperature_c\": ";
    if (has_tegrastats) {
        output << tegrastats_summary.max_temp_c;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "    \"max_temperature_name\": "
        << json_string(has_tegrastats ? tegrastats_summary.max_temp_name : "") << ",\n"
        << indent << "    \"cpu_temperature_c\": null,\n"
        << indent << "    \"gpu_temperature_c\": null,\n"
        << indent << "    \"vdd_in_mw_avg\": ";
    if (has_tegrastats) {
        output << tegrastats_summary.vdd_in_mw_avg;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "    \"vdd_in_mw_max\": ";
    if (has_tegrastats) {
        output << tegrastats_summary.vdd_in_mw_max;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "    \"throttling_detected\": null,\n"
        << indent << "    \"telemetry_source\": "
        << json_string(has_tegrastats ? "tegrastats" : "not_available") << ",\n"
        << indent << "    \"tegrastats_status\": " << json_string(tegrastats_summary.status) << ",\n"
        << indent << "    \"tegrastats_sample_count\": " << tegrastats_summary.sample_count << "\n"
        << indent << "  },\n"
        << indent << "  \"operation\": {\n"
        << indent << "    \"queue_depth\": null,\n"
        << indent << "    \"runtime_uptime_sec\": null,\n"
        << indent << "    \"timeout_observed\": "
        << (timeout_observed(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "    \"latency_budget_exceeded\": "
        << (latency_budget_exceeded(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "    \"deadline_missed\": "
        << (should_mark_deadline_missed(config, benchmark_result) ? "true" : "false") << "\n"
        << indent << "  },\n"
        << indent << "  \"missing_fields\": ";
    write_string_array_json(output, runtime_telemetry_missing_fields(tegrastats_summary));
    output
        << ",\n"
        << indent << "  \"coverage\": ";
    write_runtime_telemetry_coverage_json(output, tegrastats_summary, indent_spaces + 2);
    output
        << ",\n"
        << indent << "  \"history_seed\": ";
    write_runtime_telemetry_history_seed_json(
        output,
        config,
        engine_metadata,
        benchmark_result,
        tegrastats_summary,
        timestamp,
        indent_spaces + 2);
    output
        << ",\n"
        << indent << "  \"production_monitoring\": false\n"
        << indent << "}";
}

void write_runtime_error_classification_json(
    std::ostream& output,
    const RuntimeConfig& config,
    const BenchmarkResult& benchmark_result,
    int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const bool observed_timeout = timeout_observed(config, benchmark_result);
    output
        << "{\n"
        << indent << "  \"schema_version\": \"inferedge-runtime-error-v1\",\n"
        << indent << "  \"status\": " << json_string((benchmark_result.success && !observed_timeout) ? "none" : "classified") << ",\n"
        << indent << "  \"category\": " << json_string(runtime_error_category(config, benchmark_result)) << ",\n"
        << indent << "  \"severity\": " << json_string(runtime_error_severity(config, benchmark_result)) << ",\n"
        << indent << "  \"message\": "
        << json_string(observed_timeout ? "mean latency exceeded configured timeout threshold" : (benchmark_result.success ? "" : benchmark_result.message)) << ",\n"
        << indent << "  \"observed_mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << indent << "  \"timeout_budget_ms\": ";
    if (config.timeout_ms > 0) {
        output << config.timeout_ms;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << indent << "  \"timeout_observed\": " << (observed_timeout ? "true" : "false") << ",\n"
        << indent << "  \"retryable\": " << (runtime_retryable(config, benchmark_result) ? "true" : "false") << ",\n"
        << indent << "  \"retry_hint\": " << json_string(runtime_retry_hint(config, benchmark_result)) << "\n"
        << indent << "}";
}

void write_runtime_events_json(
    std::ostream& output,
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const BenchmarkResult& benchmark_result,
    const TegrastatsSummary& tegrastats_summary,
    int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const std::string item_indent(static_cast<std::size_t>(indent_spaces + 2), ' ');
    const bool observed_timeout = timeout_observed(config, benchmark_result);
    const bool exceeded_latency_budget = latency_budget_exceeded(config, benchmark_result);
    int event_index = 0;

    output
        << "[\n"
        << item_indent << "{\n"
        << item_indent << "  \"schema_version\": \"inferedge-runtime-event-v1\",\n"
        << item_indent << "  \"event_index\": " << event_index++ << ",\n"
        << item_indent << "  \"type\": \"runtime_configured\",\n"
        << item_indent << "  \"status\": \"ok\",\n"
        << item_indent << "  \"engine_backend\": " << json_string(engine_metadata.backend) << ",\n"
        << item_indent << "  \"engine_available\": " << (engine_metadata.available ? "true" : "false") << ",\n"
        << item_indent << "  \"engine_status_message\": " << json_string(engine_metadata.status_message) << ",\n"
        << item_indent << "  \"device\": " << json_string(config.device) << ",\n"
        << item_indent << "  \"input_mode\": " << json_string(config.input_mode()) << ",\n"
        << item_indent << "  \"timeout_policy\": "
        << json_string(config.timeout_ms > 0 ? "latency_threshold" : "not_configured") << "\n"
        << item_indent << "},\n"
        << item_indent << "{\n"
        << item_indent << "  \"schema_version\": \"inferedge-runtime-event-v1\",\n"
        << item_indent << "  \"event_index\": " << event_index++ << ",\n"
        << item_indent << "  \"type\": \"benchmark_completed\",\n"
        << item_indent << "  \"status\": " << json_string(benchmark_result.status) << ",\n"
        << item_indent << "  \"success\": " << (benchmark_result.success ? "true" : "false") << ",\n"
        << item_indent << "  \"warmup\": " << benchmark_result.warmup_runs << ",\n"
        << item_indent << "  \"runs\": " << benchmark_result.timed_runs << ",\n"
        << item_indent << "  \"mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << item_indent << "  \"p95_ms\": " << benchmark_result.p95_ms << ",\n"
        << item_indent << "  \"p99_ms\": " << benchmark_result.p99_ms << ",\n"
        << item_indent << "  \"fps\": " << benchmark_result.fps << ",\n"
        << item_indent << "  \"latency_budget_ms\": ";
    if (config.agent_latency_budget_ms > 0) {
        output << config.agent_latency_budget_ms;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << item_indent << "  \"latency_budget_exceeded\": " << (exceeded_latency_budget ? "true" : "false") << ",\n"
        << item_indent << "  \"deadline_missed\": " << (should_mark_deadline_missed(config, benchmark_result) ? "true" : "false") << "\n"
        << item_indent << "},\n"
        << item_indent << "{\n"
        << item_indent << "  \"schema_version\": \"inferedge-runtime-event-v1\",\n"
        << item_indent << "  \"event_index\": " << event_index++ << ",\n"
        << item_indent << "  \"type\": \"runtime_error_classified\",\n"
        << item_indent << "  \"status\": " << json_string((benchmark_result.success && !observed_timeout) ? "none" : "classified") << ",\n"
        << item_indent << "  \"category\": " << json_string(runtime_error_category(config, benchmark_result)) << ",\n"
        << item_indent << "  \"severity\": " << json_string(runtime_error_severity(config, benchmark_result)) << ",\n"
        << item_indent << "  \"health_reason\": "
        << json_string(runtime_health_reason(config, engine_metadata, benchmark_result)) << ",\n"
        << item_indent << "  \"timeout_policy\": "
        << json_string(config.timeout_ms > 0 ? "latency_threshold" : "not_configured") << ",\n"
        << item_indent << "  \"timeout_budget_ms\": ";
    if (config.timeout_ms > 0) {
        output << config.timeout_ms;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << item_indent << "  \"observed_mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << item_indent << "  \"timeout_observed\": " << (observed_timeout ? "true" : "false") << ",\n"
        << item_indent << "  \"retryable\": " << (runtime_retryable(config, benchmark_result) ? "true" : "false") << ",\n"
        << item_indent << "  \"retry_hint\": " << json_string(runtime_retry_hint(config, benchmark_result)) << "\n"
        << item_indent << "},\n";

    if (!config.agent_manifest_path.empty()) {
        output
            << item_indent << "{\n"
            << item_indent << "  \"schema_version\": \"inferedge-runtime-event-v1\",\n"
            << item_indent << "  \"event_index\": " << event_index++ << ",\n"
            << item_indent << "  \"type\": \"agent_context_recorded\",\n"
            << item_indent << "  \"status\": " << json_string(config.agent_manifest_applied ? "ok" : "provided") << ",\n"
            << item_indent << "  \"agent_id\": " << json_string(config.agent_id) << ",\n"
            << item_indent << "  \"task_id\": " << json_string(config.agent_task_id) << ",\n"
            << item_indent << "  \"deadline_missed\": " << (should_mark_deadline_missed(config, benchmark_result) ? "true" : "false") << ",\n"
            << item_indent << "  \"fallback_used\": " << (config.agent_fallback_used ? "true" : "false") << "\n"
            << item_indent << "},\n";
    }

    output
        << item_indent << "{\n"
        << item_indent << "  \"schema_version\": \"inferedge-runtime-event-v1\",\n"
        << item_indent << "  \"event_index\": " << event_index++ << ",\n"
        << item_indent << "  \"type\": \"runtime_operation_summary_recorded\",\n"
        << item_indent << "  \"status\": " << json_string(runtime_health_status(config, benchmark_result)) << ",\n"
        << item_indent << "  \"health_reason\": "
        << json_string(runtime_health_reason(config, engine_metadata, benchmark_result)) << ",\n"
        << item_indent << "  \"recommended_action\": "
        << json_string(runtime_operation_recommended_action(config, engine_metadata, benchmark_result)) << ",\n"
        << item_indent << "  \"risk_labels\": ";
    write_string_array_json(output, runtime_operation_risk_labels(config, engine_metadata, benchmark_result));
    output
        << ",\n"
        << item_indent << "  \"evidence_gaps\": ";
    write_string_array_json(output, runtime_operation_evidence_gaps(config, tegrastats_summary));
    output
        << "\n"
        << item_indent << "},\n"
        << item_indent << "{\n"
        << item_indent << "  \"schema_version\": \"inferedge-runtime-event-v1\",\n"
        << item_indent << "  \"event_index\": " << event_index++ << ",\n"
        << item_indent << "  \"type\": \"runtime_telemetry_recorded\",\n"
        << item_indent << "  \"status\": \"recorded\",\n"
        << item_indent << "  \"schema\": \"inferedge-runtime-telemetry-v1\",\n"
        << item_indent << "  \"collection_mode\": \"single_result_export\",\n"
        << item_indent << "  \"telemetry_source\": "
        << json_string(tegrastats_summary.status == "parsed" ? "tegrastats" : "not_available") << ",\n"
        << item_indent << "  \"missing_fields\": ";
    write_string_array_json(output, runtime_telemetry_missing_fields(tegrastats_summary));
    output
        << ",\n"
        << item_indent << "  \"observed_field_count\": "
        << runtime_telemetry_observed_fields(tegrastats_summary).size() << ",\n"
        << item_indent << "  \"missing_field_count\": "
        << runtime_telemetry_missing_fields(tegrastats_summary).size() << "\n"
        << item_indent << "},\n"
        << item_indent << "{\n"
        << item_indent << "  \"schema_version\": \"inferedge-runtime-event-v1\",\n"
        << item_indent << "  \"event_index\": " << event_index++ << ",\n"
        << item_indent << "  \"type\": \"tegrastats_summary\",\n"
        << item_indent << "  \"status\": " << json_string(tegrastats_summary.status) << ",\n"
        << item_indent << "  \"sample_count\": " << tegrastats_summary.sample_count << ",\n"
        << item_indent << "  \"ram_used_mb_max\": " << tegrastats_summary.ram_used_mb_max << ",\n"
        << item_indent << "  \"max_temp_c\": " << tegrastats_summary.max_temp_c << ",\n"
        << item_indent << "  \"vdd_in_mw_max\": " << tegrastats_summary.vdd_in_mw_max << "\n"
        << item_indent << "}\n"
        << indent << "]";
}

void write_shape_json(std::ostream& output, const std::vector<int64_t>& shape) {
    output << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            output << ", ";
        }
        output << shape[i];
    }
    output << ']';
}

void write_double_vector_json(std::ostream& output, const std::vector<double>& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            output << ", ";
        }
        output << std::fixed << std::setprecision(6) << values[i];
    }
    output << ']';
}

void write_tensor_metadata_json(std::ostream& output, const std::vector<TensorMetadata>& tensors, int indent_spaces) {
    const std::string indent(static_cast<std::size_t>(indent_spaces), ' ');
    const std::string item_indent(static_cast<std::size_t>(indent_spaces + 2), ' ');

    output << "[";
    if (!tensors.empty()) {
        output << '\n';
    }

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const TensorMetadata& tensor = tensors[i];
        output
            << item_indent << "{\n"
            << item_indent << "  \"name\": " << json_string(tensor.name) << ",\n"
            << item_indent << "  \"element_type\": " << json_string(tensor.element_type) << ",\n"
            << item_indent << "  \"shape\": ";
        write_shape_json(output, tensor.shape);
        output << '\n' << item_indent << "}";
        if (i + 1 < tensors.size()) {
            output << ',';
        }
        output << '\n';
    }

    if (!tensors.empty()) {
        output << indent;
    }
    output << "]";
}

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    const std::filesystem::path parent_path = path.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output JSON file: " + path.string());
    }

    output << content;
}

}  // namespace

std::filesystem::path write_result_json(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const ModelMetadata& model_metadata,
    const BenchmarkResult& benchmark_result) {
    const std::string timestamp = current_utc_timestamp();
    const std::filesystem::path output_path = resolve_output_path(config, engine_metadata, timestamp);
    const std::filesystem::path latest_path("results/latest.json");
    const std::string output_mode = config.output_path == "auto" ? "auto" : "explicit";
    const std::string compare_key = make_compare_key(config);
    const std::string backend_key = make_backend_key(engine_metadata, config);
    const std::string compare_name = compare_model_name(config);
    const std::string compare_source = compare_model_source(config);
    const std::string precision = config.manifest_precision.empty() ? "fp32" : config.manifest_precision;
    const TegrastatsSummary tegrastats_summary = parse_tegrastats_log(config.tegrastats_log_path);

    std::ostringstream output;
    output << std::fixed << std::setprecision(6);
    const std::string model_name = std::filesystem::path(config.model_path).filename().string();
    output
        << "{\n"
        << "  \"schema_version\": \"inferedge-runtime-result-v1\",\n"
        << "  \"compare_key\": " << json_string(compare_key) << ",\n"
        << "  \"backend_key\": " << json_string(backend_key) << ",\n"
        << "  \"runtime_role\": \"runtime-result\",\n"
        << "  \"manifest_path\": " << json_string(config.manifest_path) << ",\n"
        << "  \"manifest_applied\": " << (config.manifest_applied ? "true" : "false") << ",\n"
        << "  \"model_name\": " << json_string(model_name) << ",\n"
        << "  \"model_path\": " << json_string(config.model_path) << ",\n"
        << "  \"engine_name\": " << json_string(engine_metadata.name) << ",\n"
        << "  \"engine_backend\": " << json_string(engine_metadata.backend) << ",\n"
        << "  \"device_name\": " << json_string(config.device) << ",\n"
        << "  \"batch\": " << config.batch << ",\n"
        << "  \"height\": " << config.height << ",\n"
        << "  \"width\": " << config.width << ",\n"
        << "  \"warmup\": " << config.warmup << ",\n"
        << "  \"runs\": " << config.runs << ",\n"
        << "  \"mean_ms\": " << benchmark_result.mean_ms << ",\n"
        << "  \"p50_ms\": " << benchmark_result.p50_ms << ",\n"
        << "  \"p95_ms\": " << benchmark_result.p95_ms << ",\n"
        << "  \"p99_ms\": " << benchmark_result.p99_ms << ",\n"
        << "  \"fps_value\": " << benchmark_result.fps << ",\n"
        << "  \"success\": " << (benchmark_result.success ? "true" : "false") << ",\n"
        << "  \"status\": " << json_string(benchmark_result.status) << ",\n"
        << "  \"model\": {\n"
        << "    \"path\": " << json_string(config.model_path) << ",\n"
        << "    \"name\": " << json_string(model_name) << "\n"
        << "  },\n"
        << "  \"engine\": {\n"
        << "    \"name\": " << json_string(engine_metadata.name) << ",\n"
        << "    \"backend\": " << json_string(engine_metadata.backend) << ",\n"
        << "    \"available\": " << (engine_metadata.available ? "true" : "false") << ",\n"
        << "    \"status_message\": " << json_string(engine_metadata.status_message) << "\n"
        << "  },\n"
        << "  \"device\": {\n"
        << "    \"name\": " << json_string(config.device) << "\n"
        << "  },\n"
        << "  \"precision\": " << json_string(precision) << ",\n"
        << "  \"run_config\": {\n"
        << "    \"batch\": " << config.batch << ",\n"
        << "    \"height\": " << config.height << ",\n"
        << "    \"width\": " << config.width << ",\n"
        << "    \"warmup\": " << config.warmup << ",\n"
        << "    \"runs\": " << config.runs << ",\n"
        << "    \"timeout_ms\": ";
    if (config.timeout_ms > 0) {
        output << config.timeout_ms;
    } else {
        output << "null";
    }
    output
        << ",\n"
        << "    \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << "    \"jetson_clocks\": " << json_string(config.jetson_clocks) << ",\n"
        << "    \"tegrastats_log_path\": " << json_string(config.tegrastats_log_path) << ",\n"
        << "    \"manifest_path\": " << json_string(config.manifest_path) << ",\n"
        << "    \"manifest_applied\": " << (config.manifest_applied ? "true" : "false") << "\n"
        << "  },\n"
        << "  \"latency_ms\": {\n"
        << "    \"mean\": " << benchmark_result.mean_ms << ",\n"
        << "    \"min\": " << benchmark_result.min_ms << ",\n"
        << "    \"max\": " << benchmark_result.max_ms << ",\n"
        << "    \"std\": " << benchmark_result.std_ms << ",\n"
        << "    \"p50\": " << benchmark_result.p50_ms << ",\n"
        << "    \"p90\": " << benchmark_result.p90_ms << ",\n"
        << "    \"p95\": " << benchmark_result.p95_ms << ",\n"
        << "    \"p99\": " << benchmark_result.p99_ms << ",\n"
        << "    \"samples\": ";
    write_double_vector_json(output, benchmark_result.samples_ms);
    output
        << "\n"
        << "  },\n"
        << "  \"fps\": " << benchmark_result.fps << ",\n"
        << "  \"benchmark\": {\n"
        << "    \"success\": " << (benchmark_result.success ? "true" : "false") << ",\n"
        << "    \"status\": " << json_string(benchmark_result.status) << ",\n"
        << "    \"message\": " << json_string(benchmark_result.message) << "\n"
        << "  },\n"
        << "  \"timestamp\": " << json_string(timestamp) << ",\n"
        << "  \"system\": {\n"
        << "    \"os\": " << json_string(system_os_name()) << ",\n"
        << "    \"compiler\": " << json_string(compiler_name()) << ",\n"
        << "    \"cpp_standard\": \"17\",\n"
        << "    \"jetson\": {\n"
        << "      \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << "      \"jetson_clocks\": " << json_string(config.jetson_clocks) << ",\n"
        << "      \"tegrastats_log_path\": " << json_string(config.tegrastats_log_path) << "\n"
        << "    }\n"
        << "  },\n"
        << "  \"jetson_evidence\": {\n"
        << "    \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << "    \"jetson_clocks\": " << json_string(config.jetson_clocks) << ",\n"
        << "    \"tegrastats_log_path\": " << json_string(config.tegrastats_log_path) << ",\n"
        << "    \"tegrastats_summary\": ";
    write_tegrastats_summary_json(output, tegrastats_summary, 4);
    output
        << "\n"
        << "  },\n"
        << "  \"runtime_health_snapshot\": ";
    write_runtime_health_snapshot_json(output, config, engine_metadata, benchmark_result, tegrastats_summary, 2);
    output
        << ",\n"
        << "  \"runtime_telemetry\": ";
    write_runtime_telemetry_json(output, config, engine_metadata, benchmark_result, tegrastats_summary, timestamp, 2);
    output
        << ",\n"
        << "  \"runtime_error_classification\": ";
    write_runtime_error_classification_json(output, config, benchmark_result, 2);
    output
        << ",\n"
        << "  \"runtime_events\": ";
    write_runtime_events_json(output, config, engine_metadata, benchmark_result, tegrastats_summary, 2);
    output
        << ",\n"
        << "  \"runtime_operation_summary\": ";
    write_runtime_operation_summary_json(output, config, engine_metadata, benchmark_result, tegrastats_summary, 2);
    if (!config.agent_manifest_path.empty()) {
        output
            << ",\n"
            << "  \"agent\": ";
        write_agent_task_json(output, config, benchmark_result, 2);
    }
    output
        << ",\n"
        << "  \"model_metadata\": {\n"
        << "    \"inputs\": ";
    write_tensor_metadata_json(output, model_metadata.inputs, 4);
    output << ",\n";
    output << "    \"outputs\": ";
    write_tensor_metadata_json(output, model_metadata.outputs, 4);
    output
        << "\n"
        << "  },\n"
        << "  \"extra\": {\n"
        << "    \"runtime\": \"inferedge-runtime\",\n"
        << "    \"json_export\": \"enabled\",\n"
        << "    \"output_mode\": " << json_string(output_mode) << ",\n"
        << "    \"latest_path\": \"results/latest.json\",\n"
        << "    \"manifest_recorded\": " << (config.manifest_path.empty() ? "false" : "true") << ",\n"
        << "    \"manifest_precision\": " << json_string(config.manifest_precision) << ",\n"
        << "    \"manifest_format\": " << json_string(config.manifest_format) << ",\n"
        << "    \"manifest_preset_name\": " << json_string(config.manifest_preset_name) << ",\n"
        << "    \"manifest_build_id\": " << json_string(config.manifest_build_id) << ",\n"
        << "    \"source_model_path\": " << json_string(config.manifest_source_model_path) << ",\n"
        << "    \"source_model_sha256\": " << json_string(config.manifest_source_sha256) << ",\n"
        << "    \"runtime_artifact_sha256\": " << json_string(config.manifest_artifact_sha256) << ",\n"
        << "    \"runtime_artifact_path\": " << json_string(config.model_path) << ",\n"
        << "    \"input_mode\": " << json_string(config.input_mode()) << ",\n"
        << "    \"input_path\": " << json_string(config.input_path) << ",\n"
        << "    \"input_preprocess\": " << json_string(config.input_preprocess()) << ",\n"
        << "    \"power_mode\": " << json_string(config.power_mode) << ",\n"
        << "    \"jetson_clocks\": " << json_string(config.jetson_clocks) << ",\n"
        << "    \"tegrastats_log_path\": " << json_string(config.tegrastats_log_path) << ",\n"
        << "    \"tegrastats_status\": " << json_string(tegrastats_summary.status) << ",\n"
        << "    \"agent_manifest_recorded\": " << (config.agent_manifest_path.empty() ? "false" : "true") << ",\n";
    if (!config.agent_manifest_path.empty()) {
        output
            << "    \"agent_manifest_path\": " << json_string(config.agent_manifest_path) << ",\n"
            << "    \"agent_id\": " << json_string(config.agent_id) << ",\n"
            << "    \"agent_task_id\": " << json_string(config.agent_task_id) << ",\n"
            << "    \"agent_type\": " << json_string(config.agent_type) << ",\n";
    }
    output
        << "    \"compare_ready\": true,\n"
        << "    \"compare_key\": " << json_string(compare_key) << ",\n"
        << "    \"backend_key\": " << json_string(backend_key) << ",\n"
        << "    \"compare_model_source\": " << json_string(compare_source) << ",\n"
        << "    \"compare_model_name\": " << json_string(compare_name) << "\n"
        << "  }\n"
        << "}\n";

    const std::string json = output.str();
    write_text_file(output_path, json);
    if (output_path.lexically_normal() != latest_path.lexically_normal()) {
        write_text_file(latest_path, json);
    }

    return output_path;
}

}  // namespace inferedge_runtime
