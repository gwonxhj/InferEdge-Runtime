#include "inferedge_runtime/lab_worker_request.hpp"

#include <cctype>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace inferedge_runtime {
namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Lab worker request file not found: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string find_object_section(const std::string& json, const std::string& key) {
    const std::string key_pattern = "\"" + key + "\"";
    const std::size_t key_pos = json.find(key_pattern);
    if (key_pos == std::string::npos) {
        return "";
    }

    const std::size_t colon_pos = json.find(':', key_pos + key_pattern.size());
    if (colon_pos == std::string::npos) {
        return "";
    }

    const std::size_t object_start = json.find('{', colon_pos + 1);
    if (object_start == std::string::npos) {
        return "";
    }

    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = object_start; i < json.size(); ++i) {
        const char ch = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                return json.substr(object_start, i - object_start + 1);
            }
        }
    }

    return "";
}

std::string extract_json_string_value(const std::string& section, const std::string& key) {
    const std::string key_pattern = "\"" + key + "\"";
    const std::size_t key_pos = section.find(key_pattern);
    if (key_pos == std::string::npos) {
        return "";
    }

    const std::size_t colon_pos = section.find(':', key_pos + key_pattern.size());
    if (colon_pos == std::string::npos) {
        return "";
    }

    std::size_t cursor = colon_pos + 1;
    while (cursor < section.size() && std::isspace(static_cast<unsigned char>(section[cursor]))) {
        ++cursor;
    }
    if (section.compare(cursor, 4, "null") == 0) {
        return "";
    }

    const std::size_t value_start = section.find('"', cursor);
    if (value_start == std::string::npos) {
        return "";
    }

    std::string value;
    bool escaped = false;
    for (std::size_t i = value_start + 1; i < section.size(); ++i) {
        const char ch = section[i];
        if (escaped) {
            switch (ch) {
                case '"':
                    value.push_back('"');
                    break;
                case '\\':
                    value.push_back('\\');
                    break;
                case 'n':
                    value.push_back('\n');
                    break;
                case 'r':
                    value.push_back('\r');
                    break;
                case 't':
                    value.push_back('\t');
                    break;
                default:
                    value.push_back(ch);
                    break;
            }
            escaped = false;
        } else if (ch == '\\') {
            escaped = true;
        } else if (ch == '"') {
            return value;
        } else {
            value.push_back(ch);
        }
    }

    return "";
}

int extract_json_int_value(const std::string& section, const std::string& key) {
    const std::string key_pattern = "\"" + key + "\"";
    const std::size_t key_pos = section.find(key_pattern);
    if (key_pos == std::string::npos) {
        return 0;
    }

    const std::size_t colon_pos = section.find(':', key_pos + key_pattern.size());
    if (colon_pos == std::string::npos) {
        return 0;
    }

    std::size_t value_start = colon_pos + 1;
    while (value_start < section.size() && std::isspace(static_cast<unsigned char>(section[value_start]))) {
        ++value_start;
    }

    std::size_t value_end = value_start;
    if (value_end < section.size() && (section[value_end] == '-' || section[value_end] == '+')) {
        ++value_end;
    }
    while (value_end < section.size() && std::isdigit(static_cast<unsigned char>(section[value_end]))) {
        ++value_end;
    }

    if (value_end == value_start) {
        return 0;
    }

    try {
        return std::stoi(section.substr(value_start, value_end - value_start));
    } catch (const std::exception&) {
        return 0;
    }
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const char ch : value) {
        switch (ch) {
            case '"':
                escaped << "\\\"";
                break;
            case '\\':
                escaped << "\\\\";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << ch;
                break;
        }
    }
    return escaped.str();
}

std::string json_string(const std::string& value) {
    if (value.empty()) {
        return "null";
    }
    return "\"" + json_escape(value) + "\"";
}

std::string current_utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_time{};
#if defined(_WIN32)
    gmtime_s(&utc_time, &now_time);
#else
    gmtime_r(&now_time, &utc_time);
#endif

    std::ostringstream stream;
    stream << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

void write_text_file(const std::filesystem::path& output_path, const std::string& content) {
    if (output_path.empty()) {
        throw std::runtime_error("worker response output path is required");
    }
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("failed to open worker response output path: " + output_path.string());
    }
    output << content;
}

}  // namespace

LabWorkerRequestConfig load_lab_worker_request_config(const std::string& path) {
    LabWorkerRequestConfig request;
    if (path.empty()) {
        return request;
    }

    const std::string json = read_text_file(path);
    const std::string input_summary = find_object_section(json, "input_summary");
    const std::string options = find_object_section(json, "options");
    const std::string provenance = find_object_section(options, "provenance");

    request.job_id = extract_json_string_value(json, "job_id");
    request.requested_at = extract_json_string_value(json, "requested_at");
    request.workflow = extract_json_string_value(input_summary, "workflow");
    request.model_path = extract_json_string_value(json, "model_path");
    request.artifact_path = extract_json_string_value(json, "artifact_path");
    request.metadata_path = extract_json_string_value(json, "metadata_path");
    request.manifest_path = extract_json_string_value(json, "manifest_path");
    request.source_model_sha256 = extract_json_string_value(provenance, "source_model_sha256");
    request.artifact_sha256 = extract_json_string_value(provenance, "artifact_sha256");
    request.artifact_type = extract_json_string_value(provenance, "artifact_type");
    request.preset_name = extract_json_string_value(provenance, "preset_name");
    request.build_id = extract_json_string_value(provenance, "build_id");
    request.engine = extract_json_string_value(options, "backend");
    request.device = extract_json_string_value(options, "target");
    request.precision = extract_json_string_value(options, "precision");
    request.batch = extract_json_int_value(options, "batch");
    request.height = extract_json_int_value(options, "height");
    request.width = extract_json_int_value(options, "width");
    request.warmup = extract_json_int_value(options, "warmup");
    request.runs = extract_json_int_value(options, "runs");
    if (request.warmup <= 0) {
        request.warmup = 5;
    }
    if (request.runs <= 0) {
        request.runs = 50;
    }

    validate_lab_worker_request_config(request, path);
    return request;
}

void validate_lab_worker_request_config(const LabWorkerRequestConfig& request, const std::string& path) {
    if (request.job_id.empty()) {
        throw std::runtime_error("Lab worker request is missing job_id: " + path);
    }
    if (request.requested_at.empty()) {
        throw std::runtime_error("Lab worker request is missing requested_at: " + path);
    }
    if (request.workflow != "analyze") {
        throw std::runtime_error("Lab worker request input_summary.workflow must be analyze: " + path);
    }
    if (request.model_path.empty() && request.artifact_path.empty()) {
        throw std::runtime_error("Lab worker request requires model_path or artifact_path: " + path);
    }
    if (request.engine.empty()) {
        throw std::runtime_error("Lab worker request options is missing backend: " + path);
    }
    if (request.device.empty()) {
        throw std::runtime_error("Lab worker request options is missing target: " + path);
    }
    if (request.precision.empty()) {
        throw std::runtime_error("Lab worker request options is missing precision: " + path);
    }
    if (request.batch <= 0) {
        throw std::runtime_error("Lab worker request options.batch must be positive: " + path);
    }
    if (request.height <= 0) {
        throw std::runtime_error("Lab worker request options.height must be positive: " + path);
    }
    if (request.width <= 0) {
        throw std::runtime_error("Lab worker request options.width must be positive: " + path);
    }
    if (request.warmup <= 0) {
        throw std::runtime_error("Lab worker request options.warmup must be positive: " + path);
    }
    if (request.runs <= 0) {
        throw std::runtime_error("Lab worker request options.runs must be positive: " + path);
    }
}

std::filesystem::path write_worker_response_dry_run(
    const LabWorkerRequestConfig& request,
    const std::string& status,
    const std::string& output_path,
    const std::string& error_message) {
    if (status != "completed" && status != "failed") {
        throw std::runtime_error("worker response status must be completed or failed");
    }

    validate_lab_worker_request_config(request, "Lab worker request dry-run export");

    const std::filesystem::path resolved_output_path(output_path);
    const std::string timestamp = current_utc_timestamp();
    const std::string runtime_model_path = request.artifact_path.empty() ? request.model_path : request.artifact_path;

    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    if (status == "completed") {
        json
            << "{\n"
            << "  \"job_id\": " << json_string(request.job_id) << ",\n"
            << "  \"status\": \"completed\",\n"
            << "  \"forge_metadata\": {\n"
            << "    \"backend\": " << json_string(request.engine) << ",\n"
            << "    \"target\": " << json_string(request.device) << ",\n"
            << "    \"precision\": " << json_string(request.precision) << ",\n"
            << "    \"artifact_path\": " << json_string(request.artifact_path) << ",\n"
            << "    \"source_model_path\": " << json_string(request.model_path) << "\n"
            << "  },\n"
            << "  \"runtime_result\": {\n"
            << "    \"model_path\": " << json_string(runtime_model_path) << ",\n"
            << "    \"engine_backend\": " << json_string(request.engine) << ",\n"
            << "    \"device_name\": " << json_string(request.device) << ",\n"
            << "    \"precision\": " << json_string(request.precision) << ",\n"
            << "    \"batch\": " << request.batch << ",\n"
            << "    \"height\": " << request.height << ",\n"
            << "    \"width\": " << request.width << ",\n"
            << "    \"mean_ms\": 0.000,\n"
            << "    \"p50_ms\": 0.000,\n"
            << "    \"p95_ms\": 0.000,\n"
            << "    \"p99_ms\": 0.000,\n"
            << "    \"latency_ms\": {\n"
            << "      \"mean\": 0.000,\n"
            << "      \"p50\": 0.000,\n"
            << "      \"p95\": 0.000,\n"
            << "      \"p99\": 0.000\n"
            << "    },\n"
            << "    \"run_config\": {\n"
            << "      \"batch\": " << request.batch << ",\n"
            << "      \"height\": " << request.height << ",\n"
            << "      \"width\": " << request.width << ",\n"
            << "      \"warmup\": " << request.warmup << ",\n"
            << "      \"runs\": " << request.runs << ",\n"
            << "      \"metadata_path\": " << json_string(request.metadata_path) << ",\n"
            << "      \"manifest_path\": " << json_string(request.manifest_path) << ",\n"
            << "      \"dry_run\": true\n"
            << "    },\n"
            << "    \"timestamp\": " << json_string(timestamp) << ",\n"
            << "    \"extra\": {\n"
            << "      \"runtime_artifact_path\": " << json_string(runtime_model_path) << ",\n"
            << "      \"source_model_path\": " << json_string(request.model_path) << ",\n"
            << "      \"runtime_artifact_sha256\": " << json_string(request.artifact_sha256) << ",\n"
            << "      \"source_model_sha256\": " << json_string(request.source_model_sha256) << ",\n"
            << "      \"artifact_type\": " << json_string(request.artifact_type) << ",\n"
            << "      \"preset_name\": " << json_string(request.preset_name) << ",\n"
            << "      \"build_id\": " << json_string(request.build_id) << ",\n"
            << "      \"dry_run\": true,\n"
            << "      \"worker_response_mode\": \"dry_run\"\n"
            << "    }\n"
            << "  },\n"
            << "  \"completed_at\": " << json_string(timestamp) << "\n"
            << "}\n";
    } else {
        const std::string message = error_message.empty() ? "Runtime worker dry-run failure." : error_message;
        json
            << "{\n"
            << "  \"job_id\": " << json_string(request.job_id) << ",\n"
            << "  \"status\": \"failed\",\n"
            << "  \"error\": {\n"
            << "    \"code\": \"runtime_worker_dry_run_failed\",\n"
            << "    \"message\": " << json_string(message) << ",\n"
            << "    \"stage\": \"runtime\"\n"
            << "  },\n"
            << "  \"failed_at\": " << json_string(timestamp) << "\n"
            << "}\n";
    }

    write_text_file(resolved_output_path, json.str());
    return resolved_output_path;
}

}  // namespace inferedge_runtime
