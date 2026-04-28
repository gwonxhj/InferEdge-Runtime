#include "inferedge_runtime/lab_worker_request.hpp"

#include <cctype>
#include <fstream>
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

}  // namespace

LabWorkerRequestConfig load_lab_worker_request_config(const std::string& path) {
    LabWorkerRequestConfig request;
    if (path.empty()) {
        return request;
    }

    const std::string json = read_text_file(path);
    const std::string input_summary = find_object_section(json, "input_summary");
    const std::string options = find_object_section(json, "options");

    request.job_id = extract_json_string_value(json, "job_id");
    request.requested_at = extract_json_string_value(json, "requested_at");
    request.workflow = extract_json_string_value(input_summary, "workflow");
    request.model_path = extract_json_string_value(json, "model_path");
    request.artifact_path = extract_json_string_value(json, "artifact_path");
    request.metadata_path = extract_json_string_value(json, "metadata_path");
    request.manifest_path = extract_json_string_value(json, "manifest_path");
    request.engine = extract_json_string_value(options, "backend");
    request.device = extract_json_string_value(options, "target");
    request.precision = extract_json_string_value(options, "precision");
    request.batch = extract_json_int_value(options, "batch");
    request.height = extract_json_int_value(options, "height");
    request.width = extract_json_int_value(options, "width");
    request.warmup = extract_json_int_value(options, "warmup");
    request.runs = extract_json_int_value(options, "runs");

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

}  // namespace inferedge_runtime
