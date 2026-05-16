#include "inferedge_runtime/agent_manifest.hpp"

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
        throw std::runtime_error("agent manifest file not found: " + path);
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

    const std::size_t value_start = section.find('"', colon_pos + 1);
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

AgentManifestConfig load_agent_manifest_config(const std::string& path) {
    AgentManifestConfig manifest;
    if (path.empty()) {
        return manifest;
    }

    const std::string json = read_text_file(path);
    const std::string fallback_policy = find_object_section(json, "fallback_policy");
    const std::string lab_compat = find_object_section(json, "lab_compat");

    manifest.schema_version = extract_json_string_value(json, "schema_version");
    manifest.agent_id = extract_json_string_value(json, "agent_id");
    manifest.agent_type = extract_json_string_value(json, "agent_type");
    manifest.priority = extract_json_int_value(json, "priority");
    manifest.latency_budget_ms = extract_json_int_value(json, "latency_budget_ms");
    manifest.deadline_ms = extract_json_int_value(json, "deadline_ms");
    manifest.input_type = extract_json_string_value(json, "input_type");
    manifest.output_type = extract_json_string_value(json, "output_type");
    manifest.required_backend = extract_json_string_value(json, "required_backend");
    manifest.device_target = extract_json_string_value(json, "device_target");
    manifest.precision = extract_json_string_value(json, "precision");
    manifest.runtime_artifact_path = extract_json_string_value(json, "runtime_artifact_path");
    if (manifest.runtime_artifact_path.empty()) {
        manifest.runtime_artifact_path = extract_json_string_value(lab_compat, "runtime_artifact_path");
    }
    manifest.fallback_policy_mode = extract_json_string_value(fallback_policy, "mode");
    manifest.telemetry_contract_version = extract_json_string_value(json, "telemetry_contract_version");

    validate_agent_manifest_config(manifest, path);
    return manifest;
}

void validate_agent_manifest_config(const AgentManifestConfig& manifest, const std::string& path) {
    if (manifest.schema_version != "inferedge-agent-manifest-v1") {
        throw std::runtime_error(
            "agent manifest schema_version must be inferedge-agent-manifest-v1: " + path);
    }
    if (manifest.agent_id.empty()) {
        throw std::runtime_error("agent manifest is missing agent_id: " + path);
    }
    if (manifest.agent_type.empty()) {
        throw std::runtime_error("agent manifest is missing agent_type: " + path);
    }
    if (manifest.priority < 0) {
        throw std::runtime_error("agent manifest priority must be non-negative: " + path);
    }
    if (manifest.latency_budget_ms <= 0) {
        throw std::runtime_error("agent manifest latency_budget_ms must be positive: " + path);
    }
}

void apply_agent_manifest_defaults(RuntimeConfig& config, const AgentManifestConfig& manifest) {
    config.agent_schema_version = manifest.schema_version;
    config.agent_id = manifest.agent_id;
    config.agent_type = manifest.agent_type;
    config.agent_input_type = manifest.input_type;
    config.agent_output_type = manifest.output_type;
    config.agent_required_backend = manifest.required_backend;
    config.agent_device_target = manifest.device_target;
    config.agent_precision = manifest.precision;
    config.agent_fallback_policy_mode = manifest.fallback_policy_mode;
    config.agent_runtime_artifact_path = manifest.runtime_artifact_path;
    config.agent_telemetry_contract_version = manifest.telemetry_contract_version;
    config.agent_scheduled_priority = manifest.priority;
    config.agent_latency_budget_ms = manifest.latency_budget_ms;
    config.agent_deadline_ms = manifest.deadline_ms;

    if (config.agent_task_id.empty()) {
        config.agent_task_id = "task_" + manifest.agent_id;
    }
    if (!config.model_path_overridden && !manifest.runtime_artifact_path.empty()) {
        config.model_path = manifest.runtime_artifact_path;
    }
    if (!config.engine_overridden && !manifest.required_backend.empty()) {
        config.engine = manifest.required_backend;
    }
    if (!config.device_overridden && !manifest.device_target.empty()) {
        config.device = manifest.device_target;
    }
    if (config.manifest_precision.empty() && !manifest.precision.empty()) {
        config.manifest_precision = manifest.precision;
    }
}

}  // namespace inferedge_runtime
