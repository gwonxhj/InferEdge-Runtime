#include "inferedge_runtime/manifest.hpp"

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
        throw std::runtime_error("manifest file not found: " + path);
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

ManifestConfig load_manifest_config(const std::string& path) {
    ManifestConfig manifest;
    if (path.empty()) {
        return manifest;
    }

    const std::string json = read_text_file(path);
    const std::string artifact = find_object_section(json, "artifact");
    const std::string runtime = find_object_section(json, "runtime");

    manifest.model_path = extract_json_string_value(artifact, "model_path");
    manifest.model_name = extract_json_string_value(artifact, "model_name");
    manifest.precision = extract_json_string_value(artifact, "precision");
    manifest.format = extract_json_string_value(artifact, "format");
    manifest.engine = extract_json_string_value(runtime, "engine");
    manifest.device = extract_json_string_value(runtime, "device");
    manifest.batch = extract_json_int_value(runtime, "batch");
    manifest.height = extract_json_int_value(runtime, "height");
    manifest.width = extract_json_int_value(runtime, "width");

    return manifest;
}

void apply_manifest_defaults(RuntimeConfig& config, const ManifestConfig& manifest) {
    config.manifest_model_name = manifest.model_name;

    if (!config.model_path_overridden && !manifest.model_path.empty()) {
        config.model_path = manifest.model_path;
    }
    if (!config.engine_overridden && !manifest.engine.empty()) {
        config.engine = manifest.engine;
    }
    if (!config.device_overridden && !manifest.device.empty()) {
        config.device = manifest.device;
    }
    if (!config.batch_overridden && manifest.batch > 0) {
        config.batch = manifest.batch;
    }
    if (!config.height_overridden && manifest.height > 0) {
        config.height = manifest.height;
    }
    if (!config.width_overridden && manifest.width > 0) {
        config.width = manifest.width;
    }
}

}  // namespace inferedge_runtime
