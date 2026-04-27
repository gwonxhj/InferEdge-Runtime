#pragma once

#include "inferedge_runtime/cli.hpp"

#include <string>

namespace inferedge_runtime {

struct ManifestConfig {
    std::string model_path;
    std::string model_name;
    std::string engine;
    std::string device;
    int batch = 0;
    int height = 0;
    int width = 0;
    std::string precision;
    std::string format;
    std::string artifact_sha256;
    std::string source_model_path;
    std::string source_model_sha256;
    std::string preset_name;
    std::string build_id;
    std::string handoff_kind;
};

ManifestConfig load_manifest_config(const std::string& path);
ManifestConfig load_forge_metadata_config(const std::string& path);
void validate_forge_handoff_config(const ManifestConfig& manifest, const std::string& path);
void apply_manifest_defaults(RuntimeConfig& config, const ManifestConfig& manifest);

}  // namespace inferedge_runtime
