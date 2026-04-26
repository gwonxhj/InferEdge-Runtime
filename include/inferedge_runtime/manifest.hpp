#pragma once

#include "inferedge_runtime/cli.hpp"

#include <string>

namespace inferedge_runtime {

struct ManifestConfig {
    std::string model_path;
    std::string engine;
    std::string device;
    int batch = 0;
    int height = 0;
    int width = 0;
    std::string precision;
    std::string format;
};

ManifestConfig load_manifest_config(const std::string& path);
void apply_manifest_defaults(RuntimeConfig& config, const ManifestConfig& manifest);

}  // namespace inferedge_runtime
