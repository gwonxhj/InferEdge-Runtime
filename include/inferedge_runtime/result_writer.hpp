#pragma once

#include "inferedge_runtime/cli.hpp"
#include "inferedge_runtime/engine.hpp"

#include <filesystem>

namespace inferedge_runtime {

std::filesystem::path write_result_json(
    const RuntimeConfig& config,
    const EngineMetadata& engine_metadata,
    const ModelMetadata& model_metadata,
    const BenchmarkResult& benchmark_result);

}  // namespace inferedge_runtime
