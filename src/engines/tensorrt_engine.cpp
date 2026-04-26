#include "inferedge_runtime/engines/tensorrt_engine.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace inferedge_runtime {

TensorRTEngine::TensorRTEngine(RuntimeConfig config) : config_(std::move(config)) {}

TensorRTEngine::~TensorRTEngine() = default;

EngineMetadata TensorRTEngine::metadata() const {
    EngineMetadata metadata;
    metadata.name = "tensorrt";
    metadata.backend = "tensorrt";
    metadata.device = config_.device;
    metadata.available = false;
    metadata.status_message = "TensorRT backend is not implemented in this build. This stub prepares Jetson integration.";
    return metadata;
}

ModelMetadata TensorRTEngine::model_metadata() const {
    return {};
}

void TensorRTEngine::load_model(const std::string& model_path) {
    loaded_model_path_ = model_path;
}

void TensorRTEngine::run_once() {
    throw std::runtime_error("TensorRT backend is not implemented yet");
}

BenchmarkResult TensorRTEngine::benchmark(int warmup, int runs) {
    (void)warmup;
    (void)runs;
    throw std::runtime_error("TensorRT benchmark is not implemented yet");
}

}  // namespace inferedge_runtime
