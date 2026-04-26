#include "inferedge_runtime/engines/tensorrt_engine.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#ifdef INFEREDGE_TENSORRT_LINKED
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace inferedge_runtime {

TensorRTEngine::TensorRTEngine(RuntimeConfig config) : config_(std::move(config)) {}

TensorRTEngine::~TensorRTEngine() = default;

EngineMetadata TensorRTEngine::metadata() const {
    EngineMetadata metadata;
    metadata.name = "tensorrt";
    metadata.backend = "tensorrt";
    metadata.device = config_.device;
#ifdef INFEREDGE_TENSORRT_LINKED
    metadata.available = true;
    metadata.status_message = "TensorRT backend is linked. Engine deserialization is not implemented yet.";
#else
    metadata.available = false;
    metadata.status_message = "TensorRT backend is not implemented in this build. This stub prepares Jetson integration.";
#endif
    return metadata;
}

ModelMetadata TensorRTEngine::model_metadata() const {
    return {};
}

void TensorRTEngine::load_model(const std::string& model_path) {
    loaded_model_path_ = model_path;
}

void TensorRTEngine::run_once() {
#ifdef INFEREDGE_TENSORRT_LINKED
    throw std::runtime_error("TensorRT backend is linked, but engine execution is not implemented yet");
#else
    throw std::runtime_error("TensorRT backend is not enabled in this build");
#endif
}

BenchmarkResult TensorRTEngine::benchmark(int warmup, int runs) {
    (void)warmup;
    (void)runs;
#ifdef INFEREDGE_TENSORRT_LINKED
    throw std::runtime_error("TensorRT backend is linked, but benchmark execution is not implemented yet");
#else
    throw std::runtime_error("TensorRT backend is not enabled in this build");
#endif
}

}  // namespace inferedge_runtime
