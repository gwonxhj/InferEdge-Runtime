#include "inferedge_runtime/engines/onnxruntime_engine.hpp"

#include <stdexcept>
#include <utility>

namespace inferedge_runtime {

OnnxRuntimeEngine::OnnxRuntimeEngine(RuntimeConfig config) : config_(std::move(config)) {}

EngineMetadata OnnxRuntimeEngine::metadata() const {
    EngineMetadata metadata;
    metadata.name = "onnxruntime";
    metadata.backend = "onnxruntime";
    metadata.device = config_.device;
    metadata.available = false;
    metadata.status_message =
        "ONNX Runtime backend is disabled. Reconfigure with -DINFEREDGE_ENABLE_ORT=ON after installing ONNX Runtime C++.";
    return metadata;
}

void OnnxRuntimeEngine::load_model(const std::string& model_path) {
    loaded_model_path_ = model_path;
}

void OnnxRuntimeEngine::run_once() {
    throw std::runtime_error("ONNX Runtime backend is not available in this build");
}

}  // namespace inferedge_runtime
