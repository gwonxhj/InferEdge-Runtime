#include "inferedge_runtime/engines/onnxruntime_engine.hpp"

#ifdef INFEREDGE_ORT_LINKED
#include <onnxruntime_cxx_api.h>
#endif

#include <stdexcept>
#include <utility>

namespace inferedge_runtime {

OnnxRuntimeEngine::OnnxRuntimeEngine(RuntimeConfig config) : config_(std::move(config)) {}

EngineMetadata OnnxRuntimeEngine::metadata() const {
    EngineMetadata metadata;
    metadata.name = "onnxruntime";
    metadata.backend = "onnxruntime";
    metadata.device = config_.device;
#ifdef INFEREDGE_ORT_LINKED
    metadata.available = true;
    metadata.status_message = "ONNX Runtime backend is linked. Inference execution is not implemented yet.";
#else
    metadata.available = false;
    metadata.status_message =
        "ONNX Runtime backend is disabled. Reconfigure with -DINFEREDGE_ENABLE_ORT=ON after installing ONNX Runtime C++.";
#endif
    return metadata;
}

void OnnxRuntimeEngine::load_model(const std::string& model_path) {
    loaded_model_path_ = model_path;
}

void OnnxRuntimeEngine::run_once() {
#ifdef INFEREDGE_ORT_LINKED
    throw std::runtime_error("ONNX Runtime backend is linked, but inference execution is not implemented yet");
#else
    throw std::runtime_error("ONNX Runtime backend is not available in this build");
#endif
}

}  // namespace inferedge_runtime
