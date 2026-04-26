#include "inferedge_runtime/engine.hpp"

#include "inferedge_runtime/engines/onnxruntime_engine.hpp"
#include "inferedge_runtime/engines/tensorrt_engine.hpp"

#include <stdexcept>

namespace inferedge_runtime {

std::unique_ptr<IInferenceEngine> create_engine(const RuntimeConfig& config) {
    if (config.engine == "onnxruntime" || config.engine == "ort") {
        return std::make_unique<OnnxRuntimeEngine>(config);
    }

    if (config.engine == "tensorrt" || config.engine == "trt") {
        return std::make_unique<TensorRTEngine>(config);
    }

    throw std::invalid_argument(
        "unsupported engine: " + config.engine + " (supported: onnxruntime, ort, tensorrt, trt)");
}

}  // namespace inferedge_runtime
