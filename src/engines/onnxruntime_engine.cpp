#include "inferedge_runtime/engines/onnxruntime_engine.hpp"

#ifdef INFEREDGE_ORT_LINKED
#include <onnxruntime_cxx_api.h>
#endif

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inferedge_runtime {
namespace {

#ifdef INFEREDGE_ORT_LINKED
std::string element_type_to_string(ONNXTensorElementDataType element_type) {
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        default:
            return "unknown";
    }
}

std::string copy_allocated_name(Ort::AllocatedStringPtr name) {
    return name ? std::string{name.get()} : std::string{};
}

TensorMetadata read_tensor_metadata(const Ort::Session& session, std::size_t index, bool input, Ort::AllocatorWithDefaultOptions& allocator) {
    TensorMetadata metadata;
    metadata.name = input ? copy_allocated_name(session.GetInputNameAllocated(index, allocator))
                          : copy_allocated_name(session.GetOutputNameAllocated(index, allocator));

    const Ort::TypeInfo type_info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    const Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();

    metadata.element_type = element_type_to_string(tensor_info.GetElementType());
    metadata.shape = tensor_info.GetShape();
    for (int64_t& dim : metadata.shape) {
        if (dim < 0) {
            dim = -1;
        }
    }

    return metadata;
}
#endif

}  // namespace

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

ModelMetadata OnnxRuntimeEngine::model_metadata() const {
    return model_metadata_;
}

void OnnxRuntimeEngine::load_model(const std::string& model_path) {
    loaded_model_path_ = model_path;
    model_metadata_ = {};

#ifdef INFEREDGE_ORT_LINKED
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("model file not found: " + model_path);
    }

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "inferedge-runtime"};
    Ort::SessionOptions session_options;
    Ort::Session session{env, model_path.c_str(), session_options};
    Ort::AllocatorWithDefaultOptions allocator;

    const std::size_t input_count = session.GetInputCount();
    model_metadata_.inputs.reserve(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        model_metadata_.inputs.push_back(read_tensor_metadata(session, i, true, allocator));
    }

    const std::size_t output_count = session.GetOutputCount();
    model_metadata_.outputs.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        model_metadata_.outputs.push_back(read_tensor_metadata(session, i, false, allocator));
    }
#endif
}

void OnnxRuntimeEngine::run_once() {
#ifdef INFEREDGE_ORT_LINKED
    throw std::runtime_error("ONNX Runtime backend is linked, but inference execution is not implemented yet");
#else
    throw std::runtime_error("ONNX Runtime backend is not available in this build");
#endif
}

}  // namespace inferedge_runtime
